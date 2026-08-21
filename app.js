import {
  FaceLandmarker,
  GestureRecognizer,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const GESTURE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

const THRESHOLDS = {
  smile: 0.4,
  jawOpenTongue: 0.12,
  mouthLowerDownTongue: 0.3,
  browRaise: 0.5,
  gestureScore: 0.6,
  headTurnRatioHigh: 0.65,
  headTurnRatioLow: 0.35,
  // Shocked requires the mouth open this much, on top of the hands-on-head
  // pose below.
  jawOpenShocked: 0.35,
  // How far outside the face-mesh bounding box a hand still counts as
  // "on the head" (temples/hairline aren't covered by face landmarks).
  headTouchPadX: 0.6,
  headTouchPadYTop: 0.5,
  headTouchPadYBottom: 0.2,
  // Max distance (as a fraction of face width) from the fingertip to the
  // mouth center to count as "touching the mouth".
  thinkMouthMaxDist: 0.22,
  // Minimum upward eye-gaze blendshape score to count as "looking up".
  eyeLookUp: 0.4,
  // Max distance between the two hands' thumb tips / index tips (as a
  // multiple of each hand's own wrist-to-palm size) to count as a heart.
  heartThumbMaxDist: 1.0,
  heartIndexMaxDist: 1.0,
  // How far below the finger's own PIP joint the tip must reach the wrist
  // distance to count as "extended" (margin against noisy landmarks).
  fingerExtendMargin: 1.1,
  // How far a fist's wrist can sit below the face's bottom edge and still
  // count as raised (bent-elbow flex pose).
  flexWristYPad: 0.3,
};

const REQUIRED_CONSECUTIVE_FRAMES = 2;

// Landmark indices from MediaPipe's canonical 468-point face mesh topology.
const NOSE_TIP = 1;
const LEFT_CHEEK = 234; // subject's anatomical left
const RIGHT_CHEEK = 454; // subject's anatomical right
const MOUTH_UPPER = 13; // upper inner lip center
const MOUTH_LOWER = 14; // lower inner lip center

// MediaPipe Hand Landmarker topology: landmark 9 is the middle-finger MCP,
// roughly the center of the palm. Landmark 0 is the wrist, 4 the thumb tip,
// 8 the index fingertip.
const PALM_CENTER = 9;
const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_TIP = 8;

// Tip/PIP joint pairs per finger, used to test whether a finger is extended
// (tip farther from the wrist than its own PIP joint) or curled.
const FINGER_JOINTS = {
  index: { tip: 8, pip: 6 },
  middle: { tip: 12, pip: 10 },
  ring: { tip: 16, pip: 14 },
  pinky: { tip: 20, pip: 18 },
};

// Gesture Recognizer's built-in category names -> our state keys.
const GESTURE_STATE_MAP = {
  Thumb_Up: "thumbs_up",
  Thumb_Down: "thumbs_down",
  Pointing_Up: "one_finger",
  Victory: "two_finger",
};

const EXPRESSION_IMAGES = {
  middle_finger: "assets/hamster_middle_finger.jpg",
  heart_shape: "assets/hamster_heart_shape.jpg",
  think: "assets/hamster_think.jpg",
  flexing: "assets/hamster_flexing.jpg",
  thumbs_up: "assets/hamster_thumbs_up.jpg",
  thumbs_down: "assets/hamster_thumbs_down.jpg",
  one_finger: "assets/hamster_one_finger.jpg",
  two_finger: "assets/hamster_two_finger.jpg",
  shocked: "assets/hamster_shocked.jpg",
  tongue_out: "assets/hamster_tongue_out.jpg",
  smile: "assets/hamster_smile.jpg",
  eyebrow_raise: "assets/hamster_eyebrow_raise.jpg",
  turn_left: "assets/hamster_turn_left.jpg",
  turn_right: "assets/hamster_turn_right.jpg",
  neutral: "assets/hamster_neutral.jpg",
};

const EXPRESSION_LABELS = {
  middle_finger: "Middle Finger 🖕",
  heart_shape: "Heart Shape 🫶",
  think: "Thinking 🤔",
  flexing: "Flexing 💪",
  thumbs_up: "Thumbs Up 👍",
  thumbs_down: "Thumbs Down 👎",
  one_finger: "One Finger ☝️",
  two_finger: "Two Finger ✌️",
  shocked: "Shocked 😮",
  tongue_out: "Tongue Out 😛",
  smile: "Smile 😊",
  eyebrow_raise: "Eyebrow Raise 🤨",
  turn_left: "Turned Left ⬅️",
  turn_right: "Turned Right ➡️",
  neutral: "Neutral 😐",
};

const video = document.getElementById("webcam");
const hamsterImage = document.getElementById("hamsterImage");
const expressionLabel = document.getElementById("expressionLabel");
const cameraError = document.getElementById("cameraError");
const cameraErrorText = document.getElementById("cameraErrorText");
const loadingOverlay = document.getElementById("loadingOverlay");
const heartBurst = document.getElementById("heartBurst");
const debugPanel = document.getElementById("debugPanel");

// Some Android browsers hand lower-level frame-reading APIs the raw,
// unrotated camera sensor buffer even though the <video> element itself
// displays it correctly-oriented (the browser applies that correction only
// at the compositor level). Feeding MediaPipe the <video> element directly
// can then mean it's looking at a sideways face despite the on-screen
// preview looking normal. Drawing each frame onto a canvas first forces the
// same orientation-correction the browser uses for on-screen rendering.
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });

let debugFrameCount = 0;

function updateDebugPanel(faceResult, gestureResult, expression, error) {
  if (!debugPanel) return;
  debugFrameCount += 1;

  if (error) {
    debugPanel.textContent = `frame ${debugFrameCount} | ERROR: ${error.message}`;
    return;
  }

  const hasFace =
    faceResult && faceResult.faceBlendshapes && faceResult.faceBlendshapes.length > 0;
  const categories = hasFace ? faceResult.faceBlendshapes[0].categories : [];
  const jawOpen = blendshapeScore(categories, "jawOpen").toFixed(2);
  const smile = (
    (blendshapeScore(categories, "mouthSmileLeft") +
      blendshapeScore(categories, "mouthSmileRight")) /
    2
  ).toFixed(2);
  const browRaise = blendshapeScore(categories, "browInnerUp").toFixed(2);
  const handCount = (gestureResult && gestureResult.landmarks && gestureResult.landmarks.length) || 0;

  debugPanel.textContent =
    `frame ${debugFrameCount} | face: ${hasFace} | hands: ${handCount} | ` +
    `jawOpen: ${jawOpen} smile: ${smile} browRaise: ${browRaise} | state: ${expression}`;
}

const HEART_EMOJIS = ["💖", "💗", "💕", "❤️"];
const HEART_COUNT = 14;

function spawnHeartBurst() {
  for (let i = 0; i < HEART_COUNT; i++) {
    const heart = document.createElement("span");
    heart.textContent = HEART_EMOJIS[Math.floor(Math.random() * HEART_EMOJIS.length)];

    const size = 1.5 + Math.random() * 2;
    const duration = 2.5 + Math.random() * 1.5;
    const delay = Math.random() * 0.6;

    heart.style.setProperty("--x", `${Math.random() * 100}vw`);
    heart.style.setProperty("--size", `${size}rem`);
    heart.style.setProperty("--duration", `${duration}s`);
    heart.style.setProperty("--delay", `${delay}s`);
    heart.style.setProperty("--rot", `${(Math.random() * 40 - 20).toFixed(0)}deg`);

    const cleanup = () => heart.remove();
    heart.addEventListener("animationend", cleanup);
    setTimeout(cleanup, (duration + delay) * 1000 + 300);

    heartBurst.appendChild(heart);
  }
}

let faceLandmarker = null;
let gestureRecognizer = null;
let filesetResolver = null;
let displayedExpression = "neutral";
let candidateExpression = "neutral";
let candidateStreak = 0;

function showCameraError(message) {
  cameraErrorText.textContent = message;
  cameraError.classList.remove("hidden");
}

function blendshapeScore(blendshapes, name) {
  const match = blendshapes.find((b) => b.categoryName === name);
  return match ? match.score : 0;
}

function classifyGesture(gestureResult) {
  if (!gestureResult || !gestureResult.gestures || gestureResult.gestures.length === 0) {
    return null;
  }
  const topGesture = gestureResult.gestures[0][0];
  if (!topGesture || topGesture.score < THRESHOLDS.gestureScore) return null;
  return GESTURE_STATE_MAP[topGesture.categoryName] || null;
}

function getFaceBoundingBox(faceLandmarks) {
  let minX = 1,
    maxX = 0,
    minY = 1,
    maxY = 0;
  for (const point of faceLandmarks) {
    if (point.x < minX) minX = point.x;
    if (point.x > maxX) maxX = point.x;
    if (point.y < minY) minY = point.y;
    if (point.y > maxY) maxY = point.y;
  }
  return { minX, maxX, minY, maxY, width: maxX - minX, height: maxY - minY };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function isHandsOnHead(hands, faceBox) {
  if (hands.length < 2) return false;

  const headMinX = faceBox.minX - faceBox.width * THRESHOLDS.headTouchPadX;
  const headMaxX = faceBox.maxX + faceBox.width * THRESHOLDS.headTouchPadX;
  const headMinY = faceBox.minY - faceBox.height * THRESHOLDS.headTouchPadYTop;
  const headMaxY = faceBox.maxY + faceBox.height * THRESHOLDS.headTouchPadYBottom;

  let handsNearHead = 0;
  for (const hand of hands) {
    const palm = hand[PALM_CENTER];
    if (
      palm.x >= headMinX &&
      palm.x <= headMaxX &&
      palm.y >= headMinY &&
      palm.y <= headMaxY
    ) {
      handsNearHead += 1;
    }
  }
  return handsNearHead >= 2;
}

function isHeartShape(hands) {
  if (hands.length < 2) return false;
  const [h1, h2] = hands;

  // Scale reference from the hands' own size, so this works regardless of
  // how far the heart is held from the face.
  const handScale =
    (distance(h1[WRIST], h1[PALM_CENTER]) + distance(h2[WRIST], h2[PALM_CENTER])) / 2;

  const thumbDist = distance(h1[THUMB_TIP], h2[THUMB_TIP]);
  const indexDist = distance(h1[INDEX_TIP], h2[INDEX_TIP]);
  const thumbsTouching = thumbDist < handScale * THRESHOLDS.heartThumbMaxDist;
  const indexTipsTouching = indexDist < handScale * THRESHOLDS.heartIndexMaxDist;
  if (!thumbsTouching || !indexTipsTouching) return false;

  // Classic finger-heart shape: thumbs cross at the bottom point, index
  // fingertips meet at the top.
  const thumbMidY = (h1[THUMB_TIP].y + h2[THUMB_TIP].y) / 2;
  const indexMidY = (h1[INDEX_TIP].y + h2[INDEX_TIP].y) / 2;
  return thumbMidY > indexMidY;
}

function isFingerExtended(hand, tipIdx, pipIdx) {
  const wrist = hand[WRIST];
  return (
    distance(wrist, hand[tipIdx]) >
    distance(wrist, hand[pipIdx]) * THRESHOLDS.fingerExtendMargin
  );
}

function getExtendedFingers(hand) {
  const extended = {};
  for (const [name, joints] of Object.entries(FINGER_JOINTS)) {
    extended[name] = isFingerExtended(hand, joints.tip, joints.pip);
  }
  return extended;
}

function isMiddleFingerPose(hand) {
  const f = getExtendedFingers(hand);
  return f.middle && !f.index && !f.ring && !f.pinky;
}

function isFistPose(hand) {
  const f = getExtendedFingers(hand);
  return !f.index && !f.middle && !f.ring && !f.pinky;
}

function isThinkingPose(hand, faceLandmarks, faceBox, blendshapes) {
  const tip = hand[INDEX_TIP];
  const mouthCenter = {
    x: (faceLandmarks[MOUTH_UPPER].x + faceLandmarks[MOUTH_LOWER].x) / 2,
    y: (faceLandmarks[MOUTH_UPPER].y + faceLandmarks[MOUTH_LOWER].y) / 2,
  };
  const nearMouth = distance(tip, mouthCenter) < faceBox.width * THRESHOLDS.thinkMouthMaxDist;
  if (!nearMouth) return false;

  const lookUpScore =
    (blendshapeScore(blendshapes, "eyeLookUpLeft") +
      blendshapeScore(blendshapes, "eyeLookUpRight")) /
    2;
  return lookUpScore > THRESHOLDS.eyeLookUp;
}

function isFlexingPose(hand, faceBox) {
  if (!isFistPose(hand)) return false;
  const wrist = hand[WRIST];
  const wristOutsideFace = wrist.x < faceBox.minX || wrist.x > faceBox.maxX;
  const wristRaised = wrist.y < faceBox.maxY + faceBox.height * THRESHOLDS.flexWristYPad;
  return wristOutsideFace && wristRaised;
}

function classifyHandPose(hands, faceLandmarks, faceBox, blendshapes) {
  for (const hand of hands) {
    if (isMiddleFingerPose(hand)) return "middle_finger";
  }
  for (const hand of hands) {
    if (isThinkingPose(hand, faceLandmarks, faceBox, blendshapes)) return "think";
  }
  for (const hand of hands) {
    if (isFlexingPose(hand, faceBox)) return "flexing";
  }
  return null;
}

function classifyHeadTurn(landmarks) {
  const nose = landmarks[NOSE_TIP];
  const leftCheek = landmarks[LEFT_CHEEK];
  const rightCheek = landmarks[RIGHT_CHEEK];
  const span = leftCheek.x - rightCheek.x;
  if (Math.abs(span) < 1e-6) return null;

  // ratio ~0.5 = facing forward; ->1 as nose approaches the left-cheek
  // landmark (subject turned toward their own left), ->0 toward their right.
  // Confirmed backwards against the mirrored on-screen video, so the two
  // outcomes below are swapped relative to the raw geometric ratio.
  const ratio = (nose.x - rightCheek.x) / span;
  if (ratio > THRESHOLDS.headTurnRatioHigh) return "turn_right";
  if (ratio < THRESHOLDS.headTurnRatioLow) return "turn_left";
  return null;
}

function classifyFace(blendshapes, landmarks) {
  const jawOpenScore = blendshapeScore(blendshapes, "jawOpen");

  const mouthLowerDownScore =
    (blendshapeScore(blendshapes, "mouthLowerDownLeft") +
      blendshapeScore(blendshapes, "mouthLowerDownRight")) /
    2;
  if (
    mouthLowerDownScore > THRESHOLDS.mouthLowerDownTongue &&
    jawOpenScore > THRESHOLDS.jawOpenTongue
  ) {
    return "tongue_out";
  }

  const smileScore =
    (blendshapeScore(blendshapes, "mouthSmileLeft") +
      blendshapeScore(blendshapes, "mouthSmileRight")) /
    2;
  if (smileScore > THRESHOLDS.smile) return "smile";

  const browRaiseScore = Math.max(
    blendshapeScore(blendshapes, "browInnerUp"),
    (blendshapeScore(blendshapes, "browOuterUpLeft") +
      blendshapeScore(blendshapes, "browOuterUpRight")) /
      2
  );
  if (browRaiseScore > THRESHOLDS.browRaise) return "eyebrow_raise";

  const headTurn = classifyHeadTurn(landmarks);
  if (headTurn) return headTurn;

  return "neutral";
}

function classifyState(faceResult, gestureResult) {
  const hasFace =
    faceResult.faceBlendshapes &&
    faceResult.faceBlendshapes.length > 0 &&
    faceResult.faceLandmarks.length > 0;
  const landmarks = hasFace ? faceResult.faceLandmarks[0] : null;
  const faceBox = hasFace ? getFaceBoundingBox(landmarks) : null;
  const categories = hasFace ? faceResult.faceBlendshapes[0].categories : null;
  const hands = (gestureResult && gestureResult.landmarks) || [];

  if (hasFace) {
    const jawOpenScore = blendshapeScore(categories, "jawOpen");
    if (isHandsOnHead(hands, faceBox) && jawOpenScore > THRESHOLDS.jawOpenShocked) {
      return "shocked";
    }
    if (isHeartShape(hands)) return "heart_shape";
  }

  // The gesture recognizer's own trained classifier is more reliable than
  // our hand-rolled pose checks below, so let it claim thumbs up/down,
  // pointing up, and victory first (e.g. a thumbs-down fist would otherwise
  // also satisfy our fist-based flexing check).
  const gestureState = classifyGesture(gestureResult);
  if (gestureState) return gestureState;

  if (hasFace) {
    const handPose = classifyHandPose(hands, landmarks, faceBox, categories);
    if (handPose) return handPose;

    return classifyFace(categories, landmarks);
  }

  return "neutral";
}

function updateDisplayedExpression(nextExpression) {
  if (nextExpression === candidateExpression) {
    candidateStreak += 1;
  } else {
    candidateExpression = nextExpression;
    candidateStreak = 1;
  }

  if (
    candidateStreak >= REQUIRED_CONSECUTIVE_FRAMES &&
    candidateExpression !== displayedExpression
  ) {
    displayedExpression = candidateExpression;
    hamsterImage.src = EXPRESSION_IMAGES[displayedExpression];
    hamsterImage.alt = `Hamster reaction: ${displayedExpression}`;
    expressionLabel.textContent = EXPRESSION_LABELS[displayedExpression];

    hamsterImage.classList.remove("pop");
    void hamsterImage.offsetWidth; // restart the CSS animation
    hamsterImage.classList.add("pop");

    if (displayedExpression === "heart_shape") {
      spawnHeartBurst();
    }
  }
}

async function createFaceLandmarker(filesetResolver, delegate) {
  return FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: FACE_MODEL_URL,
      delegate,
    },
    outputFaceBlendshapes: true,
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

async function createGestureRecognizer(filesetResolver, delegate) {
  return GestureRecognizer.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: GESTURE_MODEL_URL,
      delegate,
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

async function initModels() {
  filesetResolver = await FilesetResolver.forVisionTasks(WASM_BASE);

  // The GPU delegate is unreliable across mobile browsers: on some (e.g.
  // Samsung Internet) it doesn't throw at all, it just silently feeds the
  // model garbage/empty texture data, so detection runs forever without
  // ever finding a face and without any visible error. CPU is slower per
  // frame but far more broadly correct, so it's the default everywhere;
  // GPU is only attempted as a fallback if CPU creation itself fails.
  try {
    faceLandmarker = await createFaceLandmarker(filesetResolver, "CPU");
  } catch {
    faceLandmarker = await createFaceLandmarker(filesetResolver, "GPU");
  }

  try {
    gestureRecognizer = await createGestureRecognizer(filesetResolver, "CPU");
  } catch {
    gestureRecognizer = await createGestureRecognizer(filesetResolver, "GPU");
  }
}

// Safety net for a detection call throwing at runtime after models already
// loaded successfully (e.g. a transient GPU failure if GPU ended up being
// used above). Rebuilds both models on CPU, which should never fail this
// way.
async function recoverWithCpuDelegate() {
  try {
    faceLandmarker = await createFaceLandmarker(filesetResolver, "CPU");
    gestureRecognizer = await createGestureRecognizer(filesetResolver, "CPU");
  } catch (err) {
    showCameraError(`Detection failed and could not recover: ${err.message}`);
  }
}

async function startWebcam() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCameraError("This browser does not support camera access.");
    return false;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    });
    video.srcObject = stream;
    await new Promise((resolve) => {
      video.onloadedmetadata = () => resolve();
    });
    return true;
  } catch (err) {
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      showCameraError("Camera access was denied. Please allow camera permissions and reload.");
    } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      showCameraError("No camera was found on this device.");
    } else {
      showCameraError(`Unable to access camera: ${err.message}`);
    }
    return false;
  }
}

let recoveryInFlight = false;
let lastDetectTimestamp = -1;

function predictLoop() {
  if (
    !faceLandmarker ||
    !gestureRecognizer ||
    video.paused ||
    video.ended ||
    !video.videoWidth ||
    !video.videoHeight
  ) {
    requestAnimationFrame(predictLoop);
    return;
  }

  // Deliberately NOT gated on video.currentTime: for a live getUserMedia
  // MediaStream (as opposed to a video file), currentTime doesn't reliably
  // advance on every mobile browser. Gating on it risked silently freezing
  // detection after the first frame with the camera still visibly playing.
  // MediaPipe requires a strictly increasing timestamp per detectForVideo
  // call, so we still guard against duplicate/non-increasing values here.
  const timestamp = performance.now();
  if (timestamp > lastDetectTimestamp) {
    lastDetectTimestamp = timestamp;
    try {
      if (
        captureCanvas.width !== video.videoWidth ||
        captureCanvas.height !== video.videoHeight
      ) {
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
      }
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

      const faceResult = faceLandmarker.detectForVideo(captureCanvas, timestamp);
      const gestureResult = gestureRecognizer.recognizeForVideo(captureCanvas, timestamp);

      const expression = classifyState(faceResult, gestureResult);
      updateDisplayedExpression(expression);
      updateDebugPanel(faceResult, gestureResult, expression, null);
    } catch (err) {
      console.error("Detection error, will retry on CPU delegate:", err);
      updateDebugPanel(null, null, null, err);
      if (!recoveryInFlight) {
        recoveryInFlight = true;
        recoverWithCpuDelegate().finally(() => {
          recoveryInFlight = false;
        });
      }
    }
  }

  requestAnimationFrame(predictLoop);
}

async function main() {
  const cameraReady = await startWebcam();
  if (!cameraReady) {
    loadingOverlay.classList.add("hidden");
    return;
  }

  try {
    await initModels();
  } catch (err) {
    loadingOverlay.classList.add("hidden");
    showCameraError(`Failed to load detection models: ${err.message}`);
    return;
  }

  loadingOverlay.classList.add("hidden");
  requestAnimationFrame(predictLoop);
}

main();
