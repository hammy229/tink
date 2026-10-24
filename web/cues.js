// Webcam cues with MediaPipe Face Landmarker, all in the browser.
// Only cue names (nod, shake, smile, frown, eyes_closed) leave this file. Video never does.
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WINDOW_MS = 1500;   // head-motion history for nod/shake
const NOD_AMP = 0.05;     // pitch swing, as a fraction of face height
const SHAKE_AMP = 0.10;   // yaw swing, as a fraction of face width
const CLOSED_MS = 400;    // eyes shut this long counts as eyes_closed

let landmarker, video, stream, onLive = () => {}, running = false, lastT = -1;
let hist = [], closedSince = 0, collecting = null;

const avg = (bs, a, b) => ((bs[a] || 0) + (bs[b] || 0)) / 2;

// Counts back-and-forth swings bigger than `amp` (a nod is down-up, a shake is left-right-left).
function swings(values, amp) {
  let ext = values[0], dir = 0, n = 0;
  for (const v of values) {
    if (dir >= 0 && v > ext) ext = v;
    else if (dir <= 0 && v < ext) ext = v;
    if (Math.abs(v - ext) > amp / 2 && Math.sign(v - ext) !== dir) { n++; dir = Math.sign(v - ext); ext = v; }
  }
  return n;
}

function analyze(result, now) {
  const live = new Set();
  const lm = result.faceLandmarks?.[0], cats = result.faceBlendshapes?.[0]?.categories;
  if (lm && cats) {
    const bs = Object.fromEntries(cats.map((c) => [c.categoryName, c.score]));
    const nose = lm[1], left = lm[234], right = lm[454], top = lm[10], chin = lm[152];
    hist.push({
      t: now,
      yaw: (nose.x - (left.x + right.x) / 2) / Math.max(1e-3, Math.abs(right.x - left.x)),
      pitch: (nose.y - (top.y + chin.y) / 2) / Math.max(1e-3, Math.abs(chin.y - top.y)),
    });
    hist = hist.filter((h) => now - h.t < WINDOW_MS);

    if (avg(bs, "mouthSmileLeft", "mouthSmileRight") > 0.5) live.add("smile");
    if (avg(bs, "mouthFrownLeft", "mouthFrownRight") > 0.35 || avg(bs, "browDownLeft", "browDownRight") > 0.5) live.add("frown");
    if (avg(bs, "eyeBlinkLeft", "eyeBlinkRight") > 0.6) {
      closedSince = closedSince || now;
      if (now - closedSince > CLOSED_MS) live.add("eyes_closed");
    } else closedSince = 0;
    if (swings(hist.map((h) => h.pitch), NOD_AMP) >= 2) live.add("nod");
    if (swings(hist.map((h) => h.yaw), SHAKE_AMP) >= 2) live.add("shake");
  } else {
    hist = [];
  }
  if (collecting) live.forEach((c) => collecting.add(c));
  onLive([...live], Boolean(lm));
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastT) {
    lastT = video.currentTime;
    const now = performance.now();
    analyze(landmarker.detectForVideo(video, now), now);
  }
  requestAnimationFrame(loop);
}

async function enable(videoEl, onLiveCues) {
  video = videoEl; onLive = onLiveCues;
  stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240, facingMode: "user" } });
  video.srcObject = stream;
  await video.play();
  landmarker = landmarker || await FaceLandmarker.createFromOptions(await FilesetResolver.forVisionTasks(WASM), {
    baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
  running = true;
  loop();
}

function disable() {
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  if (video) video.srcObject = null;
  hist = []; collecting = null;
  onLive([], false);
}

// Collect every cue seen between begin() and end(), i.e. while the person makes the sound.
const begin = () => { collecting = new Set(); };
const end = () => { const out = [...(collecting || [])]; collecting = null; return out; };

window.TinkCues = { enable, disable, begin, end, isOn: () => running };
