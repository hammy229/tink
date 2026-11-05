// Tink's decisions in the browser, with no server. No DOM and no I/O, so Node can test it.
// Mirrors tink/{audio,store,bayes,turn,vad}.py. tests/fixtures/parity.json holds outputs from the Python code,
// and web/engine/core.test.mjs fails if the two ever disagree.

export const SR = 16000;
export const CUES = ["nod", "shake", "smile", "frown", "eyes_closed"];
export const UNKNOWN = "unknown";
export const ROLES = ["yes", "no", "background"];
export const MIN_SOUNDS = 2;
export const MAX_TEXT = 80;
export const MAX_LEARNED_TAKES = 30;
export const DEFAULT_THRESHOLD = 0.5;
export const DEFAULT_TEMP = 0.1;
export const ASK_BELOW = 0.6;
export const DIALOG_SECONDS = 45;
export const ANSWER_MIN = 0.6;
export const URGENT_MIN = 0.4;
export const NOT_CAUGHT = "I didn't catch that. Can you make the sound again?";
export const GIVE_UP = "Okay. Make the sound again when you're ready.";
const ALPHA = 1.0;
const P_TAGGED = 0.6, P_OTHER = 0.2;

const round3 = (x) => Math.round(x * 1000) / 1000;

// ---------- audio ----------

export function decodeWav(buffer) {
  const v = new DataView(buffer);
  const tag = (o) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (v.byteLength < 44 || tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("Not a WAV file");
  let channels = 1, rate = SR, bits = 16, format = 1, offset = 12;
  while (offset + 8 <= v.byteLength) {
    const id = tag(offset), size = v.getUint32(offset + 4, true);
    if (id === "fmt ") {
      format = v.getUint16(offset + 8, true);
      channels = v.getUint16(offset + 10, true);
      rate = v.getUint32(offset + 12, true);
      bits = v.getUint16(offset + 22, true);
    } else if (id === "data") {
      const n = Math.floor(Math.min(size, v.byteLength - offset - 8) / (bits / 8) / channels);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let c = 0; c < channels; c++) {
          const at = offset + 8 + (i * channels + c) * (bits / 8);
          if (format === 3 && bits === 32) sum += v.getFloat32(at, true);
          else if (bits === 16) sum += v.getInt16(at, true) / 32768;
          else throw new Error(`Unsupported WAV: ${bits}-bit format ${format}`);
        }
        out[i] = sum / channels;
      }
      return { samples: out, rate };
    }
    offset += 8 + size + (size % 2);
  }
  throw new Error("WAV has no data");
}

export function encodeWav(samples, rate = SR) {
  const b = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o, s) => [...s].forEach((c, i) => b.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); b.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVEfmt ");
  b.setUint32(16, 16, true); b.setUint16(20, 1, true); b.setUint16(22, 1, true); b.setUint32(24, rate, true);
  b.setUint32(28, rate * 2, true); b.setUint16(32, 2, true); b.setUint16(34, 16, true);
  str(36, "data"); b.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => b.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  return b.buffer;
}

function bessel0(x) {
  let sum = 1, term = 1;
  for (let k = 1; k < 25; k++) { term *= (x / (2 * k)) ** 2; sum += term; }
  return sum;
}

const gcd = (a, b) => (b ? gcd(b, a % b) : a);

// Same recipe as scipy.signal.resample_poly: a Kaiser-windowed sinc (beta 5, half-length 10 x max(up, down)).
export function resamplePoly(x, up, down) {
  const g = gcd(up, down);
  up /= g; down /= g;
  if (up === 1 && down === 1) return Float32Array.from(x);
  const maxRate = Math.max(up, down), half = 10 * maxRate, cutoff = 1 / maxRate, beta = 5;
  const h = new Float64Array(2 * half + 1);
  const norm = bessel0(beta);
  for (let i = 0; i <= 2 * half; i++) {
    const t = i - half, r = t / half;
    const sinc = t === 0 ? 1 : Math.sin(Math.PI * cutoff * t) / (Math.PI * cutoff * t);
    h[i] = cutoff * sinc * (bessel0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / norm) * up;
  }
  const n = Math.ceil((x.length * up) / down);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const center = i * down;
    const first = Math.ceil((center - half) / up), last = Math.floor((center + half) / up);
    let sum = 0;
    for (let j = Math.max(0, first); j <= Math.min(x.length - 1, last); j++) sum += x[j] * h[j * up - center + half];
    out[i] = sum;
  }
  return out;
}

export const toModelRate = (samples, rate) => (rate === SR ? samples : resamplePoly(samples, SR, rate));

function frameRms(wav, frame) {
  const n = Math.floor(wav.length / frame), out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < frame; j++) s += wav[i * frame + j] ** 2;
    out[i] = Math.sqrt(s / frame);
  }
  return out;
}

export function trim(wav, frame = 320, rel = 0.1, pad = 0.1, minSeconds = 0.4) {
  const rms = frameRms(wav, frame);
  let out = wav;
  if (rms.length) {
    const top = rms.reduce((a, b) => Math.max(a, b), 0);
    let first = -1, last = -1;
    rms.forEach((r, i) => { if (r > top * rel) { if (first < 0) first = i; last = i; } });
    if (first >= 0) {
      const p = Math.floor(pad * SR);
      out = wav.slice(Math.max(0, first * frame - p), (last + 1) * frame + p);
    }
  }
  const short = Math.floor(minSeconds * SR) - out.length;
  if (short > 0) { const padded = new Float32Array(out.length + short); padded.set(out); return padded; }
  return out;
}

export function isSilent(wav, floor = 0.005) {
  const rms = frameRms(wav, 320);
  return rms.length === 0 || rms.reduce((a, b) => Math.max(a, b), 0) < floor;
}

// What HuBERT's feature extractor does: zero mean, unit variance.
export function normalize(wav) {
  let mean = 0;
  for (const v of wav) mean += v;
  mean /= wav.length;
  let variance = 0;
  for (const v of wav) variance += (v - mean) ** 2;
  variance /= wav.length;
  const scale = Math.sqrt(variance + 1e-7), out = new Float32Array(wav.length);
  for (let i = 0; i < wav.length; i++) out[i] = (wav[i] - mean) / scale;
  return out;
}

// A small seeded generator, so the same take always gets the same variants.
function gaussian(seed) {
  let s = seed >>> 0 || 1, spare = null;
  const u = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    const r = Math.sqrt(-2 * Math.log(u() || 1e-12)), a = 2 * Math.PI * u();
    spare = r * Math.sin(a);
    return r * Math.cos(a);
  };
}

export function augment(wav, seed = 1) {
  const normal = gaussian(seed);
  let power = 0;
  for (const v of wav) power += v * v;
  power = power / wav.length || 1e-8;
  const noisy = [25, 15].map((snr) => {
    const sigma = Math.sqrt(power / 10 ** (snr / 10));
    return wav.map((v) => v + sigma * normal());
  });
  return [resamplePoly(wav, 16, 17), resamplePoly(wav, 17, 16), ...noisy];
}

// ---------- scoring ----------

function unitInPlace(v) {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) + 1e-9;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

function meanOf(rows, dim) {
  const out = new Float64Array(dim);
  for (const r of rows) for (let i = 0; i < dim; i++) out[i] += r[i];
  for (let i = 0; i < dim; i++) out[i] /= rows.length;
  return out;
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const minus = (a, b) => { const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] - b[i]; return o; };

// `vecs` is {name: [take, ...]}, and a take is a list of rows: the take itself, then its augmented variants.
function centroids(vecs, names) {
  const perName = names.map((n) => vecs[n].flat());
  const dim = perName[0][0].length;
  const mu = meanOf(perName.flat(), dim);
  return { mu, cents: perName.map((rows) => unitInPlace(minus(meanOf(rows, dim), mu))) };
}

export function scores(vecs, query) {
  const names = Object.keys(vecs).filter((n) => vecs[n].length);
  if (names.length < MIN_SOUNDS) return {};
  const { mu, cents } = centroids(vecs, names);
  const q = unitInPlace(minus(query, mu));
  return Object.fromEntries(names.map((n, i) => [n, round3(dot(cents[i], q))]));
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const TEMPS = Array.from({ length: 50 }, (_, i) => 0.04 * (1 / 0.04) ** (i / 49));

export function calibrate(vecs) {
  const names = Object.keys(vecs).filter((n) => vecs[n].length);
  const S = [], y = [];
  names.forEach((n, si) => {
    if (vecs[n].length < 2) return;
    vecs[n].forEach((held, ti) => {
      const kept = Object.fromEntries(names.map((m) => [m, vecs[m].filter((_, j) => !(m === n && j === ti))]));
      const { mu, cents } = centroids(kept, names);
      const q = unitInPlace(minus(held[0], mu));
      S.push(cents.map((c) => dot(c, q)));
      y.push(si);
    });
  });
  if (S.length < 4 || new Set(y).size < 2) return { n: S.length, ready: false };
  const right = S.map((row, i) => row[y[i]]);
  const wrong = S.map((row, i) => Math.max(...row.filter((_, k) => k !== y[i])));
  const correct = S.map((row, i) => row.indexOf(Math.max(...row)) === y[i]);
  const probs = (row, t) => {
    const top = Math.max(...row), e = row.map((v) => Math.exp((v - top) / t)), z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z);
  };
  const nll = (t) => S.reduce((sum, row, i) => sum - Math.log(probs(row, t)[y[i]]), 0) / S.length;
  let temp = TEMPS[0], best = Infinity;
  for (const t of TEMPS) { const v = nll(t); if (v < best) { best = v; temp = t; } }
  const conf = S.map((row) => Math.max(...probs(row, temp)));
  const sure = conf.map((c) => c >= 0.9);
  const threshold = Math.min(0.9, Math.max(0.1, (median(right) + median(wrong)) / 2));
  const nRight = correct.filter(Boolean).length;
  return {
    ready: true, n: y.length, right: nRight, accuracy: round3(nRight / y.length),
    sure_n: sure.filter(Boolean).length, sure_right: correct.filter((c, i) => c && sure[i]).length,
    temp: Math.round(temp * 1e4) / 1e4, threshold: round3(threshold),
  };
}

// ---------- Bayes ----------

export const partOfDay = (h) => (h < 6 ? "night" : h < 12 ? "morning" : h < 18 ? "afternoon" : "evening");

function normalizeLogs(logs) {
  const top = Math.max(...Object.values(logs));
  const exp = Object.fromEntries(Object.entries(logs).map(([k, v]) => [k, Math.exp(v - top)]));
  const z = Object.values(exp).reduce((a, b) => a + b, 0);
  return Object.fromEntries(Object.entries(exp).map(([k, v]) => [k, v / z]));
}

export function posterior(sc, threshold, usage, soundCues, cuesSeen, hour, temp = DEFAULT_TEMP) {
  const names = Object.keys(sc);
  const audio = { ...Object.fromEntries(names.map((n) => [n, sc[n] / temp])), [UNKNOWN]: threshold / temp };
  const bucket = partOfDay(hour);
  const counts = { ...Object.fromEntries(names.map((n) => [n, usage[n]?.[bucket] ?? 0])), [UNKNOWN]: 0 };
  const total = Object.values(counts).reduce((a, b) => a + b, 0) + ALPHA * Object.keys(counts).length;
  const prior = Object.fromEntries(Object.entries(counts).map(([k, c]) => [k, (c + ALPHA) / total]));
  const seen = cuesSeen.filter((c) => CUES.includes(c));
  const cue = Object.fromEntries(names.map((n) => [n, seen.reduce((s, c) => s + Math.log((soundCues[n] ?? []).includes(c) ? P_TAGGED : P_OTHER), 0)]));
  cue[UNKNOWN] = seen.length * Math.log(P_OTHER);
  const final = normalizeLogs(Object.fromEntries(Object.keys(audio).map((k) => [k, audio[k] + Math.log(prior[k]) + cue[k]])));
  const audioOnly = normalizeLogs(audio);
  const ranked = Object.keys(final).sort((a, b) => final[b] - final[a]);
  return {
    part_of_day: bucket, cues: seen,
    ranked: ranked.map((k) => ({ name: k, p: round3(final[k]), audio_p: round3(audioOnly[k]), prior: round3(prior[k]) })),
  };
}

// ---------- one turn ----------
// `sounds` is {phrase, roles, urgent: Set}, and `pending` is {options, i, at (ms)} or null.

const hasYesNo = (sounds) => { const v = Object.values(sounds.roles); return v.includes("yes") && v.includes("no"); };

export function decideWithoutWording(ranked, sounds, pending, nowMs) {
  const top = ranked[0], phrase = sounds.phrase;
  const isMatch = top.name !== UNKNOWN && top.name in phrase;
  const role = isMatch ? sounds.roles[top.name] : undefined;
  if (role === "background") {
    return { fields: { matched: false, name: null, say: "", ask: false, source: "background", background: true, taught: null, dialog: null }, pending, said: null };
  }
  const urgent = ranked.find((r) => sounds.urgent.has(r.name) && r.p >= URGENT_MIN);
  if (urgent && urgent.name in phrase) {
    const say = phrase[urgent.name];
    return { fields: { p: urgent.p, matched: true, name: urgent.name, say, ask: false, source: "urgent", urgent: true, taught: say, dialog: null }, pending: null, said: say };
  }
  const answering = role && top.p >= ANSWER_MIN ? role : null;
  if (pending && answering && (nowMs - pending.at) / 1000 < DIALOG_SECONDS) {
    const asked = pending.options[pending.i];
    if (answering === "yes" && asked in phrase) {
      const say = phrase[asked];
      return { fields: { matched: true, name: asked, say, ask: false, source: "dialog", taught: say, dialog: { answered: "yes", meant: asked } }, pending: null, said: say };
    }
    const following = pending.i + 1;
    const nxt = following < pending.options.length ? pending.options[following] : null;
    if (nxt !== null && nxt in phrase) {
      return { fields: { matched: false, name: null, say: `Do you mean "${phrase[nxt]}"?`, ask: true, source: "dialog", dialog: { answered: "no", asking: nxt } }, pending: { ...pending, i: following, at: nowMs }, said: null };
    }
    return { fields: { matched: false, name: null, say: GIVE_UP, ask: true, source: "dialog", dialog: { answered: "no" } }, pending: null, said: null };
  }
  return null;
}

export function fallback(ranked, phrase) {
  const top = ranked[0], second = ranked[1];
  if (top.name === UNKNOWN || !(top.name in phrase)) return { say: NOT_CAUGHT, ask: true };
  if (top.p < ASK_BELOW && second && second.name in phrase) {
    return { say: `Do you mean "${phrase[top.name]}" or "${phrase[second.name]}"?`, ask: true };
  }
  return { say: phrase[top.name], ask: false };
}

export function finishTurn(ranked, sounds, nowMs) {
  const top = ranked[0], phrase = sounds.phrase;
  const isMatch = top.name !== UNKNOWN && top.name in phrase;
  let out = { ...fallback(ranked, phrase) }, source = "phrase", dialog = null, pending = null;
  if (!isMatch || top.p < ASK_BELOW) out.ask = true;
  if (out.ask && hasYesNo(sounds)) {
    const options = ranked.filter((r) => r.name in phrase && !(r.name in sounds.roles)).map((r) => r.name).slice(0, 3);
    if (options.length) {
      pending = { options, i: 0, at: nowMs };
      out = { say: `Do you mean "${phrase[options[0]]}"?`, ask: true };
      source = "dialog"; dialog = { asking: options[0] };
    }
  }
  return {
    fields: { matched: isMatch, name: isMatch ? top.name : null, say: out.say, ask: out.ask, source, dialog },
    pending, said: out.ask ? null : out.say,
  };
}

// ---------- hands-free listening ----------

export const FRAME = 320;
export const VAD_DEFAULTS = { ratio: 3, floorMin: 0.002, floorRise: 0.02, floorRiseLoud: 0.002, floorFall: 0.2, startFrames: 3, endFrames: 25, preRoll: 10, maxFrames: 200, minFrames: 5 };

export class SegmentDetector {
  constructor(cfg = {}) {
    this.cfg = { ...VAD_DEFAULTS, ...cfg };
    this.floor = null;
    this.buffer = new Float32Array(0);
    this.before = [];
    this.current = null;
    this.above = 0;
    this.quiet = 0;
    this.loud = 0;
  }

  push(samples) {
    const joined = new Float32Array(this.buffer.length + samples.length);
    joined.set(this.buffer);
    joined.set(samples, this.buffer.length);
    const n = Math.floor(joined.length / FRAME), out = [];
    for (let i = 0; i < n; i++) {
      const done = this.frame(joined.subarray(i * FRAME, (i + 1) * FRAME).slice());
      if (done) out.push(done);
    }
    this.buffer = joined.slice(n * FRAME);
    return out;
  }

  flush() {
    const done = this.current ? this.finish() : null;
    return done ? [done] : [];
  }

  threshold() { return Math.max(this.cfg.floorMin, this.floor ?? 0) * this.cfg.ratio; }

  frame(frame) {
    let s = 0;
    for (const v of frame) s += v * v;
    const rms = Math.sqrt(s / frame.length), loud = rms > this.threshold(), c = this.cfg;
    if (this.floor === null) this.floor = rms;
    else if (rms < this.floor) this.floor += (rms - this.floor) * c.floorFall;
    else this.floor += (rms - this.floor) * (loud ? c.floorRiseLoud : c.floorRise);
    if (!this.current) {
      this.above = loud ? this.above + 1 : 0;
      this.before.push(frame);
      if (this.before.length > c.preRoll + c.startFrames) this.before.shift();
      if (this.above >= c.startFrames) {
        this.current = this.before;
        this.before = [];
        this.loud = this.above;
        this.quiet = this.above = 0;
      }
      return null;
    }
    this.current.push(frame);
    if (loud) { this.quiet = 0; this.loud += 1; } else this.quiet += 1;
    return this.quiet >= c.endFrames || this.current.length >= c.maxFrames ? this.finish() : null;
  }

  finish() {
    const frames = this.current;
    this.current = null;
    this.quiet = 0;
    if (this.loud < this.cfg.minFrames) return null;
    const out = new Float32Array(frames.length * FRAME);
    frames.forEach((f, i) => out.set(f, i * FRAME));
    return out;
  }
}
