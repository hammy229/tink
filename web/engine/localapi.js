// The same API as the FastAPI server, answered inside the browser. Nothing leaves the device.
import { MODEL, ORT_VERSION } from "./config.js";
import * as core from "./core.js";

const DB = "tink-local", STORE = "kv";

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// Storage can be blocked (private windows, cleared site data). Tink still works for the visit, just without memory.
class Kv {
  constructor(db) { this.db = db; this.memory = new Map(); }
  static async open() { return new Kv(await openDb().catch(() => null)); }
  request(mode, fn) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(STORE, mode);
      const req = fn(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(req.result);
      tx.onerror = tx.onabort = () => reject(tx.error);
    });
  }
  async get(k) {
    if (!this.db) return this.memory.get(k);
    return this.request("readonly", (s) => s.get(k));
  }
  async set(k, v) {
    if (!this.db) { this.memory.set(k, v); return; }
    await this.request("readwrite", (s) => s.put(v, k));
  }
  async del(k) {
    if (!this.db) { this.memory.delete(k); return; }
    await this.request("readwrite", (s) => s.delete(k));
  }
}

class HttpError extends Error {}

export class LocalApi {
  constructor(worker, kv, state, model) {
    this.worker = worker; this.kv = kv; this.state = state; this.model = model;
    this.calib = core.calibrate(state.vecs);
    this.pending = null;
    this.waiting = new Map();
    this.nextId = 0;
    this.timings = [];
    worker.addEventListener("message", ({ data }) => {
      if (data.type !== "embedding" && data.type !== "error") return;
      const job = this.waiting.get(data.id);
      if (!job) return;
      this.waiting.delete(data.id);
      if (data.type === "error") return job.reject(new Error(data.message));
      this.timings.push(data.ms);
      return job.resolve(data.vector);
    });
  }

  static async start({ onProgress = () => {}, model = globalThis.__TINK_MODEL__ || MODEL } = {}) {
    const worker = new Worker(new URL("./model.worker.js", import.meta.url), { type: "module" });
    const ready = await new Promise((resolve, reject) => {
      const listen = ({ data }) => {
        if (data.type === "progress") onProgress(data.loaded, data.total, data.cached);
        else if (data.type === "ready") { worker.removeEventListener("message", listen); resolve(data); }
        else if (data.type === "error") { worker.removeEventListener("message", listen); reject(new Error(data.message)); }
      };
      worker.addEventListener("message", listen);
      worker.addEventListener("error", (e) => reject(new Error(e.message || "The model worker failed to start.")));
      worker.postMessage({ type: "load", model, ortVersion: ORT_VERSION });
    });
    const kv = await Kv.open();
    let state = await kv.get("state").catch(() => null);
    if (!state || state.model !== model.tag) state = { model: model.tag, sounds: {}, usage: {}, vecs: {} };
    const api = new LocalApi(worker, kv, state, model);
    api.loadInfo = ready;
    return api;
  }

  embed(wav) {
    const id = ++this.nextId, samples = core.normalize(wav);
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.worker.postMessage({ type: "embed", id, samples }, [samples.buffer]);
    });
  }

  async save() { await this.kv.set("state", this.state).catch(() => {}); }

  async clip(form) {
    const file = form.get("audio");
    if (!file) throw new HttpError("No audio in the request.");
    const { samples, rate } = core.decodeWav(await file.arrayBuffer());
    return core.trim(core.toModelRate(samples, rate));
  }

  sounds() {
    const { sounds, vecs } = this.state;
    return {
      sounds: Object.entries(sounds).map(([name, s]) => ({
        name, phrase: s.phrase, cues: s.cues ?? [], role: s.role ?? null, urgent: Boolean(s.urgent),
        voice: Boolean(s.voice), takes: vecs[name]?.length ?? 0,
      })),
      threshold: this.calib.threshold ?? core.DEFAULT_THRESHOLD,
      calibration: this.calib,
      min_sounds: core.MIN_SOUNDS,
      cues: core.CUES,
      llm: { on: false, model: "" },
      profile: { id: "local", name: "This browser", public: false },
    };
  }

  async enroll(form) {
    const name = String(form.get("name") ?? "").trim(), phrase = String(form.get("phrase") ?? "").trim();
    if (!name || !phrase) throw new HttpError("Both the sound name and the phrase are required.");
    if (name.length > core.MAX_TEXT || phrase.length > core.MAX_TEXT) throw new HttpError(`Keep the sound name and phrase under ${core.MAX_TEXT} characters.`);
    const role = String(form.get("role") ?? "");
    const wav = await this.clip(form);
    if (core.isSilent(wav) && role !== "background") throw new HttpError("Tink didn't hear a sound in that take. Move closer to the mic and try again.");
    const rows = [];
    for (const variant of [wav, ...core.augment(wav)]) rows.push(await this.embed(variant));
    const { sounds, vecs } = this.state;
    const s = (sounds[name] ??= { phrase, cues: [] });
    s.phrase = phrase;
    const cues = JSON.parse(String(form.get("cues") ?? "[]"));
    s.cues = Array.isArray(cues) ? cues.filter((c) => core.CUES.includes(c)) : [];
    if (core.ROLES.includes(role)) {
      for (const other of Object.values(sounds)) if (other.role === role) delete other.role;
      s.role = role;
    } else delete s.role;
    s.urgent = form.get("urgent") === "true" && s.role !== "background";
    (vecs[name] ??= []).push(rows);
    this.calib = core.calibrate(vecs);
    await this.save();
    return { name, takes: vecs[name].length };
  }

  async recognize(form) {
    const wav = await this.clip(form);
    const t0 = performance.now();
    const query = await this.embed(wav);
    const { sounds, vecs, usage } = this.state;
    const sc = core.scores(vecs, query);
    if (!Object.keys(sc).length) throw new HttpError(`Teach Tink at least ${core.MIN_SOUNDS} sounds first, so it can tell them apart.`);
    const now = new Date();
    const cueMap = Object.fromEntries(Object.entries(sounds).map(([n, s]) => [n, s.cues ?? []]));
    let cues = [];
    try { cues = JSON.parse(String(form.get("cues") ?? "[]")); } catch { /* bad cues are the same as no cues */ }
    const post = core.posterior(sc, this.calib.threshold ?? core.DEFAULT_THRESHOLD, usage, cueMap, Array.isArray(cues) ? cues : [], now.getHours(), this.calib.temp ?? core.DEFAULT_TEMP);
    const ranked = post.ranked;
    const view = {
      phrase: Object.fromEntries(Object.entries(sounds).map(([n, s]) => [n, s.phrase])),
      roles: Object.fromEntries(Object.entries(sounds).filter(([, s]) => s.role).map(([n, s]) => [n, s.role])),
      urgent: new Set(Object.entries(sounds).filter(([, s]) => s.urgent).map(([n]) => n)),
    };
    const nowMs = now.getTime();
    const outcome = core.decideWithoutWording(ranked, view, this.pending, nowMs) ?? core.finishTurn(ranked, view, nowMs);
    this.pending = outcome.pending;
    const top = ranked[0];
    return {
      p: top.p, taught: view.phrase[top.name] ?? null, ranked, scores: sc,
      packet: { part_of_day: post.part_of_day, webcam_cues: post.cues },
      timing: { heard_ms: Math.round(performance.now() - t0), llm_ms: null },
      voiced: Object.entries(sounds).filter(([, s]) => s.voice).map(([n]) => n).sort(),
      model: `${this.model.tag} (in this browser)`,
      ...outcome.fields,
    };
  }

  async bankVoice(form) {
    const name = String(form.get("name") ?? "");
    if (!this.state.sounds[name]) throw new HttpError(`No sound called "${name}".`);
    const wav = await this.clip(form);
    if (core.isSilent(wav)) throw new HttpError("Tink didn't hear the phrase. Move closer to the mic and try again.");
    await this.kv.set(`voice:${name}`, core.encodeWav(wav));
    this.state.sounds[name].voice = true;
    await this.save();
    return { ok: true, name };
  }

  async voiceBlob(name) {
    const data = await this.kv.get(`voice:${name}`);
    if (!data) throw new Error("no banked voice");
    return new Blob([data], { type: "audio/wav" });
  }

  async confirm(params) {
    const name = params.get("name"), { sounds, usage } = this.state;
    if (!sounds[name]) throw new HttpError(`No sound called "${name}".`);
    const bucket = core.partOfDay(new Date().getHours());
    usage[name] ??= {};
    usage[name][bucket] = (usage[name][bucket] ?? 0) + 1;
    await this.save();
    return { ok: true, usage: usage[name] };
  }

  async forget(name) {
    const { sounds, vecs, usage } = this.state;
    if (!sounds[name]) throw new HttpError(`No sound called "${name}".`);
    delete sounds[name]; delete vecs[name]; delete usage[name];
    await this.kv.del(`voice:${name}`).catch(() => {});
    this.calib = core.calibrate(vecs);
    await this.save();
    return { ok: true };
  }

  reply(params) {
    const text = String(params.get("text") ?? "").trim(), to = String(params.get("to") ?? "English");
    if (!text) throw new HttpError("Type what you want to say first.");
    if (text.length > 200) throw new HttpError("Keep it under 200 characters so it is easy to hear.");
    return { say: text, typed: text, to, source: "typed", packet: null };
  }

  // Same shape as app.js's api(): resolves with the JSON body, or throws with the server's message.
  async request(path, opts = {}) {
    const method = opts.method ?? "GET", body = opts.body;
    if (path === "/api/sounds" && method === "GET") return this.sounds();
    if (path === "/api/enroll") return this.enroll(body);
    if (path === "/api/recognize") return this.recognize(body);
    if (path === "/api/voice") return this.bankVoice(body);
    if (path === "/api/confirm") return this.confirm(body);
    if (path === "/api/reply") return this.reply(body);
    if (path.startsWith("/api/sounds/") && method === "DELETE") return this.forget(decodeURIComponent(path.slice("/api/sounds/".length)));
    throw new HttpError(`Not available in the browser: ${path}`);
  }
}
