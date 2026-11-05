// Runs the model off the main thread so the page never freezes during inference.
// Downloads once, verifies the hash, and keeps the model in the Cache API for later visits.
let session = null;
let ort = null;

async function sha256(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function download(model) {
  const cache = await caches.open("tink-model-v1").catch(() => null);
  const hit = cache && (await cache.match(model.url));
  if (hit) {
    postMessage({ type: "progress", loaded: model.bytes, total: model.bytes, cached: true });
    return { buffer: await hit.arrayBuffer(), cached: true };
  }
  const r = await fetch(model.url);
  if (!r.ok || !r.body) throw new Error(`Model download failed (${r.status})`);
  const total = Number(r.headers.get("content-length")) || model.bytes;
  const reader = r.body.getReader(), chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    postMessage({ type: "progress", loaded, total });
  }
  const whole = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) { whole.set(c, at); at += c.length; }
  return { buffer: whole.buffer, cached: false, store: cache };
}

async function load(model, ortVersion) {
  const t0 = performance.now();
  ort = await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/ort.wasm.min.mjs`);
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/`;
  // Threads need a cross-origin isolated page (see vercel.json). Without it, one thread still works.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const { buffer, cached, store } = await download(model);
  if (model.sha256) {
    const got = await sha256(buffer);
    if (got !== model.sha256) throw new Error("The downloaded model failed its integrity check.");
  }
  if (store) await store.put(model.url, new Response(buffer.slice(0))).catch(() => {});
  session = await ort.InferenceSession.create(new Uint8Array(buffer), { executionProviders: ["wasm"] });
  postMessage({ type: "ready", ms: Math.round(performance.now() - t0), threads: ort.env.wasm.numThreads, cached });
}

async function embed(id, samples) {
  const t0 = performance.now();
  const input = new ort.Tensor("float32", samples, [1, samples.length]);
  const out = await session.run({ input_values: input });
  const vector = Float32Array.from(out.embedding.data);
  postMessage({ type: "embedding", id, vector, ms: performance.now() - t0 }, [vector.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data.model, data.ortVersion);
    else if (data.type === "embed") await embed(data.id, data.samples);
  } catch (e) {
    postMessage({ type: "error", id: data.id, message: e.message || String(e) });
  }
};
