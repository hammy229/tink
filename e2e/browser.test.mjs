// The whole in-browser path in real Chromium: the real ONNX model in WebAssembly, the real UI, audio through a fake mic.
//   node --test e2e/          (needs models/voc2vec-l12.int8.onnx, from scripts/export_onnx.py)
import assert from "node:assert/strict";
import { existsSync, readFileSync, statSync, writeFileSync, mkdirSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { after, before, test } from "node:test";
import { chromium } from "playwright";

const ROOT = new URL("..", import.meta.url).pathname;
const MODEL_FILE = process.env.TINK_E2E_MODEL || "voc2vec-l12.int8.onnx";
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".onnx": "application/octet-stream" };
const ARTIFACTS = join(ROOT, "e2e", "artifacts");
const HEADERS = JSON.parse(readFileSync(join(ROOT, "vercel.json"))).headers[0].headers; // the same ones production sends

let server, browser, page, base, loadInfo, embedMs;
const requests = [];
const cspViolations = [];

// Serves the site the way vercel.json does: static files, no API, cross-origin isolated so WASM can use threads.
function serve() {
  return new Promise((resolve) => {
    const s = createServer((req, res) => {
      const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname));
      const file = path.startsWith("/models/") ? join(ROOT, path) : join(ROOT, "web", path === "/" ? "index.html" : path);
      for (const { key, value } of HEADERS) res.setHeader(key, value);
      if (!existsSync(file) || !statSync(file).isFile()) { res.writeHead(404); res.end("{}"); return; }
      res.writeHead(200, { "Content-Type": TYPES[extname(file)] || "application/octet-stream" });
      res.end(readFileSync(file));
    }).listen(0, "127.0.0.1", () => resolve(s));
  });
}

// A fake mic: page code plays a hum, a hiss, or a whistle into the stream the app records.
function fakeMic() {
  const ac = new AudioContext();
  const dest = ac.createMediaStreamDestination();
  const quiet = ac.createBufferSource();
  const buf = ac.createBuffer(1, ac.sampleRate, ac.sampleRate);
  buf.getChannelData(0).forEach((_, i, d) => { d[i] = (Math.random() - 0.5) * 0.002; });
  quiet.buffer = buf; quiet.loop = true; quiet.connect(dest); quiet.start();
  navigator.mediaDevices.getUserMedia = async () => dest.stream;
  window.__mic = {
    async sound(kind, ms = 900) {
      await ac.resume();
      const gain = ac.createGain();
      gain.gain.setValueAtTime(0, ac.currentTime);
      gain.gain.linearRampToValueAtTime(0.5, ac.currentTime + 0.03);
      gain.gain.setValueAtTime(0.5, ac.currentTime + ms / 1000 - 0.05);
      gain.gain.linearRampToValueAtTime(0, ac.currentTime + ms / 1000);
      gain.connect(dest);
      let src;
      if (kind === "hiss") {
        src = ac.createBufferSource();
        const b = ac.createBuffer(1, ac.sampleRate * 2, ac.sampleRate);
        b.getChannelData(0).forEach((_, i, d) => { d[i] = (Math.random() - 0.5) * 1.2; });
        src.buffer = b;
      } else {
        src = ac.createOscillator();
        src.type = kind === "hum" ? "sawtooth" : "sine";
        src.frequency.value = kind === "hum" ? 130 : 1400;
      }
      src.connect(gain); src.start(); src.stop(ac.currentTime + ms / 1000);
      await new Promise((r) => setTimeout(r, ms));
    },
  };
}

async function teach(name, phrase, kind, takes) {
  await page.fill("#sname", name);
  await page.fill("#sphrase", phrase);
  for (let i = 0; i < takes; i++) {
    await page.click("#take");
    await page.waitForFunction(() => document.querySelector("#take span").textContent === "Make the sound now");
    await page.evaluate((k) => window.__mic.sound(k), kind);
    await page.waitForFunction(() => !document.getElementById("take").disabled, null, { timeout: 120000 });
    assert.equal(await page.textContent("#err"), "", `take ${i} of ${name}`);
  }
  await page.click("#done");
}

const heard = () => page.textContent("#phrase");

before(async () => {
  assert.ok(existsSync(join(ROOT, "models", MODEL_FILE)), `missing models/${MODEL_FILE}`);
  server = await serve();
  base = `http://127.0.0.1:${server.address().port}/`;
  browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required", "--disable-audio-output"] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  context.on("request", (r) => requests.push({ url: r.url(), method: r.method(), size: r.postDataBuffer()?.length ?? 0 }));
  await context.addInitScript(({ file, bytes }) => {
    window.__TINK_MODEL__ = { url: `/models/${file}`, bytes, sha256: "", tag: "e2e" };
    addEventListener("tink:local-ready", (e) => { window.__ready = e.detail; });
  }, { file: MODEL_FILE, bytes: statSync(join(ROOT, "models", MODEL_FILE)).size });
  await context.addInitScript(fakeMic);
  page = await context.newPage();
  page.on("pageerror", (e) => { throw e; });
  page.on("console", (m) => { if (/Content Security Policy/.test(m.text())) cspViolations.push(m.text()); });
});

after(async () => {
  await browser?.close();
  server?.close();
});

test("with no server, the page offers to run the model and only downloads on request", async () => {
  await page.goto(base);
  await page.waitForSelector("#startLocal");
  assert.match(await heard(), /Try Tink in this browser/);
  assert.equal(requests.filter((r) => r.url.endsWith(".onnx")).length, 0);
});

test("the model loads in WebAssembly", async () => {
  await page.click("#startLocal");
  await page.waitForFunction(() => window.__ready, null, { timeout: 180000 });
  loadInfo = await page.evaluate(() => { const { timings, ...rest } = window.__ready; return rest; });
  assert.ok(loadInfo.ms > 0);
  await page.click("#mute");
});

test("a person can teach sounds and be understood", async () => {
  await teach("Low hum", "I'm thirsty.", "hum", 3);
  await teach("Hiss", "I'm cold.", "hiss", 3);
  await teach("Whistle", "Please call my mom.", "whistle", 3);
  assert.match(await page.textContent("#selfTest"), /Self-test/);

  for (const [kind, phrase] of [["hum", "I'm thirsty."], ["hiss", "I'm cold."], ["whistle", "Please call my mom."]]) {
    await page.click("#listen");
    await page.waitForFunction(() => document.getElementById("statusText").textContent === "Listening");
    await page.evaluate((k) => window.__mic.sound(k), kind);
    await page.waitForFunction((want) => document.getElementById("phrase").textContent.includes(want), phrase, { timeout: 60000 });
    await page.waitForFunction(() => !document.getElementById("listen").disabled, null, { timeout: 60000 });
  }
  mkdirSync(ARTIFACTS, { recursive: true });
  await page.locator("#stage").screenshot({ path: join(ARTIFACTS, "understood.png") });
  embedMs = await page.evaluate(() => window.__ready.timings);
});

test("pressing Right teaches Tink the clip it just heard", async () => {
  await page.click("#listen");
  await page.waitForFunction(() => document.getElementById("statusText").textContent === "Listening");
  await page.evaluate(() => window.__mic.sound("whistle"));
  await page.waitForSelector("#right:not([hidden])", { timeout: 60000 });
  await page.waitForFunction(() => !document.getElementById("listen").disabled, null, { timeout: 60000 });
  await page.click("#right");
  await page.waitForFunction(() => /learned from that sound/.test(document.getElementById("feedbackMsg").textContent), null, { timeout: 30000 });
});

test("hands-free: turn on Keep listening once, and each sound is picked up with no button", async () => {
  await page.check("#loop");
  await page.waitForFunction(() => document.getElementById("statusText").textContent === "Listening");
  await new Promise((r) => setTimeout(r, 500)); // the noise floor settles
  for (const [kind, phrase] of [["hiss", "I'm cold."], ["hum", "I'm thirsty."]]) {
    await page.evaluate((k) => window.__mic.sound(k), kind);
    await page.waitForFunction((want) => document.getElementById("phrase").textContent.includes(want), phrase, { timeout: 60000 });
    await page.waitForFunction(() => document.getElementById("statusText").textContent === "Listening", null, { timeout: 30000 });
    await new Promise((r) => setTimeout(r, 300));
  }
  await page.uncheck("#loop");
  await page.waitForFunction(() => !document.getElementById("listen").disabled, null, { timeout: 15000 });
});

test("no audio ever leaves the browser", () => {
  const uploads = requests.filter((r) => r.method !== "GET" || r.size > 0);
  assert.deepEqual(uploads, []);
});

test("a reload remembers the sounds and reuses the cached model", async () => {
  const downloadsBefore = requests.filter((r) => r.url.endsWith(".onnx")).length;
  await page.reload();
  await page.waitForFunction(() => window.__ready, null, { timeout: 120000 });
  const again = await page.evaluate(() => window.__ready.cached);
  assert.equal(again, true);
  assert.equal(requests.filter((r) => r.url.endsWith(".onnx")).length, downloadsBefore);
  assert.equal(await page.locator(".sound").count(), 3);
});

test("forgetting a sound removes it for good", async () => {
  await page.locator('.sound[data-name="Whistle"] button').last().click();
  const yes = page.locator('.confirm-row button').first();
  await yes.click();
  await page.waitForFunction(() => document.querySelectorAll(".sound").length === 2, null, { timeout: 10000 });
  await page.reload();
  await page.waitForFunction(() => window.__ready, null, { timeout: 120000 });
  assert.equal(await page.locator(".sound").count(), 2);
});

test("the content security policy doesn't block anything the app needs", () => {
  assert.deepEqual(cspViolations, []);
});

test("a model that fails its hash is refused, and the page says so", async () => {
  const context = await browser.newContext();
  await context.addInitScript(({ file, bytes }) => {
    window.__TINK_MODEL__ = { url: `/models/${file}`, bytes, sha256: "0".repeat(64), tag: "tampered" };
  }, { file: MODEL_FILE, bytes: statSync(join(ROOT, "models", MODEL_FILE)).size });
  const tampered = await context.newPage();
  await tampered.goto(base);
  await tampered.click("#startLocal");
  await tampered.waitForFunction(() => /integrity/.test(document.getElementById("why").textContent), null, { timeout: 120000 });
  assert.match(await tampered.textContent("#phrase"), /didn't load/);
  await tampered.waitForSelector("#startLocal");
  await context.close();
});

test("record how fast it ran", async () => {
  const stats = await page.evaluate(() => ({ threads: window.__ready.threads, isolated: self.crossOriginIsolated }));
  const sorted = [...embedMs].sort((x, y) => x - y);
  const at = (q) => Math.round(sorted[Math.floor(q * (sorted.length - 1))]);
  mkdirSync(ARTIFACTS, { recursive: true });
  writeFileSync(join(ARTIFACTS, "browser-timing.json"), JSON.stringify({ first_load: loadInfo, ...stats, embeddings: embedMs.length, embed_ms_p50: at(0.5), embed_ms_p95: at(0.95) }, null, 2));
  assert.equal(stats.isolated, true);
});
