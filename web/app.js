// Tink frontend: records a clip from the mic, sends it to the FastAPI backend, speaks the phrase.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const stage = $("stage"), phrase = $("phrase"), why = $("why"), statusText = $("statusText");

  let sounds = [], threshold = 0;
  let muted = false, lastPhrase = null, busy = false;
  const log = [];

  // ---------- API ----------
  async function api(path, opts) {
    const r = await fetch(path, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || `Server error ${r.status}`);
    return body;
  }
  async function refresh(lit) {
    const r = await api("/api/sounds");
    sounds = r.sounds; threshold = r.threshold;
    renderSounds(lit);
  }

  // ---------- Mic recording ----------
  // Records until the sound ends (silenceMs of quiet after something loud) or maxMs passes.
  // Resolves with a 16-bit WAV Blob at the mic's native rate; the backend resamples to 16 kHz.
  let stream, ctx;
  async function recordClip({ maxMs = 4000, silenceMs = 700, startTimeoutMs = 5000, level = 0.02 } = {}) {
    stream = stream || await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true },
    });
    ctx = ctx || new AudioContext();
    if (ctx.state === "suspended") await ctx.resume();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(2048, 1, 1);
    const chunks = [];
    const t0 = performance.now();
    let heardAt = 0, lastLoud = 0;

    return new Promise((resolve, reject) => {
      const finish = (err) => {
        proc.onaudioprocess = null; src.disconnect(); proc.disconnect(); setLevel(0);
        if (err) return reject(err);
        resolve(encodeWav(chunks, ctx.sampleRate));
      };
      proc.onaudioprocess = (e) => {
        const d = new Float32Array(e.inputBuffer.getChannelData(0));
        chunks.push(d);
        let sum = 0; for (const v of d) sum += v * v;
        const rms = Math.sqrt(sum / d.length), now = performance.now();
        setLevel(Math.min(1, rms * 12));
        if (rms > level) { lastLoud = now; heardAt = heardAt || now; }
        if (!heardAt && now - t0 > startTimeoutMs) return finish(new Error("silence"));
        if (heardAt && (now - lastLoud > silenceMs || now - heardAt > maxMs)) finish();
      };
      src.connect(proc); proc.connect(ctx.destination);
    });
  }
  function encodeWav(chunks, rate) {
    const n = chunks.reduce((a, c) => a + c.length, 0);
    const buf = new DataView(new ArrayBuffer(44 + n * 2));
    const str = (o, s) => [...s].forEach((ch, i) => buf.setUint8(o + i, ch.charCodeAt(0)));
    str(0, "RIFF"); buf.setUint32(4, 36 + n * 2, true); str(8, "WAVEfmt ");
    buf.setUint32(16, 16, true); buf.setUint16(20, 1, true); buf.setUint16(22, 1, true);
    buf.setUint32(24, rate, true); buf.setUint32(28, rate * 2, true); buf.setUint16(32, 2, true); buf.setUint16(34, 16, true);
    str(36, "data"); buf.setUint32(40, n * 2, true);
    let o = 44;
    for (const c of chunks) for (const v of c) { buf.setInt16(o, Math.max(-1, Math.min(1, v)) * 0x7fff, true); o += 2; }
    return new Blob([buf], { type: "audio/wav" });
  }
  const bars = $("bars");
  const setLevel = (v) => bars.style.setProperty("--lvl", v.toFixed(2));
  function micError(e) {
    if (e.message === "silence") return "Tink didn't hear anything. Move closer to the mic and try again.";
    if (e.name === "NotAllowedError") return "Tink needs the microphone. Allow it in the browser's address bar, then try again.";
    if (e.name === "NotFoundError") return "No microphone found. Plug one in and try again.";
    return e.message;
  }

  // ---------- Sound: chime + voice ----------
  function chime(ok) {
    try {
      ctx = ctx || new AudioContext();
      const now = ctx.currentTime;
      (ok ? [1568, 2093, 2637] : [523, 392]).forEach((f, i) => {
        const o = ctx.createOscillator(), g = ctx.createGain(), t = now + i * 0.07;
        o.type = "sine"; o.frequency.value = f;
        g.gain.setValueAtTime(0, t);
        g.gain.linearRampToValueAtTime(0.12, t + 0.01);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 0.9);
        o.connect(g).connect(ctx.destination); o.start(t); o.stop(t + 1);
      });
    } catch {}
  }
  function speak(text) {
    if (muted || !text) return;
    try { speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(text); u.rate = 0.95; speechSynthesis.speak(u); } catch {}
  }

  // ---------- Listen ----------
  const setStage = (state, status) => { stage.className = `stage ${state}`; statusText.textContent = status; };
  const clock = () => new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }).replace(/\s?[AP]M/i, "");

  async function listen() {
    if (busy || !sounds.length) return;
    busy = true; $("listen").disabled = true;
    setStage("listening", "Listening...");
    phrase.textContent = ""; why.textContent = "";
    renderSounds();
    try {
      const clip = await recordClip();
      setStage("listening", "Thinking...");
      const fd = new FormData(); fd.append("audio", clip, "clip.wav");
      const r = await api("/api/recognize", { method: "POST", body: fd });
      if (r.matched) {
        const s = sounds.find((x) => x.name === r.name);
        setStage("heard", "Heard you");
        phrase.textContent = r.phrase;
        why.innerHTML = `Heard <b>${esc(r.name.toLowerCase())}</b> with a <span class="conf">${r.score.toFixed(2)}</span> match`;
        lastPhrase = r.phrase; $("repeat").disabled = false;
        addLog(r.phrase, r.score, false);
        renderSounds(s && s.name);
        chime(true); speak(r.phrase); burst();
      } else {
        setStage("missed", "Not sure");
        phrase.textContent = "I didn't catch that.";
        why.innerHTML = `Closest match <span class="conf">${r.score.toFixed(2)}</span> is below ${threshold.toFixed(2)}. Try the sound again.`;
        addLog("Didn't catch that", r.score, true);
        chime(false);
      }
    } catch (e) {
      if (e.message === "silence" && $("loop").checked) { busy = false; return listen(); }
      setStage("error", "Stopped");
      phrase.textContent = "Something went wrong.";
      why.textContent = micError(e);
      $("loop").checked = false;
    }
    busy = false; $("listen").disabled = !sounds.length;
    if ($("loop").checked) setTimeout(listen, 900);
    else if (!stage.classList.contains("error")) statusText.textContent = "Ready";
  }

  function addLog(text, c, miss) {
    log.push({ t: clock(), text, c, miss });
    $("log").innerHTML = log.slice().reverse().map((e) =>
      `<li><time>${e.t}</time><span class="${e.miss ? "miss" : ""}">${esc(e.text)}</span><span class="c">${e.c.toFixed(2)}</span></li>`).join("");
    $("logEmpty").hidden = true;
  }

  // ---------- Sounds list ----------
  const dots = (n) => Array.from({ length: 5 }, (_, i) => `<i class="${i < n ? "" : "off"}"></i>`).join("");
  function renderSounds(lit) {
    $("sounds").innerHTML = sounds.map((s) => `
      <li class="sound${s.name === lit ? " lit" : ""}">
        <span class="name">${esc(s.name)}</span>
        <span class="takes" aria-label="${s.takes} takes">${dots(s.takes)}</span>
        <button class="forget" data-name="${esc(s.name)}">Forget</button>
        <span class="says">says "${esc(s.phrase)}"</span>
      </li>`).join("");
    $("soundsEmpty").hidden = sounds.length > 0;
    $("count").textContent = sounds.length ? `${sounds.length} sound${sounds.length > 1 ? "s" : ""}` : "";
    $("listen").disabled = busy || !sounds.length;
    if (!sounds.length && !busy) {
      setStage("idle", "Ready");
      phrase.textContent = "Teach Tink a sound to begin.";
      why.textContent = 'Use "Teach a new sound" on the right.';
    } else if (stage.classList.contains("idle") && phrase.textContent.startsWith("Teach")) {
      phrase.textContent = "Make a sound Tink knows.";
      why.textContent = "Press Listen, or turn on Keep listening.";
    }
  }
  $("sounds").addEventListener("click", async (e) => {
    const b = e.target.closest(".forget");
    if (!b) return;
    if (!b.classList.contains("armed")) {
      b.classList.add("armed"); b.textContent = "Tap again to forget";
      return setTimeout(() => { b.classList.remove("armed"); b.textContent = "Forget"; }, 3000);
    }
    try { await api(`/api/sounds/${encodeURIComponent(b.dataset.name)}`, { method: "DELETE" }); await refresh(); }
    catch (err) { $("err").textContent = err.message; }
  });

  // ---------- Enroll ----------
  let enrollTakes = 0;
  const drawTakes = () => { $("enrollDots").innerHTML = enrollTakes ? dots(enrollTakes) + `<span class="help">&nbsp;${enrollTakes} of 5</span>` : ""; };
  $("enroll").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("sname").value.trim(), says = $("sphrase").value.trim();
    if (!name || !says) { $("err").textContent = "Add both the sound and what Tink should say."; return; }
    $("err").textContent = "";
    const btn = $("take"); btn.disabled = true; btn.textContent = "Make the sound now...";
    try {
      const clip = await recordClip({ maxMs: 4000 });
      btn.textContent = "Saving...";
      const fd = new FormData(); fd.append("name", name); fd.append("phrase", says); fd.append("audio", clip, "take.wav");
      const r = await api("/api/enroll", { method: "POST", body: fd });
      enrollTakes = r.takes; drawTakes(); chime(true);
      $("sname").readOnly = true; $("done").hidden = false;
      await refresh(name);
    } catch (err) {
      $("err").textContent = micError(err);
    }
    btn.disabled = false;
    btn.textContent = enrollTakes >= 5 ? "Record another take" : "Record take";
  });
  $("done").addEventListener("click", () => {
    enrollTakes = 0; drawTakes(); $("enroll").reset(); $("sname").readOnly = false; $("done").hidden = true; $("err").textContent = "";
  });

  // ---------- Controls ----------
  $("listen").addEventListener("click", listen);
  $("loop").addEventListener("change", (e) => { if (e.target.checked) listen(); });
  $("repeat").addEventListener("click", () => { chime(true); speak(lastPhrase); });
  $("mute").addEventListener("click", (e) => {
    muted = !muted; e.target.setAttribute("aria-pressed", muted); e.target.textContent = muted ? "Unmute voice" : "Mute voice";
  });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.repeat && !/INPUT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)) { e.preventDefault(); listen(); }
  });

  // ---------- Pixie dust ----------
  const cv = $("dust"), g = cv.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let W = 0, H = 0, motes = [];
  function size() { const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1; W = r.width; H = r.height; cv.width = W * d; cv.height = H * d; g.setTransform(d, 0, 0, d, 0, 0); }
  const star = () => ({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - 0.5) * 0.15, vy: -Math.random() * 0.25 - 0.05, r: Math.random() * 1.4 + 0.4, life: Math.random(), decay: 0.002, star: true });
  function burst() {
    if (reduce) return;
    for (let i = 0; i < 70; i++) motes.push({ x: W / 2, y: H * 0.42, vx: (Math.random() - 0.5) * 4, vy: (Math.random() - 0.5) * 4 - 1, r: Math.random() * 2 + 1, life: 1, decay: 0.012 + Math.random() * 0.01 });
  }
  function frame() {
    g.clearRect(0, 0, W, H);
    g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--gold").trim() || "#f2c14e";
    motes = motes.filter((m) => m.star || m.life > 0);
    for (const m of motes) {
      m.x += m.vx; m.y += m.vy;
      if (!m.star) m.vy += 0.03;
      m.life -= m.decay;
      if (m.star && m.life <= 0) Object.assign(m, star(), { life: 1 });
      if (m.y < -5) m.y = H + 5;
      g.globalAlpha = Math.max(0, m.star ? 0.25 + 0.5 * Math.abs(Math.sin(m.life * 6)) : m.life);
      g.beginPath(); g.arc(m.x, m.y, m.r, 0, 7); g.fill();
    }
    g.globalAlpha = 1;
    if (!reduce) requestAnimationFrame(frame);
  }
  addEventListener("resize", size);
  size();
  for (let i = 0; i < 60; i++) motes.push(star());
  frame();

  refresh().catch(() => {
    setStage("error", "Offline");
    phrase.textContent = "Tink can't reach its server.";
    why.textContent = "Start it with: uv run uvicorn tink.app:app --reload";
  });
})();
