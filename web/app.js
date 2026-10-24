// Tink frontend: records a clip (and webcam cues), sends both to the backend, shows the
// Bayesian ranking, and speaks what the LLM (or the taught phrase) says, in the chosen language.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const pct = (p) => `${Math.round(p * 100)}%`;
  const cueLabel = (c) => c.replace("_", " ");
  const stage = $("stage"), phrase = $("phrase"), why = $("why"), statusText = $("statusText");
  const HINT = $("hint").textContent;

  // Languages Tink can speak in. `name` goes to the LLM, `lang` picks the speech voice.
  const LANGS = [
    ["English", "English", "en-US"], ["Español", "Spanish", "es-ES"], ["हिन्दी", "Hindi", "hi-IN"],
    ["中文", "Mandarin Chinese", "zh-CN"], ["Tiếng Việt", "Vietnamese", "vi-VN"], ["العربية", "Arabic", "ar-SA"],
    ["Tagalog", "Tagalog", "fil-PH"], ["한국어", "Korean", "ko-KR"], ["Français", "French", "fr-FR"],
    ["Português", "Portuguese", "pt-BR"], ["اردو", "Urdu", "ur-PK"], ["తెలుగు", "Telugu", "te-IN"],
    ["தமிழ்", "Tamil", "ta-IN"], ["ગુજરાતી", "Gujarati", "gu-IN"],
  ].map(([label, name, lang]) => ({ label, name, lang }));

  let sounds = [], threshold = 0, minSounds = 2, allCues = [], llmModel = null;
  let muted = false, lastSaid = null, lastLang = null, lastName = null, lastAsk = false;
  let busy = false, enrolling = false;
  const log = [];
  const tintOf = (name) => { const i = sounds.findIndex((x) => x.name === name); return i < 0 ? "var(--ash)" : `var(--t${i % 6})`; };

  // ---------- API ----------
  async function api(path, opts = {}, ms = 25000) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), ms);
    try {
      const r = await fetch(path, { ...opts, signal: ctl.signal });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.detail || `Server error ${r.status}`);
      return body;
    } catch (e) {
      if (e.name === "AbortError") throw new Error("Tink's server took too long to answer. Check that it is running.");
      if (e instanceof TypeError) throw new Error("Tink can't reach its server. Start it with: uv run uvicorn tink.app:app");
      throw e;
    } finally { clearTimeout(timer); }
  }
  async function refresh(lit) {
    const r = await api("/api/sounds");
    sounds = r.sounds; threshold = r.threshold; minSounds = r.min_sounds || 2;
    if (!allCues.length) { allCues = r.cues; renderCueBoxes(); }
    llmModel = r.llm.on ? r.llm.model : null;
    $("mode").textContent = llmModel ? `voc2vec + ${llmModel}` : "voc2vec";
    $("leavesNote").textContent = llmModel
      ? `Audio and video never leave this laptop. Only the text below goes to ${llmModel}.`
      : "Nothing leaves this laptop. Add an xAI key and Grok will word and translate what Tink says.";
    renderSounds(lit);
  }

  // ---------- Language ----------
  const speakSel = $("speakIn");
  speakSel.innerHTML = LANGS.map((l, i) => `<option value="${i}">${esc(l.label)}</option>`).join("");
  try { const saved = localStorage.getItem("tink.speakIn"); if (saved && LANGS[saved]) speakSel.value = saved; } catch {}
  const speakLang = () => LANGS[speakSel.value] || LANGS[0];
  speakSel.addEventListener("change", () => {
    try { localStorage.setItem("tink.speakIn", speakSel.value); } catch {}
    if (!llmModel && speakLang().name !== "English") $("hint").textContent = "Translation needs Grok. Without a key, Tink says the taught phrase as written.";
  });

  // ---------- Mic recording ----------
  // Records until the sound ends (silenceMs of quiet after something loud) or maxMs passes.
  // Resolves with a 16-bit WAV Blob at the mic's native rate; the backend resamples to 16 kHz.
  let stream, ctx;
  async function recordClip({ maxMs = 4000, silenceMs = 700, startTimeoutMs = 6000, level = 0.02 } = {}) {
    if (!stream || stream.getAudioTracks().every((t) => t.readyState === "ended")) {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true },
      });
    }
    ctx = ctx || new AudioContext();
    if (ctx.state === "suspended") await ctx.resume();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(2048, 1, 1);
    const chunks = [];
    const t0 = performance.now();
    let heardAt = 0, lastLoud = 0, done = false;

    return new Promise((resolve, reject) => {
      const finish = (err) => {
        if (done) return;
        done = true; clearTimeout(guard);
        proc.onaudioprocess = null; src.disconnect(); proc.disconnect(); setLevel(0);
        if (err) return reject(err);
        resolve(encodeWav(chunks, ctx.sampleRate));
      };
      // If the audio callback never fires (suspended context, unplugged mic), don't hang forever.
      const guard = setTimeout(() => finish(heardAt ? null : new Error("silence")), startTimeoutMs + maxMs + 500);
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
  // The live mic level drives the light, the bars, and nothing else.
  let level = 0;
  const setLevel = (v) => { level = v; stage.style.setProperty("--lvl", v.toFixed(2)); };
  function micError(e) {
    if (e.message === "silence") return "Tink didn't hear anything. Move closer to the mic and try again.";
    if (e.name === "NotAllowedError") return "Tink needs permission. Allow it in the browser's address bar, then try again.";
    if (e.name === "NotFoundError") return "No microphone or camera found. Plug one in and try again.";
    return e.message;
  }

  // ---------- Camera cues ----------
  $("camera").addEventListener("change", async (e) => {
    $("camNote").hidden = !e.target.checked;
    if (!e.target.checked) { window.TinkCues?.disable(); $("cam").hidden = true; return; }
    if (!window.TinkCues) { e.target.checked = false; $("camNote").hidden = true; $("hint").textContent = "The camera module is still loading. Try again in a second."; return; }
    $("cam").hidden = false;
    $("liveCues").innerHTML = `<span class="cue">starting camera</span>`;
    try {
      await window.TinkCues.enable($("video"), (live, face) => {
        $("liveCues").innerHTML = face
          ? allCues.map((c) => `<span class="cue${live.includes(c) ? " on" : ""}">${cueLabel(c)}</span>`).join("")
          : `<span class="cue">no face in view</span>`;
      });
    } catch (err) {
      e.target.checked = false; $("cam").hidden = true; $("camNote").hidden = true;
      $("hint").textContent = micError(err);
    }
  });

  // ---------- Chime + voice ----------
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
  // Resolves when Tink has finished speaking, so Keep listening never records Tink's own voice.
  function speak(text, lang) {
    return new Promise((resolve) => {
      if (muted || !text || !("speechSynthesis" in window)) return resolve();
      try {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 0.95;
        if (lang) {
          u.lang = lang;
          const v = speechSynthesis.getVoices().find((x) => x.lang === lang) || speechSynthesis.getVoices().find((x) => x.lang.startsWith(lang.slice(0, 2)));
          if (v) u.voice = v;
        }
        document.body.classList.add("speaking");
        const done = () => { clearTimeout(cap); document.body.classList.remove("speaking"); resolve(); };
        const cap = setTimeout(done, Math.max(3000, text.length * 110));
        u.onend = u.onerror = done;
        speechSynthesis.speak(u);
      } catch { resolve(); }
    });
  }
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  // ---------- Listen ----------
  const scrim = $("scrim");
  const setStage = (state, status, tint) => {
    stage.className = `stage ${state}`; statusText.textContent = status;
    if (tint) scrim.style.setProperty("--scrim", tint); else scrim.style.removeProperty("--scrim");
  };
  const clock = () => new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const hideResult = () => { for (const id of ["ranked", "feedback", "taught"]) $(id).hidden = true; };

  async function listen() {
    if (busy || enrolling || sounds.length < minSounds) return;
    busy = true; $("listen").disabled = true; $("take").disabled = true;
    setStage("listening", "Listening");
    phrase.textContent = " "; why.textContent = ""; hideResult();
    renderSounds();
    let spoken = Promise.resolve();
    try {
      window.TinkCues?.isOn() && window.TinkCues.begin();
      let clip, cues = [];
      try { clip = await recordClip(); } finally { cues = window.TinkCues?.end() ?? []; }
      setStage("listening", "Matching the sound");
      const slow = llmModel && setTimeout(() => { statusText.textContent = `Wording it with ${llmModel}`; }, 700);
      const fd = new FormData();
      fd.append("audio", clip, "clip.wav");
      fd.append("cues", JSON.stringify(cues));
      fd.append("speak_in", speakLang().name);
      try { spoken = showResult(await api("/api/recognize", { method: "POST", body: fd })); } finally { clearTimeout(slow); }
    } catch (e) {
      if (e.message === "silence" && $("loop").checked) { busy = false; $("take").disabled = false; return listen(); }
      setStage("error", "Stopped");
      phrase.textContent = e.message === "silence" ? "I didn't hear anything." : "Something went wrong.";
      why.textContent = micError(e);
      $("loop").checked = false;
    }
    await spoken;
    busy = false; $("take").disabled = false; $("listen").disabled = sounds.length < minSounds;
    if ($("loop").checked) { await wait(lastAsk ? 1500 : 500); listen(); }
    else if (!stage.classList.contains("error")) statusText.textContent = "Ready";
  }

  function showResult(r) {
    lastAsk = r.ask; lastName = r.name;
    const top = r.ranked[0];
    const heard = r.matched && !r.ask;
    setStage(heard ? "heard" : "missed", r.ask ? "Checking with you" : heard ? "Heard you" : "Not sure", heard ? tintOf(r.name) : null);
    phrase.textContent = r.say;
    const cueText = r.packet.webcam_cues.length ? ` · saw ${r.packet.webcam_cues.map(cueLabel).join(", ")}` : "";
    const via = r.source === "grok" ? ` · worded by ${llmModel}` : "";
    const d = r.dialog || {};
    why.innerHTML = d.asking
      ? `${d.answered === "no" ? "Okay, not that. " : ""}Answer with your <b>yes</b> or <b>no</b> sound.`
      : d.answered === "yes" ? `You said <b>yes</b>.`
      : d.answered === "no" ? `You said <b>no</b>.`
      : r.matched
      ? `Heard <b>${esc(top.name.toLowerCase())}</b> · <span class="conf">${pct(top.p)}</span> sure${cueText}${via}`
      : `Nothing matched closely enough${cueText}`;
    $("hint").textContent = d.asking && !$("loop").checked ? "Press Listen (or space), then make the yes or no sound." : HINT;

    $("ranked").innerHTML = r.ranked.slice(0, 3).map((x) => `
      <li class="${x.name === "unknown" ? "unknown" : ""}" style="--t:${tintOf(x.name)}">
        <span class="sw" aria-hidden="true"></span>
        <span>${x.name === "unknown" ? "Didn't catch it" : esc(x.name)}</span>
        <span class="p"><b>${pct(x.p)}</b> · sound alone ${pct(x.audio_p)}</span>
        <span class="bar" style="width:${Math.max(1.5, x.p * 100)}%"></span>
      </li>`).join("");
    $("ranked").hidden = Boolean(d.answered);
    $("packet").textContent = JSON.stringify(r.packet, null, 2);
    $("packetBox").hidden = !llmModel;
    $("feedback").hidden = !r.matched;
    $("right").hidden = $("wrong").hidden = false;
    $("feedbackMsg").textContent = "Did Tink get it right?";

    // When Grok rewords or translates, show the taught phrase with a way back to it.
    $("taught").hidden = !(r.source === "grok" && r.matched && !r.ask && r.taught && r.taught !== r.say);
    $("taughtText").textContent = r.taught || "";

    const lang = r.source === "grok" ? speakLang().lang : null;
    lastSaid = r.say; lastLang = lang; $("repeat").disabled = false;
    addLog(r.say, r.ask ? "Asked" : r.matched ? "Said" : "Not caught", r.matched ? r.name : null);
    if (r.matched) renderSounds(r.name);
    chime(r.matched && !r.ask);
    if (heard) burst();
    return wait(r.matched && !r.ask ? 350 : 0).then(() => speak(r.say, lang));
  }

  function feedbackDone(msg) {
    $("right").hidden = $("wrong").hidden = true;
    $("feedbackMsg").textContent = msg;
  }
  $("right").addEventListener("click", async () => {
    try {
      await api("/api/confirm", { method: "POST", body: new URLSearchParams({ name: lastName }) });
      feedbackDone("Thanks. Tink will remember this time of day.");
    } catch (e) { feedbackDone(e.message); }
  });
  $("wrong").addEventListener("click", () => {
    const last = log[log.length - 1];
    if (last && last.stamp === "Said") { last.stamp = "Wrong"; renderLog(); }
    feedbackDone("Got it. Try the sound again, or teach Tink more takes.");
  });
  $("useTaught").addEventListener("click", () => {
    const text = $("taughtText").textContent;
    phrase.textContent = text; lastSaid = text; lastLang = null; speak(text);
    $("taught").hidden = true;
    why.insertAdjacentText("beforeend", " · switched to the taught phrase");
  });

  // Every turn stays in the log, stamped. A wrong guess is struck through, never removed.
  function addLog(text, stamp, name) {
    log.push({ t: clock(), text, stamp, name });
    renderLog();
  }
  function renderLog() {
    $("log").innerHTML = log.slice(-12).reverse().map((e) => `
      <li><time>${esc(e.t)}</time><span class="text${e.stamp === "Wrong" ? " struck" : ""}">${esc(e.text)}</span>
      <span class="stamp${e.stamp === "Wrong" ? " wrong" : ""}" style="--t:${e.name ? tintOf(e.name) : "var(--control-line)"}">${e.stamp}</span></li>`).join("");
    $("logEmpty").hidden = log.length > 0;
  }

  // ---------- Sounds list + first run ----------
  const dots = (n) => Array.from({ length: 5 }, (_, i) => `<i class="${i < n ? "" : "off"}"></i>`).join("");
  function renderSounds(lit) {
    $("sounds").innerHTML = sounds.map((s) => `
      <li class="sound${s.name === lit ? " lit" : ""}" style="--t:${tintOf(s.name)}">
        <span class="sw" aria-hidden="true"></span>
        <span class="name">${esc(s.name)}</span>
        <span class="takes" aria-label="${s.takes} takes">${dots(s.takes)}</span>
        <button class="forget" data-name="${esc(s.name)}" aria-label="Forget ${esc(s.name)}" title="Forget"><i class="ph ph-trash" aria-hidden="true"></i></button>
        <span class="says">says "${esc(s.phrase)}"</span>
        ${s.cues.length || s.role ? `<span class="cuelist">${s.role ? `<span class="cue role">${s.role} answer</span>` : ""}${s.cues.map((c) => `<span class="cue">${cueLabel(c)}</span>`).join("")}</span>` : ""}
      </li>`).join("");
    $("soundsEmpty").hidden = sounds.length > 0;
    $("count").textContent = sounds.length ? `${sounds.length} sound${sounds.length > 1 ? "s" : ""}` : "";

    const ready = sounds.length >= minSounds;
    $("listen").disabled = busy || enrolling || !ready;
    $("guide").hidden = ready;
    $("guideCount").textContent = `${Math.min(sounds.length, minSounds)} of ${minSounds} taught`;
    $("guide").firstElementChild.classList.toggle("done", ready);
    if (!busy && !ready) {
      setStage("idle", "Ready"); hideResult();
      phrase.textContent = sounds.length ? "One more sound to go." : "Teach Tink two sounds.";
      why.textContent = "Tink compares sounds against each other, so it needs at least two.";
    } else if (!busy && stage.classList.contains("idle")) {
      phrase.textContent = "Make a sound Tink knows.";
      why.textContent = "Press Listen, or turn on Keep listening.";
    }
  }
  // Forgetting deletes the recorded takes, so it asks once, inline, and waits for an answer.
  $("sounds").addEventListener("click", async (e) => {
    const forget = e.target.closest(".forget"), yes = e.target.closest(".confirm-yes"), no = e.target.closest(".confirm-no");
    if (forget) {
      const li = forget.closest(".sound");
      if (li.querySelector(".confirm-row")) return;
      const s = sounds.find((x) => x.name === forget.dataset.name);
      li.insertAdjacentHTML("beforeend", `
        <div class="confirm-row" role="group" aria-label="Confirm forget">
          <span>Forget "${esc(s.name)}" and its ${s.takes} recorded take${s.takes === 1 ? "" : "s"}?</span>
          <button class="btn danger small confirm-yes" data-name="${esc(s.name)}">Forget</button>
          <button class="btn quiet small confirm-no">Cancel</button>
        </div>`);
      li.querySelector(".confirm-no").focus();
    } else if (no) {
      no.closest(".confirm-row").remove();
    } else if (yes) {
      try { await api(`/api/sounds/${encodeURIComponent(yes.dataset.name)}`, { method: "DELETE" }); await refresh(); }
      catch (err) { $("err").textContent = err.message; }
    }
  });

  // ---------- Teach ----------
  function renderCueBoxes() {
    $("cueBoxes").innerHTML = allCues.map((c) =>
      `<label><input type="checkbox" name="cue" value="${c}" id="cue-${c}"> ${cueLabel(c)}</label>`).join("");
  }
  let enrollTakes = 0;
  const drawTakes = () => { $("enrollDots").innerHTML = enrollTakes ? dots(enrollTakes) + `<span class="help">&nbsp;${enrollTakes} of 5</span>` : ""; };
  const takeLabel = (t) => { $("take").querySelector("span").textContent = t; };
  $("enroll").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (busy) { $("err").textContent = "Turn off Keep listening first, then record."; return; }
    const name = $("sname").value.trim(), says = $("sphrase").value.trim();
    if (!name || !says) { $("err").textContent = "Add both the sound and what it means."; return; }
    $("err").textContent = "";
    enrolling = true; $("listen").disabled = true; $("take").disabled = true;
    $("enroll").classList.add("recording"); takeLabel("Make the sound now");
    try {
      const clip = await recordClip({ maxMs: 4000 });
      $("enroll").classList.remove("recording"); takeLabel("Saving");
      const cues = [...document.querySelectorAll("#cueBoxes input:checked")].map((i) => i.value);
      const fd = new FormData();
      fd.append("name", name); fd.append("phrase", says); fd.append("cues", JSON.stringify(cues));
      fd.append("role", document.querySelector('#roleBox input:checked')?.value || "");
      fd.append("audio", clip, "take.wav");
      const r = await api("/api/enroll", { method: "POST", body: fd });
      enrollTakes = r.takes; drawTakes(); chime(true);
      $("sname").readOnly = true; $("done").hidden = false;
      await refresh(name);
    } catch (err) {
      $("err").textContent = micError(err);
    }
    enrolling = false; $("enroll").classList.remove("recording");
    $("take").disabled = false; $("listen").disabled = sounds.length < minSounds;
    takeLabel(enrollTakes >= 5 ? "Record another take" : enrollTakes ? "Record next take" : "Record take");
  });
  $("done").addEventListener("click", () => {
    enrollTakes = 0; drawTakes(); $("enroll").reset(); $("sname").readOnly = false; $("done").hidden = true; $("err").textContent = "";
    takeLabel("Record take"); $("sname").focus();
  });

  // ---------- Controls ----------
  $("listen").addEventListener("click", listen);
  $("loop").addEventListener("change", (e) => { if (e.target.checked) listen(); });
  $("repeat").addEventListener("click", () => { chime(true); speak(lastSaid, lastLang); });
  $("mute").addEventListener("click", (e) => {
    const b = e.currentTarget;
    muted = !muted; b.setAttribute("aria-pressed", muted); b.querySelector("span").textContent = muted ? "Voice muted" : "Mute voice";
    if (muted) try { speechSynthesis.cancel(); } catch {}
  });
  function togglePresent(on = !document.body.classList.contains("present")) {
    document.body.classList.toggle("present", on);
    $("present").setAttribute("aria-pressed", on);
    $("present").querySelector("i").className = on ? "ph ph-corners-in" : "ph ph-corners-out";
    $("present").querySelector("span").textContent = on ? "Exit" : "Present";
    requestAnimationFrame(size);
  }
  $("present").addEventListener("click", () => togglePresent());
  document.addEventListener("keydown", (e) => {
    const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
    if (e.code === "Space" && !e.repeat && !typing && !/BUTTON|SUMMARY/.test(document.activeElement.tagName)) { e.preventDefault(); listen(); }
    else if (e.key.toLowerCase() === "p" && !typing && !e.metaKey && !e.ctrlKey) togglePresent();
    else if (e.key === "Escape" && document.body.classList.contains("present")) togglePresent(false);
  });

  // ---------- The light behind the screen ----------
  // Drawn every frame from the mic level: a warm hotspot that swells with the voice and
  // blooms when Tink understands. A faint paper fibre texture is generated once.
  const cv = $("dust"), g = cv.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let W = 0, H = 0, glow = 0, bloom = 0, t = 0, fibres = null;
  function size() {
    const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
    W = r.width; H = r.height; cv.width = W * d; cv.height = H * d; g.setTransform(d, 0, 0, d, 0, 0);
    fibres = document.createElement("canvas"); fibres.width = W; fibres.height = H;
    const f = fibres.getContext("2d");
    f.strokeStyle = "rgba(21,17,14,.07)"; f.lineWidth = 1;
    for (let i = 0; i < (W * H) / 900; i++) {
      const x = Math.random() * W, y = Math.random() * H, a = Math.random() * Math.PI, l = 4 + Math.random() * 12;
      f.beginPath(); f.moveTo(x, y); f.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l); f.stroke();
    }
    if (reduce) frame();
  }
  function burst() { bloom = 1; if (reduce) frame(); }
  function frame() {
    t += 1;
    const lit = stage.classList.contains("listening") || stage.classList.contains("heard");
    glow += ((lit ? 0.45 : 0.18) + level * 0.9 - glow) * 0.18;
    bloom *= 0.965;
    g.clearRect(0, 0, W, H);
    const x = W / 2 + Math.sin(t / 90) * W * 0.04, y = H * 0.44 + Math.cos(t / 130) * H * 0.03;
    const r = Math.max(W, H) * (0.22 + glow * 0.32 + bloom * 0.45);
    const a = Math.min(0.92, 0.35 + glow * 0.5 + bloom * 0.3);
    const light = g.createRadialGradient(x, y, 0, x, y, r);
    light.addColorStop(0, `rgba(255,252,242,${a})`);
    light.addColorStop(0.45, `rgba(255,248,230,${a * 0.45})`);
    light.addColorStop(1, "rgba(255,248,230,0)");
    g.fillStyle = light; g.fillRect(0, 0, W, H);
    const edge = g.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.75);
    edge.addColorStop(0, "rgba(21,17,14,0)"); edge.addColorStop(1, "rgba(21,17,14,.32)");
    g.fillStyle = edge; g.fillRect(0, 0, W, H);
    if (fibres) g.drawImage(fibres, 0, 0);
    if (!reduce) requestAnimationFrame(frame);
  }
  new ResizeObserver(size).observe(scrim);
  size();
  frame();

  refresh().catch((e) => {
    setStage("error", "Offline");
    phrase.textContent = "Tink can't reach its server.";
    why.textContent = e.message;
  });
})();
