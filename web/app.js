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
  const isRoom = (s) => s.role === "background";
  const realSounds = () => sounds.filter((s) => !isRoom(s));
  const tintOf = (name) => { const i = realSounds().findIndex((x) => x.name === name); return i < 0 ? "var(--ash)" : `var(--t${i % 6})`; };

  // ---------- API ----------
  // ---------- Profile ----------
  // On a hosted Tink each person has a profile; its id is a private link (?p=...). Locally there is none.
  let profileId = new URLSearchParams(location.search).get("p");
  try { profileId = profileId || localStorage.getItem("tink.profile"); } catch {}
  let isPublic = false;
  const NO_PROFILE = /Create a profile first|No profile with that link|profile link is not valid/;
  function useProfile(id) {
    profileId = id;
    try { localStorage.setItem("tink.profile", id); } catch {}
    const u = new URL(location.href); u.searchParams.set("p", id); history.replaceState(null, "", u);
  }

  async function api(path, opts = {}, ms = 25000) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), ms);
    try {
      const headers = { ...(opts.headers || {}), ...(profileId ? { "X-Tink-Profile": profileId } : {}) };
      const r = await fetch(path, { ...opts, headers, signal: ctl.signal });
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
    voiced = new Set(sounds.filter((s) => s.voice).map((s) => s.name));
    if (!allCues.length) { allCues = r.cues; renderCueBoxes(); }
    llmModel = r.llm.on ? r.llm.model : null;
    isPublic = Boolean(r.profile?.public);
    // The LLM (wording and translation) is optional and off unless a key is set; hide its controls then.
    $("speakInBox").hidden = $("personLangBox").hidden = !llmModel;
    if (!llmModel) $("packetBox").hidden = true;
    $("profileChip").hidden = !isPublic;
    if (isPublic) $("profileName").textContent = r.profile.name;
    renderSelfTest(r.calibration || {});
    $("mode").textContent = llmModel ? `voc2vec + ${llmModel}` : "voc2vec";
    $("leavesNote").textContent = isPublic
      ? `Recordings go only to this Tink server, kept under your private link. Video never leaves your browser.${llmModel ? ` Only the text below goes to ${llmModel}.` : ""}`
      : llmModel
      ? `Audio and video never leave this laptop. Only the text below goes to ${llmModel}.`
      : "Everything stays on this laptop: every sound, every banked voice, every word.";
    renderSounds(lit);
  }

  // Tink tests itself after every change: each recorded take is held out and matched against the rest.
  function renderSelfTest(c) {
    const el = $("selfTest");
    if (!c.ready) {
      el.innerHTML = sounds.length ? "Record at least 2 takes of 2 sounds and Tink will test itself on them." : "";
      return;
    }
    const sure = c.sure_n ? ` When it was 90% sure or more, it was right <b>${c.sure_right} of ${c.sure_n}</b>.` : "";
    el.innerHTML = `Self-test on your ${c.n} takes: right on <b>${c.right}</b> (<span class="conf">${pct(c.accuracy)}</span>).${sure}`;
  }

  // ---------- Language ----------
  const speakSel = $("speakIn");
  speakSel.innerHTML = LANGS.map((l, i) => `<option value="${i}">${esc(l.label)}</option>`).join("");
  try { const saved = localStorage.getItem("tink.speakIn"); if (saved && LANGS[saved]) speakSel.value = saved; } catch {}
  const speakLang = () => LANGS[speakSel.value] || LANGS[0];
  speakSel.addEventListener("change", () => {
    try { localStorage.setItem("tink.speakIn", speakSel.value); } catch {}

  });

  // ---------- Mic recording ----------
  // Records until the sound ends (silenceMs of quiet after something loud) or maxMs passes.
  // Resolves with a 16-bit WAV Blob at the mic's native rate; the backend resamples to 16 kHz.
  let stream, ctx;
  async function recordClip({ maxMs = 4000, silenceMs = 700, startTimeoutMs = 6000, level = 0.02, fixedMs = 0 } = {}) {
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
      const guard = fixedMs
        ? setTimeout(() => finish(), fixedMs) // room noise: record a fixed stretch, whatever the level
        : setTimeout(() => finish(heardAt ? null : new Error("silence")), startTimeoutMs + maxMs + 500);
      proc.onaudioprocess = (e) => {
        const d = new Float32Array(e.inputBuffer.getChannelData(0));
        chunks.push(d);
        let sum = 0; for (const v of d) sum += v * v;
        const rms = Math.sqrt(sum / d.length), now = performance.now();
        setLevel(Math.min(1, rms * 12));
        if (rms > level) { lastLoud = now; heardAt = heardAt || now; }
        if (fixedMs) return;
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

  // ---------- Their own voice ----------
  // Message banking: while the person can still speak, they record each phrase in their own voice.
  // Tink plays that recording instead of a synthetic voice whenever the sound is heard.
  const voiceCache = new Map();
  let voiced = new Set(), player = null, lastVoiceName = null;
  async function playVoice(name) {
    if (muted) return;
    let url = voiceCache.get(name);
    if (!url) {
      const r = await fetch(`/api/voice/${encodeURIComponent(name)}`, { headers: profileId ? { "X-Tink-Profile": profileId } : {} });
      if (!r.ok) throw new Error("no banked voice");
      url = URL.createObjectURL(await r.blob());
      voiceCache.set(name, url);
    }
    player?.pause();
    player = new Audio(url);
    document.body.classList.add("speaking");
    try { await new Promise((res) => { player.onended = player.onerror = res; player.play().catch(res); }); }
    finally { document.body.classList.remove("speaking"); }
  }
  // Their banked voice when there is one (and the words were not reworded), otherwise the computer voice.
  const sayFor = (name, text, lang) => (name && voiced.has(name) && !lang
    ? playVoice(name).catch(() => speak(text, lang))
    : speak(text, lang));

  // ---------- Listen ----------
  const scrim = $("scrim");
  const setStage = (state, status, tint) => {
    stage.className = `stage ${state}`;
    if (statusText.textContent !== status) {
      statusText.textContent = status;
      if (!still) statusText.animate([{ opacity: 0, transform: "translateY(3px)" }, { opacity: 1, transform: "none" }], { duration: 200, easing: EASE });
    }
    if (tint) scrim.style.setProperty("--scrim", tint); else scrim.style.removeProperty("--scrim");
  };
  const clock = () => new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const hideResult = () => { for (const id of ["ranked", "feedback", "taught"]) $(id).hidden = true; };

  // ---------- Motion ----------
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const EASE = "cubic-bezier(.16, 1, .3, 1)";
  // The focal moment: the phrase is cut into the paper. A blade sweeps left to right and the
  // text lands like a paper cut-out settling onto the screen.
  function castPhrase(text, delay = 0) {
    phrase.textContent = text;
    if (still || !text.trim()) return;
    phrase.animate([
      { clipPath: "inset(-12% 100% -12% 0)", transform: "translateY(5px)", filter: "drop-shadow(5px 7px 0 rgba(21,17,14,.28))" },
      { clipPath: "inset(-12% 0 -12% 0)", transform: "translateY(0)", filter: "drop-shadow(0 0 0 rgba(21,17,14,0))" },
    ], { duration: 640, delay, easing: EASE, fill: "backwards" });
    // The blade is placed when the cut starts, after the ranking and the rest of the result have laid out.
    setTimeout(() => requestAnimationFrame(() => {
      const c = phrase.parentElement.getBoundingClientRect(), p = phrase.getBoundingClientRect();
      const blade = document.createElement("span");
      blade.className = "blade";
      Object.assign(blade.style, { left: `${p.left - c.left}px`, top: `${p.top - c.top - 6}px`, height: `${p.height + 12}px` });
      phrase.parentElement.append(blade);
      blade.animate([
        { transform: "translateX(0)", opacity: 1 },
        { transform: `translateX(${p.width}px)`, opacity: 1, offset: 0.85 },
        { transform: `translateX(${p.width}px)`, opacity: 0 },
      ], { duration: 620, easing: EASE, fill: "forwards" }).onfinish = () => blade.remove();
    }), delay);
  }
  // Continuity: the heard sound's swatch flies out of the list and blooms into the whole screen.
  function flyTint(name) {
    const sw = document.querySelector(`.sound[data-name="${CSS.escape(name)}"] .sw`);
    if (still || !sw) return false;
    const a = sw.getBoundingClientRect(), b = scrim.getBoundingClientRect();
    if (!a.width) return false; // list hidden (present mode, phone scrolled away)
    const f = document.createElement("div");
    f.className = "flight";
    f.style.setProperty("--t", tintOf(name));
    Object.assign(f.style, { left: `${a.left}px`, top: `${a.top}px`, width: `${a.width}px`, height: `${a.height}px`, borderRadius: "5px", transformOrigin: "0 0" });
    document.body.append(f);
    const sx = b.width / a.width, sy = b.height / a.height;
    const end = `translate(${b.left - a.left}px, ${b.top - a.top}px) scale(${sx}, ${sy})`;
    // travel reads as a journey (ease in and out), then the colour pours into the screen and fades
    f.animate([
      { transform: "none", borderRadius: "5px", opacity: 1, easing: "cubic-bezier(.5, 0, .2, 1)" },
      { transform: end, borderRadius: `${22 / sx}px / ${22 / sy}px`, opacity: 0.8, offset: 0.68 },
      { transform: end, borderRadius: `${22 / sx}px / ${22 / sy}px`, opacity: 0 },
    ], { duration: 720, easing: "linear" }).onfinish = () => f.remove();
    return true;
  }

  async function listen() {
    if (busy || enrolling || alarmOn || realSounds().length < minSounds) return;
    busy = true; $("listen").disabled = true; $("take").disabled = true;
    $("listen").querySelector(".lbl").textContent = "Listening";
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
    $("listen").querySelector(".lbl").textContent = "Listen";
    busy = false; $("take").disabled = false; $("listen").disabled = alarmOn || realSounds().length < minSounds;
    if ($("loop").checked && !alarmOn) { await wait(lastAsk ? 1500 : 500); listen(); }
    else if (!stage.classList.contains("error") && !alarmOn) statusText.textContent = "Ready";
  }

  function showResult(r) {
    if (r.background) {
      setStage("missed", "Only room noise");
      phrase.textContent = $("loop").checked ? "\u00a0" : "Just room noise.";
      why.textContent = "Tink ignores sounds that match the room.";
      return Promise.resolve();
    }
    if (r.urgent) return startAlarm(r);
    lastAsk = r.ask; lastName = r.name;
    const top = r.ranked[0];
    const heard = r.matched && !r.ask;
    setStage(heard ? "heard" : "missed", r.ask ? "Checking with you" : heard ? "Heard you" : "Not sure", heard ? tintOf(r.name) : null);
    castPhrase(r.say, heard && flyTint(r.name) ? 420 : 0);
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

    $("ranked").innerHTML = r.ranked.slice(0, 3).map((x, i) => `
      <li class="${x.name === "unknown" ? "unknown" : ""}${heard && x.name === r.name ? " lead" : ""}" style="--t:${tintOf(x.name)}; --i:${i}">
        <span class="sw" aria-hidden="true"></span>
        <span>${x.name === "unknown" ? "Didn't catch it" : esc(x.name)}</span>
        <span class="p"><b>${pct(x.p)}</b>${pct(x.p) !== pct(x.audio_p) ? ` · sound alone ${pct(x.audio_p)}` : ""}</span>
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
    lastVoiceName = heard ? r.name : null;
    return wait(r.matched && !r.ask ? 350 : 0).then(() => sayFor(lastVoiceName, r.say, lang));
  }

  function feedbackDone(msg) {
    $("right").hidden = $("wrong").hidden = true;
    $("feedbackMsg").textContent = msg;
  }
  $("right").addEventListener("click", async () => {
    try {
      await api("/api/confirm", { method: "POST", body: new URLSearchParams({ name: lastName }) });
      feedbackDone("Thanks. Tink will remember this time of day.");
      document.querySelector(`.sound[data-name="${CSS.escape(lastName)}"]`)?.classList.add("learned");
    } catch (e) { feedbackDone(e.message); }
  });
  $("wrong").addEventListener("click", () => {
    const last = log[log.length - 1];
    if (last && last.stamp === "Said") { last.stamp = "Wrong"; renderLog(); }
    feedbackDone("Got it. Try the sound again, or teach Tink more takes.");
  });
  $("useTaught").addEventListener("click", () => {
    const text = $("taughtText").textContent;
    castPhrase(text); lastSaid = text; lastLang = null; sayFor(lastName, text);
    $("taught").hidden = true;
    why.insertAdjacentText("beforeend", " · switched to the taught phrase");
  });

  // Every turn stays in the log, stamped. A wrong guess is struck through, never removed.
  function addLog(text, stamp, name) {
    log.push({ t: clock(), text, stamp, name });
    renderLog(true);
  }
  function renderLog(fresh = false) {
    $("log").innerHTML = log.slice(-12).reverse().map((e, i) => `
      <li${fresh && i === 0 ? ' class="fresh"' : ""}><time>${esc(e.t)}</time><span class="text${e.stamp === "Wrong" ? " struck" : ""}">${esc(e.text)}</span>
      <span class="stamp${e.stamp === "Wrong" ? " wrong" : e.stamp === "Urgent" || e.stamp === "Answered" ? " urgent" : e.stamp === "Replied" ? " replied" : ""}" style="--t:${e.name ? tintOf(e.name) : "var(--control-line)"}">${e.stamp}</span></li>`).join("");
    $("logEmpty").hidden = log.length > 0;
  }

  // ---------- Sounds list + first run ----------
  const dots = (n, fresh = false) => Array.from({ length: 5 }, (_, i) => `<i class="${i < n ? (fresh && i === n - 1 ? "new" : "") : "off"}"></i>`).join("");
  function renderSounds(lit) {
    $("sounds").innerHTML = sounds.map((s) => `
      <li class="sound${s.name === lit ? " lit" : ""}" data-name="${esc(s.name)}" style="--t:${tintOf(s.name)}">
        <span class="sw" aria-hidden="true"></span>
        <span class="name">${esc(s.name)}</span>
        <span class="takes" aria-label="${s.takes} takes">${dots(s.takes)}</span>
        ${isRoom(s) ? "<span></span>" : `<button class="bank" data-name="${esc(s.name)}" aria-label="Record them saying ${esc(s.phrase)}" title="Record them saying it"><i class="ph ph-microphone-stage" aria-hidden="true"></i></button>`}
        <button class="forget" data-name="${esc(s.name)}" aria-label="Forget ${esc(s.name)}" title="Forget"><i class="ph ph-trash" aria-hidden="true"></i></button>
        <span class="says">says "${esc(s.phrase)}"</span>
        ${s.cues.length || s.role || s.urgent || s.voice ? `<span class="cuelist">${s.voice ? `<button class="cue voice" data-name="${esc(s.name)}" title="Play their voice"><i class="ph-fill ph-play" aria-hidden="true"></i> own voice</button>` : ""}${s.urgent ? `<span class="cue urgent">urgent</span>` : ""}${isRoom(s) ? `<span class="cue">room noise, ignored</span>` : s.role ? `<span class="cue role">${s.role} answer</span>` : ""}${s.cues.map((c) => `<span class="cue">${cueLabel(c)}</span>`).join("")}</span>` : ""}
      </li>`).join("");
    $("soundsEmpty").hidden = sounds.length > 0;
    const real = realSounds().length, room = sounds.find(isRoom);
    $("count").textContent = real ? `${real} sound${real > 1 ? "s" : ""}` : "";
    $("roomCount").textContent = room ? `${room.takes} take${room.takes === 1 ? "" : "s"} recorded` : "not recorded";

    const ready = real >= minSounds;
    $("listen").disabled = busy || enrolling || !ready;
    $("guide").hidden = ready;
    $("guideCount").textContent = `${Math.min(real, minSounds)} of ${minSounds} taught`;
    $("guide").firstElementChild.classList.toggle("done", ready);
    if (!busy && !ready) {
      setStage("idle", "Ready"); hideResult();
      phrase.textContent = real ? "One more sound to go." : "Teach Tink two sounds.";
      why.textContent = "Tink compares sounds against each other, so it needs at least two.";
    } else if (!busy && stage.classList.contains("idle")) {
      phrase.textContent = "Make a sound Tink knows.";
      why.textContent = "Press Listen, or turn on Keep listening.";
    }
  }
  // Forgetting deletes the recorded takes, so it asks once, inline, and waits for an answer.
  $("sounds").addEventListener("click", async (e) => {
    const bank = e.target.closest(".bank"), play = e.target.closest(".cue.voice");
    if (bank) return bankVoice(bank.dataset.name, bank);
    if (play) return playVoice(play.dataset.name).catch(() => {});
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
  const drawTakes = (fresh = false) => { $("enrollDots").innerHTML = enrollTakes ? dots(enrollTakes, fresh) + `<span class="help">&nbsp;${enrollTakes} of 5</span>` : ""; };
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
      fd.append("urgent", $("surgent").checked ? "true" : "false");
      fd.append("audio", clip, "take.wav");
      const r = await api("/api/enroll", { method: "POST", body: fd });
      enrollTakes = r.takes; drawTakes(true); chime(true);
      $("sname").readOnly = true; $("done").hidden = false; $("bankNow").hidden = false;
      await refresh(name);
    } catch (err) {
      $("err").textContent = micError(err);
    }
    enrolling = false; $("enroll").classList.remove("recording");
    $("take").disabled = false; $("listen").disabled = realSounds().length < minSounds;
    takeLabel(enrollTakes >= 5 ? "Record another take" : enrollTakes ? "Record next take" : "Record take");
  });
  $("done").addEventListener("click", () => {
    enrollTakes = 0; drawTakes(); $("enroll").reset(); $("surgent").checked = false; $("bankNow").hidden = true; $("sname").readOnly = false; $("done").hidden = true; $("err").textContent = "";
    takeLabel("Record take"); $("sname").focus();
  });

  // ---------- Answer back ----------
  // The caregiver types; Tink translates into the person's language (with Grok) and says it to them.
  const personSel = $("personLang");
  personSel.innerHTML = LANGS.map((l, i) => `<option value="${i}">${esc(l.label)}</option>`).join("");
  try { const saved = localStorage.getItem("tink.personLang"); if (saved && LANGS[saved]) personSel.value = saved; } catch {}
  personSel.addEventListener("change", () => { try { localStorage.setItem("tink.personLang", personSel.value); } catch {} });
  const personLang = () => LANGS[personSel.value] || LANGS[0];

  async function sendReply(text) {
    text = text.trim();
    if (!text) { $("replyErr").textContent = "Type what you want to say first."; return; }
    if (busy || enrolling || alarmOn) { $("replyErr").textContent = "Wait until Tink finishes listening, then say it."; return; }
    $("replyErr").textContent = "";
    busy = true; $("replySend").disabled = true; $("listen").disabled = true;
    setStage("reply", llmModel && personLang().name !== "English" ? `Translating into ${personLang().name}` : "Saying it to them");
    try {
      const r = await api("/api/reply", { method: "POST", body: new URLSearchParams({ text, to: personLang().name }) });
      hideResult(); $("guide").hidden = true;
      statusText.textContent = "To them";
      castPhrase(r.say);
      why.innerHTML = r.say !== r.typed
        ? `You typed: <b>${esc(r.typed)}</b> · translated into ${esc(r.to)} by ${esc(llmModel || "Grok")}`
        : "From you";
      if (r.packet) { $("packet").textContent = JSON.stringify(r.packet, null, 2); $("packetBox").hidden = false; }
      lastSaid = r.say; lastLang = r.source === "grok" ? personLang().lang : null; $("repeat").disabled = false;
      addLog(r.say, "Replied", null);
      $("replyText").value = "";
      await speak(r.say, lastLang);
    } catch (e) {
      $("replyErr").textContent = e.message; setStage("idle", "Ready");
    }
    busy = false; $("replySend").disabled = false; $("listen").disabled = realSounds().length < minSounds;
    if ($("loop").checked) setTimeout(listen, 600);
  }
  $("replyForm").addEventListener("submit", (e) => { e.preventDefault(); sendReply($("replyText").value); });
  $("quickReplies").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) sendReply(b.textContent); });

  async function bankVoice(name, button) {
    const s = sounds.find((x) => x.name === name);
    if (!s) return;
    if (busy || enrolling || alarmOn) { $("err").textContent = "Turn off Keep listening first, then record their voice."; return; }
    enrolling = true; button?.classList.add("recording"); $("listen").disabled = true; hideResult();
    setStage("listening", "Recording their voice");
    castPhrase(s.phrase);
    why.textContent = "Ask them to say this out loud, in their own voice. Tink stops when they finish.";
    try {
      const clip = await recordClip({ maxMs: 8000, silenceMs: 1100, startTimeoutMs: 8000 });
      const fd = new FormData();
      fd.append("name", name); fd.append("audio", clip, "voice.wav");
      await api("/api/voice", { method: "POST", body: fd });
      voiceCache.delete(name);
      chime(true);
      await refresh(name);
      setStage("idle", "Voice banked");
      why.textContent = `Saved. When Tink hears "${s.name.toLowerCase()}", it will play this recording.`;
    } catch (err) {
      setStage("error", "Not saved"); why.textContent = micError(err);
    }
    enrolling = false; button?.classList.remove("recording"); $("listen").disabled = realSounds().length < minSounds;
  }
  $("bankNow").addEventListener("click", (e) => bankVoice($("sname").value.trim(), e.currentTarget));

  // ---------- Room noise ----------
  $("roomTake").addEventListener("click", async () => {
    if (busy || enrolling) { $("roomErr").textContent = "Turn off Keep listening first, then record."; return; }
    const b = $("roomTake"), label = b.querySelector("span");
    enrolling = true; b.disabled = true; $("roomErr").textContent = ""; label.textContent = "Recording the room, stay quiet";
    b.classList.add("filling");
    try {
      const clip = await recordClip({ fixedMs: 3000 });
      const fd = new FormData();
      fd.append("name", "Room noise"); fd.append("phrase", "(room noise)"); fd.append("role", "background");
      fd.append("audio", clip, "room.wav");
      await api("/api/enroll", { method: "POST", body: fd });
      await refresh();
    } catch (err) { $("roomErr").textContent = micError(err); }
    enrolling = false; b.disabled = false; b.classList.remove("filling"); label.textContent = "Record 3 more seconds";
    $("listen").disabled = realSounds().length < minSounds;
  });

  // ---------- Urgent alarm ----------
  // An urgent sound turns the screen alarm red, sounds a two-tone alarm, and repeats the phrase
  // until someone presses "I'm here". Keep listening pauses until then.
  let alarmOn = false, alarmTimer = null, alarmEntry = null;
  function siren() {
    try {
      ctx = ctx || new AudioContext();
      const now = ctx.currentTime;
      [880, 660, 880, 660].forEach((f, i) => {
        const o = ctx.createOscillator(), g = ctx.createGain(), t = now + i * 0.3;
        o.type = "triangle"; o.frequency.value = f;
        g.gain.setValueAtTime(0.0001, t); g.gain.linearRampToValueAtTime(0.2, t + 0.02); g.gain.setValueAtTime(0.2, t + 0.26); g.gain.linearRampToValueAtTime(0.0001, t + 0.3);
        o.connect(g).connect(ctx.destination); o.start(t); o.stop(t + 0.32);
      });
    } catch {}
  }
  async function startAlarm(r) {
    alarmOn = true; lastName = r.name; lastAsk = false;
    document.body.classList.add("alarm");
    setStage("heard", "Urgent", "var(--alarm)");
    castPhrase(r.say);
    why.innerHTML = `Heard <b>${esc(r.name.toLowerCase())}</b>, marked urgent · <span class="conf">${pct(r.p)}</span> sure`;
    hideResult(); $("guide").hidden = true; $("alarmBox").hidden = false; $("listen").disabled = true;
    lastSaid = r.say; lastLang = null; lastVoiceName = r.name; $("repeat").disabled = false;
    addLog(r.say, "Urgent", r.name); alarmEntry = log[log.length - 1];
    renderSounds(r.name);
    $("ack").focus();
    const ring = async () => { if (!alarmOn) return; siren(); await wait(1300); if (alarmOn) await sayFor(r.name, r.say); if (alarmOn) alarmTimer = setTimeout(ring, 4000); };
    ring();
  }
  $("ack").addEventListener("click", () => {
    alarmOn = false; clearTimeout(alarmTimer);
    try { speechSynthesis.cancel(); } catch {}
    document.body.classList.remove("alarm", "speaking"); $("alarmBox").hidden = true; $("listen").disabled = realSounds().length < minSounds;
    if (alarmEntry) { alarmEntry.stamp = "Answered"; renderLog(); }
    setStage("idle", "Ready");
    why.textContent = "Someone answered. Tink is listening again when you are ready.";
    if ($("loop").checked) setTimeout(listen, 800);
  });

  // ---------- Controls ----------
  $("listen").addEventListener("click", listen);
  $("loop").addEventListener("change", (e) => { if (e.target.checked) listen(); });
  $("repeat").addEventListener("click", () => { chime(true); sayFor(lastVoiceName, lastSaid, lastLang); });
  $("mute").addEventListener("click", (e) => {
    const b = e.currentTarget;
    muted = !muted; b.setAttribute("aria-pressed", muted); b.querySelector("span").textContent = muted ? "Voice muted" : "Mute voice";
    if (muted) try { speechSynthesis.cancel(); } catch {}
  });
  function togglePresent(on = !document.body.classList.contains("present")) {
    if (!still && document.startViewTransition && !togglePresent.busy) {
      togglePresent.busy = true;
      document.startViewTransition(() => applyPresent(on)).finished.finally(() => { togglePresent.busy = false; });
    } else applyPresent(on);
  }
  function applyPresent(on) {
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
  // Drawn every frame from the mic level: a soft hotspot that swells with the voice and
  // blooms when Tink understands. The screen's grid is CSS, so no texture is drawn here.
  const cv = $("dust"), g = cv.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let W = 0, H = 0, glow = 0, bloom = 0, t = 0;
  function size() {
    const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
    W = r.width; H = r.height; cv.width = W * d; cv.height = H * d; g.setTransform(d, 0, 0, d, 0, 0);
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
    edge.addColorStop(0, "rgba(17,17,17,0)"); edge.addColorStop(1, `rgba(17,17,17,${lit ? 0.08 : 0.04})`);
    g.fillStyle = edge; g.fillRect(0, 0, W, H);
    if (!reduce) requestAnimationFrame(frame);
  }
  new ResizeObserver(size).observe(scrim);
  size();
  frame();

  // ---------- Start ----------
  function showOnboarding(msg = "") {
    stage.classList.add("onboarding"); $("onboard").hidden = false; $("onboardErr").textContent = msg;
    $("onboardName").focus();
  }
  $("onboard").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("onboardName").value.trim();
    if (!name) { $("onboardErr").textContent = "Type their first name to start."; return; }
    try {
      profileId = null;
      const r = await api("/api/profiles", { method: "POST", body: new URLSearchParams({ name }) });
      useProfile(r.id);
      stage.classList.remove("onboarding"); $("onboard").hidden = true;
      await refresh();
    } catch (err) { $("onboardErr").textContent = err.message; }
  });
  $("newProfile").addEventListener("click", () => { $("onboard").reset(); showOnboarding(); });
  $("copyLink").addEventListener("click", async () => {
    const link = `${location.origin}${location.pathname}?p=${encodeURIComponent(profileId)}`;
    const label = $("copyLink").querySelector("span");
    try { await navigator.clipboard.writeText(link); label.textContent = "Link copied"; }
    catch { prompt("Copy this private link:", link); }
    setTimeout(() => { label.textContent = "Copy private link"; }, 2000);
  });

  (async function start() {
    try {
      await refresh();
      if (profileId && isPublic) useProfile(profileId);
    } catch (e) {
      if (!NO_PROFILE.test(e.message)) {
        // No server behind this page (a static copy, or the server is down): say how to run
        // it instead of leaving controls that can only fail.
        document.body.classList.add("no-server");
        setStage("error", "Not running here");
        phrase.textContent = "Tink runs on your laptop.";
        why.innerHTML = 'There is no Tink server behind this page. The model is too big to host here, so run it yourself:<br><code>git clone https://github.com/hammy229/tink && cd tink && uv run uvicorn tink.app:app</code>';
        return;
      }
      // A dead or stale link: forget it, then try without one (a local Tink needs none).
      const hadLink = Boolean(profileId);
      profileId = null;
      try { localStorage.removeItem("tink.profile"); } catch {}
      try { await refresh(); } catch { showOnboarding(hadLink ? "That link is not valid anymore. Start a new profile." : ""); }
    }
  })();
})();
