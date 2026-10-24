// The Neverland layer: a lightweight starfield, pixie-dust pointer trails, and
// scroll reveals. It is decorative only and switches itself off for reduced motion.
(() => {
  const canvas = document.getElementById("magicDust");
  if (!canvas) return;

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const revealTargets = document.querySelectorAll(
    ".manifesto-grid, .section-heading, .step-card, .demo-intro > *, .privacy > *, .tech-top > *, .pipeline, .final-cta > *"
  );

  if (reduce) {
    revealTargets.forEach((el) => el.classList.add("revealed"));
    return;
  }

  document.documentElement.classList.add("motion-ready");
  revealTargets.forEach((el, i) => {
    el.classList.add("reveal-target");
    el.style.setProperty("--reveal-delay", `${(i % 3) * 90}ms`);
  });

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("revealed");
      observer.unobserve(entry.target);
    }
  }, { rootMargin: "0px 0px -10%", threshold: 0.08 });
  revealTargets.forEach((el) => observer.observe(el));

  const ctx = canvas.getContext("2d", { alpha: true });
  const stars = [];
  const sparks = [];
  let width = 0, height = 0, dpr = 1, lastPointer = 0;
  let pointerX = innerWidth / 2, pointerY = innerHeight / 2;

  function resize() {
    dpr = Math.min(devicePixelRatio || 1, 1.5);
    width = innerWidth; height = innerHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const target = width < 700 ? 44 : 92;
    while (stars.length < target) stars.push(makeStar(true));
    if (stars.length > target) stars.length = target;
  }

  function makeStar(initial = false) {
    return {
      x: Math.random() * width,
      y: initial ? Math.random() * height : height + 8,
      r: 0.45 + Math.random() * 1.5,
      speed: 0.025 + Math.random() * 0.09,
      phase: Math.random() * Math.PI * 2,
      gold: Math.random() > 0.58,
    };
  }

  function addSparks(x, y, count, force = 1) {
    for (let i = 0; i < count && sparks.length < 150; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = (0.35 + Math.random() * 1.35) * force;
      sparks.push({
        x, y,
        vx: Math.cos(angle) * speed - 0.15,
        vy: Math.sin(angle) * speed - 0.18,
        life: 1,
        decay: 0.015 + Math.random() * 0.025,
        r: 0.8 + Math.random() * 2.1,
        gold: Math.random() > 0.3,
      });
    }
  }

  addEventListener("resize", resize, { passive: true });
  addEventListener("pointermove", (event) => {
    pointerX = event.clientX; pointerY = event.clientY;
    const now = performance.now();
    if (now - lastPointer > 34 && event.pointerType !== "touch") {
      addSparks(pointerX, pointerY, 2, 0.7);
      lastPointer = now;
    }
    const hero = document.querySelector(".hero-visual");
    if (hero) {
      hero.style.setProperty("--pointer-x", `${(pointerX / width - .5) * 12}px`);
      hero.style.setProperty("--pointer-y", `${(pointerY / height - .5) * 10}px`);
    }
  }, { passive: true });
  addEventListener("pointerdown", (event) => addSparks(event.clientX, event.clientY, 24, 1.7), { passive: true });

  function glow(x, y, radius, color, alpha) {
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 4);
    gradient.addColorStop(0, color.replace("ALPHA", String(alpha)));
    gradient.addColorStop(1, color.replace("ALPHA", "0"));
    ctx.fillStyle = gradient;
    ctx.beginPath(); ctx.arc(x, y, radius * 4, 0, Math.PI * 2); ctx.fill();
  }

  function frame(now) {
    ctx.clearRect(0, 0, width, height);
    ctx.globalCompositeOperation = "screen";

    for (const star of stars) {
      star.y -= star.speed;
      star.x += Math.sin(now * 0.00018 + star.phase) * 0.025;
      if (star.y < -8) Object.assign(star, makeStar(false), { y: height + 8 });
      const twinkle = 0.22 + (Math.sin(now * 0.0016 + star.phase) + 1) * 0.22;
      const color = star.gold ? `rgba(255,218,96,${twinkle})` : `rgba(144,255,171,${twinkle})`;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2); ctx.fill();
      if (star.r > 1.25) glow(star.x, star.y, star.r, star.gold ? "rgba(255,211,79,ALPHA)" : "rgba(104,230,135,ALPHA)", twinkle * .18);
    }

    for (let i = sparks.length - 1; i >= 0; i--) {
      const spark = sparks[i];
      spark.x += spark.vx; spark.y += spark.vy;
      spark.vx *= 0.985; spark.vy = spark.vy * 0.985 + 0.008;
      spark.life -= spark.decay;
      if (spark.life <= 0) { sparks.splice(i, 1); continue; }
      const color = spark.gold ? "rgba(255,218,88,ALPHA)" : "rgba(132,244,153,ALPHA)";
      glow(spark.x, spark.y, spark.r, color, spark.life * .72);
      ctx.fillStyle = color.replace("ALPHA", String(spark.life));
      ctx.beginPath(); ctx.arc(spark.x, spark.y, spark.r * .48, 0, Math.PI * 2); ctx.fill();
    }
    requestAnimationFrame(frame);
  }

  resize();
  addSparks(width * .88, height * .2, 30, 1.1);
  requestAnimationFrame(frame);
})();
