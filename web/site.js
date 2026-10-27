// The landing page's small motions: headline letters turn red under the cursor, and
// sections fade up once as they scroll in. Decorative only; off for reduced motion.
(() => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Wrap each letter in a span so it can take the hover colour. Screen readers get the
  // original text through aria-label, and the spans themselves are hidden from them.
  if (matchMedia("(hover: hover)").matches) {
    for (const el of document.querySelectorAll("[data-letters]")) {
      el.setAttribute("aria-label", el.textContent.replace(/\s+/g, " ").trim());
      const walk = (node) => {
        for (const child of [...node.childNodes]) {
          if (child.nodeType === Node.ELEMENT_NODE) { walk(child); continue; }
          if (child.nodeType !== Node.TEXT_NODE) continue;
          const frag = document.createDocumentFragment();
          for (const word of child.textContent.split(/(\s+)/)) {
            if (!word) continue;
            if (/^\s+$/.test(word)) { frag.append(word); continue; }
            const w = document.createElement("span");
            w.style.whiteSpace = "nowrap";
            w.setAttribute("aria-hidden", "true");
            for (const ch of word) {
              const s = document.createElement("span");
              s.className = "ch"; s.textContent = ch; w.append(s);
            }
            frag.append(w);
          }
          child.replaceWith(frag);
        }
      };
      walk(el);
    }
  }

  if (reduce) return;
  const targets = document.querySelectorAll(".hero-copy, .hero-figure, .statement, .columns > div, .section-head, .step-list li, .demo-intro, .ledger, .pipeline li, .measured");
  document.documentElement.classList.add("motion-ready");
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("revealed");
      observer.unobserve(entry.target);
    }
  }, { rootMargin: "0px 0px -8%", threshold: 0.05 });
  targets.forEach((el, i) => {
    el.classList.add("reveal");
    el.style.transitionDelay = `${(i % 3) * 80}ms`;
    observer.observe(el);
  });
})();
