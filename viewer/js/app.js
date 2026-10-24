// Page shell: loads a trace, renders the outline, header and the current page,
// and holds the selection (focused query token, key token, head) that blocks react to.
"use strict";

(function () {
  const { el } = AV;
  const DATA_DIR = "data";

  const app = (AV.app = {
    trace: null,
    state: { pos: 0, kpos: 0, head: 0 },
    settings: {
      scale: AV.storage.get("scale", "robust"),
      zoom: AV.storage.get("zoom", window.innerWidth < 700 ? 0.75 : 1),
      theme: AV.storage.get("theme", "auto"),
    },
    comps: [],
    pickKey: false,
    page: null,

    register(c) {
      this.comps.push(c);
      return c;
    },

    select(vars) {
      const changed = new Set();
      for (const [k, v] of Object.entries(vars)) {
        if (this.state[k] !== v) {
          this.state[k] = v;
          changed.add(k);
        }
      }
      if (!changed.size) return;
      for (const c of this.comps) c.onState && c.onState(changed);
      header.update();
    },

    tokenRef(i) {
      const t = this.trace.tokens[i];
      return t ? `“${AV.tokPlain(t.text)}” (${i})` : String(i);
    },

    template(s) {
      return s.replace(/\{(\w+)\}/g, (m, k) => (k === "pos" || k === "kpos" ? this.tokenRef(this.state[k]) : k in this.state ? String(this.state[k]) : m));
    },

    bindText(node, s) {
      node.textContent = this.template(s);
      const deps = [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);
      if (deps.length) this.register({ onState: (ch) => deps.some((d) => ch.has(d)) && (node.textContent = this.template(s)) });
    },

    go(pageId) {
      if (location.hash.slice(1) === pageId) route();
      else location.hash = pageId;
    },

    refreshAll() {
      AV.color.rebuild();
      for (const c of this.comps) c.refresh && c.refresh();
    },
  });

  // ------------------------------------------------------------------ header

  function seg(label, options, current, onPick) {
    const btns = options.map(([value, text, title]) =>
      el("button", { type: "button", "aria-pressed": String(value === current), title: title || null, onclick: () => onPick(value) }, text));
    const wrap = el("div", { class: "seg" }, label ? el("span", { class: "seg-label", text: label }) : null, el("div", { class: "seg-btns", role: "group", "aria-label": label || null }, btns));
    wrap.set = (value) => options.forEach(([v], i) => btns[i].setAttribute("aria-pressed", String(v === value)));
    return wrap;
  }

  const header = {
    build(traces) {
      const root = document.getElementById("top");
      const s = app.settings;
      const title = el("span", { class: "trace-title" });
      title.textContent = app.trace.m.title;
      title.title = app.trace.m.prompt;
      let picker = null;
      if (traces.length > 1) {
        picker = el("select", { class: "trace-select", id: "trace-select", "aria-label": "Trace", onchange: (e) => openTrace(traces, e.target.value) },
          traces.map((t) => el("option", { value: t.id, selected: t.id === app.trace.id }, t.title)));
      }
      const scale = seg("Colors", [["robust", "Clip outliers", "Color limit at the 99th percentile of |value|"], ["max", "Full range", "Color limit at the largest |value|"]], s.scale, (v) => {
        s.scale = v;
        AV.storage.set("scale", v);
        scale.set(v);
        app.refreshAll();
      });
      const zoom = seg("Squares", [[0.75, "S"], [1, "M"], [1.5, "L"]], s.zoom, (v) => {
        s.zoom = v;
        AV.storage.set("zoom", v);
        zoom.set(v);
        app.refreshAll();
      });
      const theme = seg("Theme", [["auto", "Auto"], ["light", "Light"], ["dark", "Dark"]], s.theme, (v) => {
        s.theme = v;
        AV.storage.set("theme", v);
        theme.set(v);
        applyTheme();
      });
      const navBtn = el("button", { type: "button", class: "nav-toggle", "aria-expanded": "false", "aria-controls": "outline", onclick: () => outline.toggle() }, "Outline");
      this.navBtn = navBtn;

      this.toks = app.trace.tokens.map((t, i) =>
        el("button", { type: "button", class: "tok", title: `position ${i} · token id ${t.id}`, onclick: (e) => app.select({ [e.shiftKey || app.pickKey ? "kpos" : "pos"]: i }) },
          AV.tokNode(t.text), el("span", { class: "tok-idx", text: String(i) })));
      this.pickMode = seg("Clicking a token picks", [[false, "Query"], [true, "Key"]], app.pickKey, (v) => {
        app.pickKey = v;
        this.pickMode.set(v);
      });
      const nHeads = app.trace.axisSize("head");
      this.heads = seg("Head", Array.from({ length: nHeads }, (_, h) => [h, String(h)]), app.state.head, (h) => app.select({ head: h }));
      this.heads.querySelector(".seg-btns").classList.add("head-btns");
      this.pickers = el("div", { class: "pickers" }, this.pickMode, nHeads ? this.heads : null);

      root.replaceChildren(
        el("div", { class: "bar1" },
          el("div", { class: "brand" }, el("span", { class: "brand-name" }, el("span", { class: "dot", "aria-hidden": "true" }), "Forward Pass"), title),
          picker,
          el("div", { class: "settings" }, scale, zoom, theme, navBtn)),
        el("div", { class: "bar2" }, el("div", { class: "strip-wrap" }, el("div", { class: "strip", role: "group", "aria-label": "Prompt tokens" }, this.toks)), this.pickers));
      new ResizeObserver(() => document.documentElement.style.setProperty("--header-h", root.offsetHeight + "px")).observe(root);
      this.update();
    },

    update() {
      const uses = app.page ? app.page.uses : new Set();
      this.toks.forEach((t, i) => {
        t.classList.toggle("is-q", i === app.state.pos);
        t.classList.toggle("is-k", uses.has("kpos") && i === app.state.kpos);
        t.setAttribute("aria-pressed", String(i === app.state.pos));
      });
      this.pickMode.hidden = !uses.has("kpos");
      this.heads.hidden = !uses.has("head");
      this.pickers.hidden = this.pickMode.hidden && this.heads.hidden;
      this.heads.set(app.state.head);
      const chip = this.toks[app.state.pos], wrap = chip.closest(".strip-wrap");
      const cr = chip.getBoundingClientRect(), wr = wrap.getBoundingClientRect();
      if (cr.left < wr.left || cr.right > wr.right) wrap.scrollLeft += cr.left - wr.left - (wr.width - cr.width) / 2;
    },
  };

  function applyTheme() {
    const t = app.settings.theme;
    if (t === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", t);
  }

  // ------------------------------------------------------------------ outline

  const outline = {
    build() {
      const nav = document.getElementById("outline");
      this.links = new Map();
      const secs = app.trace.m.sections.map((s) => {
        const pages = el("div", { class: "ol-pages" }, s.pages.map((p) => {
          const a = el("a", { class: "ol-page", href: "#" + p.id, title: p.title, onclick: () => this.close() }, p.short);
          this.links.set(p.id, a);
          return a;
        }));
        if (s.short) return el("div", { class: "ol-layer" }, el("span", { class: "ol-title", title: s.title, text: s.short }), pages);
        return el("div", { class: "ol-sec" }, el("span", { class: "ol-title", text: s.title }), pages);
      });
      // Consecutive short-titled sections (the layers) share one group.
      const out = [];
      for (const n of secs) {
        if (n.classList.contains("ol-layer")) {
          let g = out[out.length - 1];
          if (!g || !g.classList.contains("ol-layers")) out.push((g = el("div", { class: "ol-sec ol-layers" }, el("span", { class: "ol-title", text: "Layers" }))));
          g.append(n);
        } else out.push(n);
      }
      nav.replaceChildren(...out, el("p", { class: "ol-hint" }, "← → step through pages. Hover any square for its value; click to select that token or head. Shift-click a token to pick the key."));
    },
    mark(id) {
      for (const [pid, a] of this.links) {
        if (pid === id) {
          a.setAttribute("aria-current", "page");
          a.scrollIntoView({ block: "nearest" });
        } else a.removeAttribute("aria-current");
      }
    },
    toggle(open) {
      const nav = document.getElementById("outline");
      const now = open ?? !nav.classList.contains("open");
      nav.classList.toggle("open", now);
      header.navBtn.setAttribute("aria-expanded", String(now));
    },
    close() {
      this.toggle(false);
    },
  };

  // ------------------------------------------------------------------ pages

  let renderToken = 0;

  async function renderPage(page) {
    const token = ++renderToken;
    for (const c of app.comps) c.destroy && c.destroy();
    app.comps = [];
    app.page = page;
    AV.tip.hide();
    const main = document.getElementById("main");
    const pages = app.trace.pages, i = pages.indexOf(page);
    const pager = () =>
      el("nav", { class: "pager", "aria-label": "Previous and next page" },
        el("a", { href: "#" + (pages[i - 1] || page).id, "aria-disabled": String(i === 0), title: pages[i - 1] ? pages[i - 1].title : null }, "← " + (pages[i - 1] ? pages[i - 1].short : "")),
        el("a", { href: "#" + (pages[i + 1] || page).id, "aria-disabled": String(i === pages.length - 1), title: pages[i + 1] ? pages[i + 1].title : null }, (pages[i + 1] ? pages[i + 1].short : "") + " →"));
    const blocks = el("div", { class: "blocks" }, el("p", { class: "loading", text: "Loading activations…" }));
    main.replaceChildren(
      el("div", { class: "page-head" }, el("div", {}, el("div", { class: "crumb", text: page.section.title }), el("h1", { text: page.title })), pager()),
      blocks);
    outline.mark(page.id);
    header.update();
    document.title = `${page.title} · ${app.trace.m.model.display_name} · Forward Pass`;
    try {
      await app.trace.ensure(AV.blockTensors(page.blocks));
    } catch (e) {
      if (token === renderToken) blocks.replaceChildren(el("p", { class: "error", text: String(e.message || e) }));
      return;
    }
    if (token !== renderToken) return;
    blocks.replaceChildren(...page.blocks.map((b) => {
      const render = AV.blocks[b.type];
      return render ? render(b, app) : el("p", { class: "error", text: `This viewer doesn't know how to show a "${b.type}" block.` });
    }));
    main.append(el("div", { class: "page-foot" }, pager()));
  }

  function route() {
    const id = decodeURIComponent(location.hash.slice(1));
    const page = app.trace.pages.find((p) => p.id === id) || app.trace.pages[0];
    const changed = page !== app.page;
    renderPage(page);
    if (changed) window.scrollTo(0, 0);
  }

  function pageUses(page) {
    const s = JSON.stringify(page.blocks);
    const uses = new Set([...s.matchAll(/[$]([a-z]+)|\{([a-z]+)\}/g)].map((m) => m[1] || m[2]));
    if (page.blocks.some((b) => b.type === "multiples")) uses.add("head");
    if (page.blocks.some((b) => b.type === "cache")) uses.add("kpos");
    if (page.blocks.some((b) => b.type === "topk" || b.type === "predictions" || b.type === "tokens" || b.type === "fnplot")) uses.add("pos");
    return uses;
  }

  async function openTrace(traces, id) {
    const entry = traces.find((t) => t.id === id) || traces[0];
    AV.storage.set("trace", entry.id);
    app.trace = await AV.loadTrace(entry, DATA_DIR);
    for (const p of app.trace.pages) p.uses = pageUses(p);
    const d = app.trace.m.extra.defaults || {};
    const last = app.trace.tokens.length - 1;
    app.state = { pos: d.pos ?? last, kpos: d.kpos ?? last, head: d.head ?? 0 };
    app.page = null;
    header.build(traces);
    outline.build();
    route();
  }

  async function boot() {
    applyTheme();
    const watchTheme = () => app.trace && app.refreshAll();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", watchTheme);
    new MutationObserver(watchTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    window.addEventListener("hashchange", () => app.trace && route());
    document.addEventListener("keydown", (e) => {
      if (!app.trace || e.altKey || e.ctrlKey || e.metaKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
      const pages = app.trace.pages, i = pages.indexOf(app.page);
      if (e.key === "ArrowRight" && i < pages.length - 1) app.go(pages[i + 1].id);
      else if (e.key === "ArrowLeft" && i > 0) app.go(pages[i - 1].id);
      else if (e.key === "Escape") outline.close();
    });
    if (document.fonts) document.fonts.ready.then(() => app.trace && app.refreshAll());
    let traces;
    try {
      traces = await AV.loadScript(`${DATA_DIR}/traces.js`, "traces", "");
    } catch (e) {
      document.getElementById("main").replaceChildren(el("p", { class: "error" }, "No exported traces found. Run ", el("code", { text: "python -m actviz" }), " to generate viewer/data/."));
      return;
    }
    try {
      await openTrace(traces, AV.storage.get("trace", null));
    } catch (e) {
      document.getElementById("main").replaceChildren(el("p", { class: "error", text: String(e.message || e) }));
    }
  }

  boot();
})();
