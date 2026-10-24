// Shared helpers, color scales, and the data loader.
"use strict";

const AV = (window.AV = {});

// ---------------------------------------------------------------- DOM helpers

AV.el = function (tag, attrs, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c != null && c !== false) e.append(c instanceof Node ? c : String(c));
  }
  return e;
};

AV.fmt = function (v) {
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "∞";
  if (v === -Infinity) return "−∞";
  const a = Math.abs(v);
  const s = a === 0 ? "0" : a >= 1e5 || a < 1e-4 ? v.toExponential(2) : String(Number(v.toPrecision(4)));
  return s.replace("-", "−");
};

// Shorter form for numbers printed beside squares.
AV.fmtShort = function (v) {
  if (!Number.isFinite(v)) return AV.fmt(v);
  const a = Math.abs(v);
  const s = a === 0 ? "0" : a >= 1e4 || a < 1e-3 ? v.toExponential(1) : String(Number(v.toPrecision(3)));
  return s.replace(/-/g, "−");
};

AV.fmtPct = function (p) {
  if (p >= 0.1) return (p * 100).toFixed(1) + "%";
  if (p >= 0.0001) return (p * 100).toFixed(2) + "%";
  return p > 0 ? "<0.01%" : "0%";
};

// Token text for canvas labels: one line, trimmed.
AV.tokPlain = (s) => s.replace(/\n/g, "↵").replace(/\t/g, "⇥").trim() || "␣";

// Token text as DOM with spaces shown as faint dots, so " Mary" and "Mary" differ visibly.
AV.tokNode = function (s) {
  const span = AV.el("span", { class: "tok-text" });
  for (const part of s.replace(/\n/g, "↵").replace(/\t/g, "⇥").split(/( +)/)) {
    if (!part) continue;
    span.append(/^ +$/.test(part) ? AV.el("span", { class: "sp", text: "·".repeat(part.length) }) : part);
  }
  return span;
};

AV.escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Minimal inline markdown for exporter-written prose: **bold**, *italic*, `code`, blank-line paragraphs.
AV.markdown = function (md) {
  return md
    .split(/\n\s*\n/)
    .map((p) => {
      let h = AV.escapeHtml(p.replace(/\s*\n\s*/g, " "));
      h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
      h = h.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      h = h.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
      return `<p>${h}</p>`;
    })
    .join("");
};

AV.storage = {
  get(k, dflt) {
    try {
      const v = localStorage.getItem("actviz:" + k);
      return v == null ? dflt : JSON.parse(v);
    } catch (e) {
      return dflt;
    }
  },
  set(k, v) {
    try {
      localStorage.setItem("actviz:" + k, JSON.stringify(v));
    } catch (e) {
      /* storage unavailable: preferences just won't persist */
    }
  },
};

// ---------------------------------------------------------------- color scales
// Scales are interpolated in OKLab between the anchor colors defined as CSS tokens,
// so light and dark themes each get their own validated stops.

AV.color = (function () {
  const toLin = (c) => ((c /= 255) <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const toSrgb = (c) => Math.round(Math.min(1, Math.max(0, c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055)) * 255);

  function parse(css) {
    css = css.trim();
    if (css.startsWith("#")) {
      const h = css.length === 4 ? css.slice(1).replace(/./g, "$&$&") : css.slice(1, 7);
      return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
    }
    const m = css.match(/[\d.]+/g);
    return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
  }
  function toLab([r, g, b]) {
    const R = toLin(r), G = toLin(g), B = toLin(b);
    const l = Math.cbrt(0.4122214708 * R + 0.5363619982 * G + 0.0514459279 * B);
    const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
    const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
    return [
      0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
      1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
      0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
    ];
  }
  function fromLab([L, a, b]) {
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
    return [
      toSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
      toSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
      toSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
    ];
  }
  // Evenly spaced stops → n colors.
  function ramp(stops, n) {
    const labs = stops.map((c) => toLab(parse(c)));
    const out = [];
    for (let i = 0; i < n; i++) {
      const t = (i / (n - 1)) * (labs.length - 1);
      const j = Math.min(labs.length - 2, Math.floor(t)), f = t - j;
      out.push(fromLab(labs[j].map((v, k) => v + (labs[j + 1][k] - v) * f)));
    }
    return out;
  }
  const pack = ([r, g, b]) => (255 << 24) | (b << 16) | (g << 8) | r; // little-endian RGBA in a Uint32

  let scales = null;
  function build() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n);
    const mk = (rgbs) => ({ rgb: rgbs, u32: Uint32Array.from(rgbs, pack), css: rgbs.map(([r, g, b]) => `rgb(${r},${g},${b})`) });
    scales = {
      signed: mk(ramp([v("--div-neg"), v("--div-neg-2"), v("--div-mid"), v("--div-pos-2"), v("--div-pos")], 257)),
      seq: mk(ramp([v("--seq-0"), v("--seq-1"), v("--seq-2")], 257)),
      masked: parse(v("--masked")),
      maskedLine: v("--masked-line").trim(),
      ink: v("--fg").trim(),
      ink2: v("--fg-2").trim(),
      muted: v("--muted").trim(),
      rule: v("--rule").trim(),
      accent: v("--accent").trim(),
      ring: v("--sel-ring").trim(),
      hover: v("--hover-ring").trim(),
      surface: v("--surface").trim(),
    };
    scales.maskedU32 = pack(scales.masked);
    return scales;
  }
  return {
    get: () => scales || build(),
    rebuild: build,
    // Index into a 257-entry scale for value v with color limit s.
    index(kind, v, s) {
      if (kind === "signed") return Math.round((Math.max(-1, Math.min(1, v / s)) + 1) * 128);
      return Math.round(Math.max(0, Math.min(1, v / s)) * 256);
    },
    gradientCss(kind) {
      const sc = kind === "signed" ? this.get().signed : this.get().seq;
      const stops = [];
      for (let i = 0; i <= 8; i++) stops.push(sc.css[Math.round((i / 8) * 256)]);
      return `linear-gradient(90deg, ${stops.join(",")})`;
    },
  };
})();

// ---------------------------------------------------------------- data loading
// Data files are scripts that call ACTVIZ.receive(kind, key, payload), so the viewer
// works from file:// as well as over HTTP.

(function () {
  const received = new Map();
  const waiting = new Map();

  window.ACTVIZ = {
    receive(kind, key, payload) {
      const k = kind + ":" + key;
      received.set(k, payload);
      const w = waiting.get(k);
      if (w) w.resolve(payload);
    },
  };

  AV.loadScript = function (src, kind, key) {
    const k = kind + ":" + key;
    if (received.has(k)) return Promise.resolve(received.get(k));
    if (waiting.has(k)) return waiting.get(k).promise;
    const w = {};
    w.promise = new Promise((resolve, reject) => Object.assign(w, { resolve, reject }));
    waiting.set(k, w);
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = () => {
      waiting.delete(k);
      s.remove();
      if (!received.has(k)) w.reject(new Error(`${src} loaded but didn't contain ${kind} "${key}"`));
    };
    s.onerror = () => {
      waiting.delete(k);
      s.remove();
      w.reject(new Error(`Couldn't load ${src}. Regenerate the data with "python -m actviz".`));
    };
    document.head.append(s);
    return w.promise;
  };

  // float16 → float32 lookup table
  const F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    F16[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }

  function b64bytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  class Trace {
    constructor(dir, manifest) {
      this.dir = dir;
      this.m = manifest;
      this.id = manifest.id;
      this.tokens = manifest.tokens;
      this.chunkBytes = new Map();
      this.arrays = new Map();
      this.labelLists = new Map();
      this.pages = [];
      for (const s of manifest.sections) for (const p of s.pages) this.pages.push({ ...p, section: s });
    }

    meta(id) {
      const m = this.m.tensors[id];
      if (!m) throw new Error(`Unknown tensor "${id}"`);
      return m;
    }

    // The stored tensor behind `id` (following an alias), plus the indices the alias pins.
    resolve(id) {
      const meta = this.meta(id);
      if (!meta.of) return { meta, store: meta, storeId: id, at: {} };
      const inner = this.resolve(meta.of);
      return { meta, store: inner.store, storeId: inner.storeId, at: { ...inner.at, ...meta.at } };
    }

    chunksFor(ids) {
      const need = new Set();
      for (const id of ids) need.add(this.resolve(id).store.chunk);
      return [...need].filter((c) => !this.chunkBytes.has(c));
    }

    labelsFor(ids) {
      const need = new Set();
      for (const id of ids) {
        for (const a of this.resolve(id).store.axes) {
          const lab = (this.m.axes[a] || {}).labels;
          if (typeof lab === "string" && lab !== "tokens" && !this.labelLists.has(lab)) need.add(lab);
        }
      }
      return [...need];
    }

    async ensure(ids, labelsToo = true) {
      const chunks = this.chunksFor(ids);
      const labels = labelsToo ? this.labelsFor(ids) : [];
      await Promise.all([
        ...chunks.map(async (c) => {
          const b64 = await AV.loadScript(`${this.dir}/${this.m.chunks[c].file}`, "chunk", `${this.id}/${c}`);
          this.chunkBytes.set(c, b64bytes(b64).buffer);
        }),
        ...labels.map(async (name) => {
          const list = await AV.loadScript(`${this.dir}/${this.m.labels[name].file}`, "labels", `${this.id}/${name}`);
          this.labelLists.set(name, list);
        }),
      ]);
    }

    // Float32Array of a stored (non-alias) tensor.
    array(storeId) {
      let a = this.arrays.get(storeId);
      if (a) return a;
      const m = this.meta(storeId);
      const buf = this.chunkBytes.get(m.chunk);
      if (!buf) throw new Error(`Chunk "${m.chunk}" isn't loaded`);
      const n = m.shape.reduce((p, q) => p * q, 1);
      if (m.dtype === "f32") a = new Float32Array(buf.slice(m.offset, m.offset + 4 * n));
      else {
        const h = new Uint16Array(buf, m.offset, n);
        a = new Float32Array(n);
        for (let i = 0; i < n; i++) a[i] = F16[h[i]];
      }
      this.arrays.set(storeId, a);
      return a;
    }

    tensor(id) {
      const r = this.resolve(id);
      return { meta: r.meta, shape: r.store.shape, axes: r.store.axes, at: r.at, data: this.array(r.storeId) };
    }

    axisSize(axis) {
      for (const m of Object.values(this.m.tensors)) {
        const i = m.axes.indexOf(axis);
        if (i >= 0 && !m.of) return m.shape[i];
      }
      return 0;
    }

    axisTitle(axis) {
      return (this.m.axes[axis] || {}).title || axis;
    }

    hasLabels(axis) {
      const a = this.m.axes[axis] || {};
      return !!(a.labels || a.prefix);
    }

    axisLabel(axis, i) {
      const a = this.m.axes[axis] || {};
      if (a.labels === "tokens") return this.tokens[i] ? AV.tokPlain(this.tokens[i].text) : String(i);
      if (Array.isArray(a.labels)) return a.labels[i] ?? String(i);
      if (typeof a.labels === "string") {
        const list = this.labelLists.get(a.labels);
        return list ? list[i] : String(i);
      }
      if (a.prefix) return a.prefix + i;
      return String(i);
    }

    // Raw label for tooltips (keeps spaces), e.g. token text in quotes.
    axisLabelRaw(axis, i) {
      const a = this.m.axes[axis] || {};
      if (a.labels === "tokens") return this.tokens[i] ? JSON.stringify(this.tokens[i].text) : null;
      if (typeof a.labels === "string" && !Array.isArray(a.labels)) {
        const list = this.labelLists.get(a.labels);
        return list ? JSON.stringify(list[i]) : null;
      }
      if (Array.isArray(a.labels)) return a.labels[i];
      return null;
    }
  }

  AV.loadTrace = async function (entry, dataDir) {
    const dir = `${dataDir}/${entry.id}`;
    const manifest = await AV.loadScript(`${dir}/manifest.js`, "manifest", entry.id);
    return new Trace(dir, manifest);
  };
})();
