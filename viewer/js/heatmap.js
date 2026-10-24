// A tensor (or a slice of one) drawn as a grid of colored squares.
"use strict";

(function () {
  const { el, fmt } = AV;

  // Which selection variable an axis drives when clicked (overridable per term with `select`).
  const DEFAULT_SELECT = { pos: "pos", q_pos: "pos", k_pos: "kpos", head: "head" };
  const font = (px) => `${px}px "IBM Plex Mono", ui-monospace, Menlo, monospace`;
  const measurer = document.createElement("canvas").getContext("2d");
  const textW = (s, px) => ((measurer.font = font(px)), measurer.measureText(s).width);

  function decode(i, sizes) {
    const out = new Array(sizes.length);
    for (let j = sizes.length - 1; j >= 0; j--) {
      out[j] = i % sizes[j];
      i = (i - out[j]) / sizes[j];
    }
    return out;
  }

  // Resolve a term against the current selection into a 2-D accessor over the data.
  AV.makeView = function (trace, term, state) {
    const t = trace.tensor(term.tensor);
    const { axes, shape } = t;
    const strides = new Array(axes.length);
    for (let i = axes.length - 1, s = 1; i >= 0; i--) (strides[i] = s), (s *= shape[i]);
    const ax = (a) => axes.indexOf(a);
    const fixed = { ...t.at }, uses = new Set();
    for (const [a, v] of Object.entries(term.fix || {})) {
      if (typeof v === "string" && v.startsWith("$")) {
        uses.add(v.slice(1));
        fixed[a] = state[v.slice(1)];
      } else fixed[a] = v;
    }
    const free = axes.filter((a) => !(a in fixed));
    let rows = term.rows, cols = term.cols;
    if (!rows && !cols) rows = free.length >= 2 ? [free[0]] : [];
    if (!rows) rows = free.filter((a) => !cols.includes(a));
    if (!cols) cols = free.filter((a) => !rows.includes(a));
    let base = 0;
    for (const [a, v] of Object.entries(fixed)) base += Math.max(0, Math.min(v, shape[ax(a)] - 1)) * strides[ax(a)];
    const offsets = (list) => {
      const sizes = list.map((a) => shape[ax(a)]);
      const n = sizes.reduce((p, q) => p * q, 1);
      const off = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const idx = decode(i, sizes);
        let o = 0;
        for (let j = 0; j < list.length; j++) o += idx[j] * strides[ax(list[j])];
        off[i] = o;
      }
      return { n, sizes, off };
    };
    const R = offsets(rows), C = offsets(cols), data = t.data;
    return {
      id: term.tensor,
      meta: t.meta,
      kind: t.meta.kind,
      rows, cols, fixed, uses,
      nRows: R.n, nCols: C.n,
      rowSizes: R.sizes, colSizes: C.sizes,
      // Flattened multi-axis rows/cols get a gap between groups (e.g. one group per head).
      rowGroup: R.sizes.length > 1 ? R.n / R.sizes[0] : 0,
      colGroup: C.sizes.length > 1 ? C.n / C.sizes[0] : 0,
      value: (r, c) => data[base + R.off[r] + C.off[c]],
      rowCoords: (r) => decode(r, R.sizes),
      colCoords: (c) => decode(c, C.sizes),
    };
  };

  function autoCell(nRows, nCols) {
    if (nRows === 1 && nCols === 1) return 22;
    if (nCols === 1) return 16;
    if (nCols <= 24) return nRows <= 32 ? 16 : 10;
    if (nCols <= 96) return 8;
    if (nCols <= 1024) return 5;
    if (nCols <= 4096) return 4;
    return 3;
  }

  AV.colorLimit = function (app, meta) {
    if (meta.kind === "prob") return 1;
    return app.settings.scale === "max" ? meta.stats.absmax : meta.stats.robust;
  };

  class Heatmap {
    // opt: { thumb, cell, groupScale: () => number, padLabel: px, fixed: bool, noLabels: bool }
    constructor(app, term, opt = {}) {
      this.app = app;
      this.term = term;
      this.opt = opt;
      this.el = el("div", { class: "hm" + (opt.thumb ? " thumb" : "") });
      if (!opt.thumb) {
        this.labelEl = el("span", { class: "hm-label" });
        this.dimsEl = el("span", { class: "hm-dims" });
        this.legendEl = el("span", { class: "legend" });
        this.el.append(el("div", { class: "hm-head" }, this.labelEl, this.dimsEl, this.legendEl));
      }
      this.body = el("div", { class: "hm-body" });
      this.cv = el("canvas", { "aria-hidden": "true" });
      this.ov = el("canvas", { class: "ov", role: "img" });
      this.body.append(this.cv, this.ov);
      this.el.append(this.body);
      this.ov.addEventListener("pointermove", (e) => this.onMove(e));
      this.ov.addEventListener("pointerleave", () => this.onLeave());
      this.ov.addEventListener("click", (e) => this.onClick(e));
      this.hover = null;
      this.lastW = -1;
      this.build();
      this.ro = new ResizeObserver(() => this.relayout());
      this.ro.observe(this.body);
    }

    destroy() {
      this.ro.disconnect();
    }

    build() {
      const v = (this.view = AV.makeView(this.app.trace, this.term, this.app.state));
      const sel = { ...DEFAULT_SELECT, ...(this.term.select || {}) };
      this.bind = [];
      v.rows.forEach((a, i) => sel[a] && this.bind.push({ where: "row", i, v: sel[a] }));
      v.cols.forEach((a, i) => sel[a] && this.bind.push({ where: "col", i, v: sel[a] }));
      this.boundVars = new Set(this.bind.map((b) => b.v));
      const label = this.term.label || v.meta.label;
      this.tmplVars = new Set([...label.matchAll(/\{(\w+)\}/g)].map((m) => m[1]));
      if (this.labelEl) {
        this.labelEl.textContent = this.app.template(label);
        const part = (list, sizes) => list.map((a, i) => `${this.app.trace.axisTitle(a)} ${sizes[i]}`).join(" · ");
        const dims = [v.rows.length && part(v.rows, v.rowSizes), v.cols.length && part(v.cols, v.colSizes)].filter(Boolean);
        this.dimsEl.textContent = dims.length ? dims.join(" × ") : "scalar";
        this.dimsEl.title = v.meta.desc || v.meta.label;
      }
      const g = this.geom(Infinity);
      this.natural = g.W;
      this.naturalCell = g.cs;
      if (this.labelEl) this.updateLegend();
      if (this.opt.thumb) this.el.style.cssText = `flex: 0 0 auto; width: ${g.W}px;`;
      else if (this.opt.fixed || g.W <= 380) this.el.style.cssText = `flex: 0 0 auto; min-width: min(100%, ${g.W}px);`;
      else this.el.style.cssText = `flex: 1 1 0%; width: 100%; min-width: min(100%, 260px); max-width: ${Math.ceil(g.W)}px;`;
    }

    limit() {
      if (this.opt.groupScale) return this.opt.groupScale();
      return AV.colorLimit(this.app, this.view.meta);
    }

    updateLegend() {
      const v = this.view, kind = v.kind, s = this.limit();
      const compact = this.natural < 260;
      const range = this.term.norm === "range";
      const bar = el("span", { class: "legend-bar", style: `background:${AV.color.gradientCss(range ? "positive" : kind)}${compact ? ";width:36px" : ""}` });
      if (range) {
        this.legendEl.replaceChildren(bar, "low – high, relative to this row");
        this.legendEl.title = "pale is the row's 1st percentile or lower, darkest is its maximum";
        this.dimsEl.hidden = compact;
        return;
      }
      const mag = fmt(s).replace("−", "");
      const absmax = kind === "signed" ? v.meta.stats.absmax : v.meta.stats.max;
      const clipped = this.term.norm !== "row" && kind !== "prob" && absmax > s * 1.001;
      const span = this.term.norm === "row" ? "per-row scale" : kind === "signed" ? "±" + mag : kind === "prob" ? "0 – 1" : "0 – " + mag;
      const clipNote = clipped ? `clipped; largest is ${fmt(absmax)}` : "";
      this.legendEl.replaceChildren(bar, span, clipped && !compact ? ` · ${clipNote}` : "");
      this.legendEl.title = (kind === "signed" ? "blue negative, gray zero, red positive" : "pale low, dark high") + (clipNote ? "; " + clipNote : "");
      this.dimsEl.hidden = compact;
    }

    geom(width) {
      const v = this.view, t = this.term, tr = this.app.trace, thumb = this.opt.thumb;
      let cs = t.cell || this.opt.cell || autoCell(v.nRows, v.nCols);
      cs = Math.max(2, Math.round(thumb ? cs : cs * this.app.settings.zoom));
      const fs = Math.min(11, Math.max(9, cs));
      const lastRowAxis = v.rows[v.rows.length - 1];
      let labelW = 0;
      this.rowLabels = null;
      if (!thumb && t.labels !== false && !this.opt.noLabels && v.nRows > 1 && lastRowAxis && tr.hasLabels(lastRowAxis)) {
        this.rowLabels = Array.from({ length: v.nRows }, (_, r) => tr.axisLabel(lastRowAxis, v.rowCoords(r).at(-1)));
        labelW = Math.min(104, Math.ceil(Math.max(...this.rowLabels.map((s) => textW(s, fs)))) + 8);
      }
      // Small squares only label the selected row, so the gutter can be narrower.
      if (cs < 7) labelW = Math.min(labelW, 56);
      this.ownLabelW = labelW;
      labelW = Math.max(labelW, this.opt.padLabel || 0);
      let valuesW = 0;
      if (t.values && v.nCols === 1) {
        for (let r = 0; r < v.nRows; r++) valuesW = Math.max(valuesW, textW(AV.fmtShort(v.value(r, 0)), fs));
        valuesW = Math.ceil(valuesW) + 9;
      }
      const gapFor = (c) => Math.max(2, Math.round(c * 0.6));
      const colsW = (n, c) => n * c + (v.colGroup && n > 0 ? Math.floor((n - 1) / v.colGroup) * gapFor(c) : 0);
      const avail = Math.max(24, width - labelW - valuesW);
      let wrap = t.wrap ? Math.min(t.wrap, v.nCols) : v.nCols;
      if (colsW(wrap, cs) > avail) {
        if (!t.wrap && v.nCols <= 96) while (cs > 2 && colsW(v.nCols, cs) > avail) cs--;
        if (colsW(wrap, cs) > avail) {
          let maxCols = Math.max(1, Math.floor(avail / cs));
          while (maxCols > 1 && colsW(maxCols, cs) > avail) maxCols--;
          const unit = v.colGroup && v.colGroup <= maxCols ? v.colGroup : maxCols >= 16 ? 8 : 1;
          maxCols = Math.max(unit, Math.floor(maxCols / unit) * unit);
          const nb = Math.ceil(v.nCols / maxCols);
          wrap = Math.min(maxCols, Math.ceil(v.nCols / nb / unit) * unit);
        }
      }
      const ggap = v.colGroup ? gapFor(cs) : 0, rgap = v.rowGroup ? gapFor(cs) : 0;
      const nb = Math.ceil(v.nCols / wrap);
      let colLabH = 0;
      this.colLabels = null;
      if (!thumb && nb === 1 && v.nCols > 1 && v.nCols <= 64 && cs >= 9 && v.cols.length === 1 && tr.hasLabels(v.cols[0])) {
        this.colLabels = Array.from({ length: v.nCols }, (_, c) => tr.axisLabel(v.cols[0], c));
        colLabH = Math.min(84, Math.ceil(Math.max(...this.colLabels.map((s) => textW(s, fs)))) + 6);
      }
      const rowsH = v.nRows * cs + (v.rowGroup ? Math.floor((v.nRows - 1) / v.rowGroup) * rgap : 0);
      const bandGap = v.nRows > 1 ? Math.max(6, Math.round(cs * 0.8)) : 0;
      return {
        cs, fs, labelW, valuesW, colLabH, ggap, rgap, wrap, nb, rowsH, bandGap,
        inner: cs >= 7 ? 1 : 0,
        W: labelW + colsW(wrap, cs) + valuesW,
        H: colLabH + nb * rowsH + (nb - 1) * bandGap,
      };
    }

    // Precompute pixel positions of every row and column.
    place() {
      const g = this.g, v = this.view;
      this.colX = new Float32Array(v.nCols);
      this.colTop = new Float32Array(v.nCols);
      for (let c = 0; c < v.nCols; c++) {
        const band = Math.floor(c / g.wrap), cc = c - band * g.wrap;
        this.colX[c] = g.labelW + cc * g.cs + (v.colGroup ? Math.floor(cc / v.colGroup) * g.ggap : 0);
        this.colTop[c] = g.colLabH + band * (g.rowsH + g.bandGap);
      }
      this.rowY = new Float32Array(v.nRows);
      for (let r = 0; r < v.nRows; r++) this.rowY[r] = r * g.cs + (v.rowGroup ? Math.floor(r / v.rowGroup) * g.rgap : 0);
    }

    relayout(force = false) {
      const w = this.opt.thumb ? this.natural : this.body.clientWidth;
      if (!w || (w === this.lastW && !force)) return;
      this.lastW = w;
      this.g = this.geom(w);
      this.place();
      this.draw();
      this.drawOverlay();
    }

    refresh() {
      this.build();
      this.relayout(true);
    }

    onState(changed) {
      const hits = (set) => [...set].some((x) => changed.has(x));
      if (hits(this.view.uses)) this.refresh();
      else if (hits(this.boundVars)) this.drawOverlay();
      if (this.labelEl && hits(this.tmplVars)) this.labelEl.textContent = this.app.template(this.term.label || this.view.meta.label);
    }

    rowLimits() {
      const v = this.view, out = new Float32Array(v.nRows);
      const tmp = new Float32Array(v.nCols);
      for (let r = 0; r < v.nRows; r++) {
        let n = 0;
        for (let c = 0; c < v.nCols; c++) {
          const x = Math.abs(v.value(r, c));
          if (Number.isFinite(x)) tmp[n++] = x;
        }
        const sorted = tmp.subarray(0, n).sort();
        out[r] = sorted[Math.min(n - 1, Math.floor(n * 0.99))] || 1;
      }
      return out;
    }

    // Per row: [low, span] running from the 1st percentile to the maximum.
    rowRanges() {
      const v = this.view, out = new Float32Array(2 * v.nRows), tmp = new Float32Array(v.nCols);
      for (let r = 0; r < v.nRows; r++) {
        let n = 0;
        for (let c = 0; c < v.nCols; c++) {
          const x = v.value(r, c);
          if (Number.isFinite(x)) tmp[n++] = x;
        }
        const sorted = tmp.subarray(0, n).sort();
        out[2 * r] = sorted[Math.floor(n * 0.01)];
        out[2 * r + 1] = sorted[n - 1] - out[2 * r] || 1;
      }
      return out;
    }

    draw() {
      const g = this.g, v = this.view, dpr = window.devicePixelRatio || 1;
      const W = Math.max(1, Math.ceil(g.W * dpr)), H = Math.max(1, Math.ceil(g.H * dpr));
      for (const c of [this.cv, this.ov]) {
        c.width = W;
        c.height = H;
        c.style.width = g.W + "px";
        c.style.height = g.H + "px";
      }
      const ctx = this.cv.getContext("2d");
      const range = this.term.norm === "range";
      const pal = AV.color.get(), scale = v.kind === "signed" && !range ? pal.signed.u32 : pal.seq.u32;
      const img = ctx.createImageData(W, H), buf = new Uint32Array(img.data.buffer);
      const lim = this.limit(), rowLim = this.term.norm === "row" ? this.rowLimits() : null;
      const rowLo = range ? this.rowRanges() : null;
      const masked = [], side = g.cs - g.inner;
      for (let r = 0; r < v.nRows; r++) {
        const s = rowLim ? rowLim[r] : lim, ry = this.rowY[r];
        for (let c = 0; c < v.nCols; c++) {
          const val = v.value(r, c), x = this.colX[c], y = this.colTop[c] + ry;
          let col;
          if (range && Number.isFinite(val)) col = scale[Math.round(Math.max(0, Math.min(1, (val - rowLo[2 * r]) / rowLo[2 * r + 1])) * 256)];
          else if (Number.isFinite(val)) col = scale[AV.color.index(v.kind, val, s)];
          else (col = pal.maskedU32), masked.push(x, y);
          const x0 = Math.round(x * dpr), x1 = Math.round((x + side) * dpr);
          for (let yy = Math.round(y * dpr), y1 = Math.round((y + side) * dpr); yy < y1; yy++) buf.fill(col, yy * W + x0, yy * W + x1);
        }
      }
      ctx.putImageData(img, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (masked.length && g.cs >= 6) {
        ctx.strokeStyle = pal.maskedLine;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < masked.length; i += 2) {
          ctx.moveTo(masked[i] + 1, masked[i + 1] + side - 1);
          ctx.lineTo(masked[i] + side - 1, masked[i + 1] + 1);
        }
        ctx.stroke();
      }
      ctx.font = font(g.fs);
      ctx.textBaseline = "middle";
      if (this.rowLabels && g.cs >= 7) {
        ctx.fillStyle = pal.ink2;
        ctx.textAlign = "right";
        for (let b = 0; b < g.nb; b++) {
          const top = g.colLabH + b * (g.rowsH + g.bandGap);
          for (let r = 0; r < v.nRows; r++) ctx.fillText(this.fit(this.rowLabels[r], g.labelW - 8, g.fs), g.labelW - 5, top + this.rowY[r] + side / 2);
        }
      }
      if (this.colLabels) {
        ctx.fillStyle = pal.ink2;
        ctx.textAlign = "left";
        for (let c = 0; c < v.nCols; c++) {
          ctx.save();
          ctx.translate(this.colX[c] + side / 2, g.colLabH - 4);
          ctx.rotate(-Math.PI / 2);
          ctx.fillText(this.fit(this.colLabels[c], g.colLabH - 6, g.fs), 0, 0);
          ctx.restore();
        }
      }
      if (g.valuesW) {
        ctx.fillStyle = pal.ink;
        ctx.textAlign = "left";
        for (let r = 0; r < v.nRows; r++) ctx.fillText(AV.fmtShort(v.value(r, 0)), this.colX[0] + g.cs + 5, this.colTop[0] + this.rowY[r] + side / 2);
      }
      this.ov.setAttribute("aria-label", `${this.labelEl ? this.labelEl.textContent : v.meta.label}: ${v.nRows} by ${v.nCols} grid`);
    }

    fit(s, w, px) {
      if (textW(s, px) <= w) return s;
      while (s.length > 1 && textW(s + "…", px) > w) s = s.slice(0, -1);
      return s + "…";
    }

    // Indices along a row/col list whose bound axes equal the current selection.
    selected(where) {
      const b = this.bind.filter((x) => x.where === where);
      if (!b.length) return null;
      const v = this.view, n = where === "row" ? v.nRows : v.nCols, coords = where === "row" ? v.rowCoords : v.colCoords;
      const out = [];
      for (let i = 0; i < n; i++) {
        const k = coords(i);
        if (b.every((x) => k[x.i] === this.app.state[x.v])) out.push(i);
      }
      return out;
    }

    drawOverlay() {
      if (!this.g) return;
      const g = this.g, v = this.view, dpr = window.devicePixelRatio || 1, pal = AV.color.get();
      const ctx = this.ov.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, this.ov.width, this.ov.height);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const side = g.cs - g.inner;
      const ring = (x, y, w, h, color, lw) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.strokeRect(x - lw / 2, y - lw / 2, w + lw, h + lw);
      };
      if (!this.opt.thumb) {
        const rows = this.selected("row"), cols = this.selected("col");
        const lw = g.cs >= 6 ? 2 : 1.5;
        // Runs of consecutive selected columns within each band.
        const runs = (list) => {
          const out = [];
          for (const c of list) {
            const last = out[out.length - 1];
            if (last && c === last.end + 1 && this.colTop[c] === this.colTop[last.start]) last.end = c;
            else out.push({ start: c, end: c });
          }
          return out;
        };
        const rowRect = (r, run) => {
          const x = this.colX[run.start];
          return [x, this.colTop[run.start] + this.rowY[r], this.colX[run.end] + side - x, side];
        };
        const thin = rows && cols ? 1 : lw;
        if (rows) {
          const allCols = runs(Array.from({ length: v.nCols }, (_, c) => c));
          for (const r of rows) for (const run of allCols) ring(...rowRect(r, run), pal.accent, thin);
        }
        if (cols) {
          for (const run of runs(cols)) {
            const x = this.colX[run.start];
            ring(x, this.colTop[run.start], this.colX[run.end] + side - x, g.rowsH - g.inner, pal.accent, thin);
          }
        }
        if (rows && cols) {
          for (const r of rows)
            for (const run of runs(cols)) {
              ring(...rowRect(r, run), pal.ring, lw + 2);
              ring(...rowRect(r, run), pal.accent, lw);
            }
        }
        if (rows) {
          if (this.rowLabels && g.cs < 7) {
            ctx.font = font(10);
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";
            ctx.fillStyle = pal.ink;
            for (const r of rows)
              for (let b = 0; b < g.nb; b++)
                ctx.fillText(this.fit(this.rowLabels[r], g.labelW - 8, 10), g.labelW - 5, g.colLabH + b * (g.rowsH + g.bandGap) + this.rowY[r] + side / 2);
          }
        }
      }
      if (this.hover) {
        const { r, c } = this.hover;
        ring(this.colX[c], this.colTop[c] + this.rowY[r], side, side, pal.hover, 1.5);
      }
    }

    hit(px, py) {
      const g = this.g, v = this.view;
      if (!g) return null;
      let y = py - g.colLabH;
      if (y < 0) return null;
      const bandH = g.rowsH + g.bandGap, band = Math.floor(y / bandH);
      y -= band * bandH;
      if (band >= g.nb || y >= g.rowsH) return null;
      const inGroup = (pos, group, gap) => {
        if (!group) return Math.floor(pos / g.cs);
        const pitch = group * g.cs + gap, gi = Math.floor(pos / pitch), w = pos - gi * pitch;
        return w >= group * g.cs ? -1 : gi * group + Math.floor(w / g.cs);
      };
      const r = inGroup(y, v.rowGroup, g.rgap);
      if (r < 0 || r >= v.nRows) return null;
      const x = px - g.labelW;
      if (x < 0) return { r, c: null };
      const cc = inGroup(x, v.colGroup, g.ggap);
      const c = band * g.wrap + cc;
      if (cc < 0 || cc >= g.wrap || c >= v.nCols) return null;
      return { r, c };
    }

    tipHtml(h) {
      const v = this.view, tr = this.app.trace, esc = AV.escapeHtml;
      const lines = [];
      const coord = (axis, i) => {
        const raw = tr.axisLabelRaw(axis, i);
        return `${esc(tr.axisTitle(axis))} ${i}${raw ? "  " + esc(raw) : ""}`;
      };
      const rk = v.rowCoords(h.r), ck = h.c == null ? [] : v.colCoords(h.c);
      v.rows.forEach((a, i) => lines.push(coord(a, rk[i])));
      v.cols.forEach((a, i) => ck.length && lines.push(coord(a, ck[i])));
      const fixed = Object.entries(this.view.fixed).filter(([a]) => !(a in (this.app.trace.resolve(v.id).at || {})));
      const fixedTxt = fixed.map(([a, i]) => coord(a, i)).join("\n");
      let val = "";
      if (h.c != null) {
        const x = v.value(h.r, h.c);
        val = Number.isFinite(x) ? fmt(x) + (v.kind === "prob" || v.meta.percent ? `  (${AV.fmtPct(x)})` : "") : fmt(x) + "  masked out";
      }
      const label = esc(this.labelEl ? this.labelEl.textContent : v.meta.label);
      return `<div class="tl">${label}</div>${val ? `<div class="tv">${esc(val)}</div>` : ""}<div class="tc">${lines.join("\n")}</div>${
        fixedTxt ? `<div class="tc tl">${fixedTxt}</div>` : ""
      }`;
    }

    onMove(e) {
      const rect = this.ov.getBoundingClientRect();
      const h = this.hit(e.clientX - rect.left, e.clientY - rect.top);
      const same = h && this.hover && h.c != null && h.r === this.hover.r && h.c === this.hover.c;
      if (!same) {
        this.hover = h && h.c != null ? h : null;
        this.drawOverlay();
      }
      if (h) AV.tip.show(this.tipHtml(h), e.clientX, e.clientY);
      else AV.tip.hide();
    }

    onLeave() {
      this.hover = null;
      this.drawOverlay();
      AV.tip.hide();
    }

    onClick(e) {
      const rect = this.ov.getBoundingClientRect();
      const h = this.hit(e.clientX - rect.left, e.clientY - rect.top);
      if (!h) return;
      if (e.pointerType === "touch" || !matchMedia("(hover: hover)").matches) AV.tip.flash(this.tipHtml(h), e.clientX, e.clientY);
      if (this.opt.thumb) return;
      const v = this.view, vars = {};
      const rk = v.rowCoords(h.r), ck = h.c == null ? null : v.colCoords(h.c);
      for (const b of this.bind) {
        if (b.where === "row") vars[b.v] = rk[b.i];
        else if (ck) vars[b.v] = ck[b.i];
      }
      if (Object.keys(vars).length) this.app.select(vars);
    }
  }

  AV.Heatmap = Heatmap;

  AV.tip = {
    node: null,
    timer: 0,
    show(html, x, y) {
      if (!this.node) document.body.append((this.node = el("div", { class: "tip", hidden: true })));
      clearTimeout(this.timer);
      this.node.innerHTML = html;
      this.node.hidden = false;
      const r = this.node.getBoundingClientRect();
      const left = x + 14 + r.width > window.innerWidth - 8 ? x - 14 - r.width : x + 14;
      const top = y + 16 + r.height > window.innerHeight - 8 ? y - 12 - r.height : y + 16;
      this.node.style.left = Math.max(8, left) + "px";
      this.node.style.top = Math.max(8, top) + "px";
    },
    flash(html, x, y) {
      this.show(html, x, y);
      this.timer = setTimeout(() => this.hide(), 2600);
    },
    hide() {
      if (this.node) this.node.hidden = true;
    },
  };
})();
