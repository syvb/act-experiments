// Renderers for each block type a page can contain. Each takes (block, app) and
// returns an element; anything that reacts to the selection registers with app.
"use strict";

(function () {
  const { el, fmt, fmtPct } = AV;

  function card(app, title, cls = "") {
    const c = el("section", { class: "card " + cls });
    if (title) {
      const h = el("h3", { class: "card-title" });
      app.bindText(h, title);
      c.append(h);
    }
    return c;
  }

  // A small component wrapper so non-heatmap blocks can react to the selection too.
  function reactive(app, deps, render) {
    render();
    app.register({ onState: (changed) => deps.some((d) => changed.has(d)) && render() });
  }

  const B = (AV.blocks = {});

  B.text = (block) => {
    const d = el("div", { class: "prose" });
    d.innerHTML = AV.markdown(block.text);
    return d;
  };

  B.tokens = (block, app) => {
    const c = card(app, block.title || "Tokens");
    const tbody = el("tbody");
    const table = el("table", { class: "tbl" }, el("thead", {}, el("tr", {}, el("th", { text: "position" }), el("th", { text: "token" }), el("th", { text: "token id" }))), tbody);
    const rows = app.trace.tokens.map((t, i) => {
      const tr = el("tr", { class: "clickable", onclick: () => app.select({ pos: i }) },
        el("td", { class: "num", text: String(i) }),
        el("td", {}, el("span", { class: "tk" }, AV.tokNode(t.text))),
        el("td", { class: "num", text: String(t.id) }));
      tbody.append(tr);
      return tr;
    });
    reactive(app, ["pos"], () => rows.forEach((r, i) => r.classList.toggle("is-sel", i === app.state.pos)));
    c.append(el("div", { class: "tbl-wrap" }, table));
    return c;
  };

  B.equation = (block, app) => {
    const c = card(app, block.title, "eq");
    if (block.formula) c.append(el("div", { class: "eq-formula", text: block.formula }));
    const terms = [];
    const walk = (line) => (line.row ? line.row.forEach(walk) : line.grid ? line.grid.flat().forEach((x) => x && walk(x)) : terms.push(line));
    block.lines.forEach(walk);
    const groups = {};
    for (const t of terms) if (t.tensor && t.scale) (groups[t.scale] ||= []).push(t.tensor);
    const groupScale = (name) => () => Math.max(...groups[name].map((id) => AV.colorLimit(app, app.trace.meta(id))));
    const heatmap = (t, opt = {}) =>
      app.register(new AV.Heatmap(app, t, { ...opt, groupScale: t.scale && groups[t.scale].length > 1 ? groupScale(t.scale) : null }));
    const other = (t) => {
      if (t.op != null) return el("span", { class: "op", text: t.op });
      if (t.weight) {
        const name = el("b");
        app.bindText(name, t.weight);
        return el("div", { class: "weight", title: (t.note ? t.note + ". " : "") + "Weights aren't included in the page; only activations are." },
          name, el("span", { text: t.shape.join(" × ") + (t.note ? " · " + t.note : "") }));
      }
      return el("span");
    };
    const stacked = [];
    const lines = block.lines.map((line) => {
      if (line.row) {
        // A column of numbers beside another with the same row labels doesn't repeat them.
        let prev = null;
        return el("div", { class: "eq-row" }, line.row.map((t) => {
          if (!t.tensor) return other(t);
          const hm = heatmap(t);
          const sig = hm.view.nCols === 1 && hm.rowLabels ? hm.rowLabels.join("\u0000") + "|" + hm.naturalCell : null;
          if (sig && sig === prev) {
            hm.opt.noLabels = true;
            hm.build();
          }
          prev = sig;
          return hm.el;
        }));
      }
      if (line.grid) {
        const n = Math.max(...line.grid.map((r) => r.length));
        return el("div", { class: "eq-grid", style: `grid-template-columns: repeat(${n}, auto)` },
          line.grid.flatMap((r) => Array.from({ length: n }, (_, i) => (r[i] ? (r[i].tensor ? heatmap(r[i], { fixed: true }).el : other(r[i])) : el("span")))));
      }
      if (!line.tensor) return other(line);
      const hm = heatmap(line);
      stacked.push(hm);
      return hm.el;
    });
    // Tensors stacked vertically share a label gutter, so their columns line up.
    const gutter = Math.max(0, ...stacked.map((h) => h.ownLabelW));
    for (const h of stacked) {
      if (h.ownLabelW < gutter) {
        h.opt.padLabel = gutter;
        h.build();
      }
    }
    c.append(el("div", { class: "eq-lines" }, lines));
    if (block.note) {
      const note = el("p", { class: "eq-note" });
      app.bindText(note, block.note);
      c.append(note);
    }
    return c;
  };

  // Small multiples of a tensor split along one axis (attention heads).
  B.multiples = (block, app) => {
    const c = card(app, block.title);
    const n = app.trace.axisSize(block.split);
    const labelled = block.rows.some((r) => r.label);
    const grid = el("div", { class: "mult-grid", style: `grid-template-columns: ${labelled ? "auto " : ""}repeat(${n}, auto)` });
    const cells = [];
    for (const row of block.rows) {
      if (labelled) grid.append(el("div", { class: "mult-row-label", text: row.label || "" }));
      for (let h = 0; h < n; h++) {
        const hm = app.register(new AV.Heatmap(app, { tensor: row.tensor, fix: { [block.split]: h }, label: `${row.label ? row.label + " " : ""}H${h}` }, { thumb: true, cell: block.cell || 5 }));
        const cell = el("div", {
          class: "mult-cell", role: "button", tabindex: "0",
          "aria-label": `${row.label || ""} head ${h}`,
          onclick: () => {
            app.select({ head: h });
            if (row.page) app.go(row.page);
          },
          onkeydown: (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), e.currentTarget.click()),
        }, hm.el, el("span", { class: "mult-cap", text: "H" + h }));
        cells.push([cell, h]);
        grid.append(cell);
      }
    }
    reactive(app, ["head"], () => cells.forEach(([cell, h]) => cell.classList.toggle("is-sel", h === app.state.head)));
    c.append(el("div", { class: "multiples" }, grid));
    return c;
  };

  function probBar(p) {
    const sc = AV.color.get().seq;
    return el("span", { class: "pbar", style: `width:${Math.max(1, Math.round(p * 140))}px;background:${sc.css[192]}` });
  }

  B.topk = (block, app) => {
    const c = card(app, block.title);
    const body = el("div", { class: "tbl-wrap" });
    const foot = el("p", { class: "next-line" });
    c.append(body, foot);
    reactive(app, ["pos"], () => {
      const entry = app.trace.m.extra.topk[app.state.pos];
      const tbody = el("tbody");
      entry.top.forEach((t, i) => {
        const isNext = entry.next && entry.next.id === t.id;
        tbody.append(el("tr", { class: isNext ? "is-sel" : "" },
          el("td", { class: "num", text: String(i + 1) }),
          el("td", {}, el("span", { class: "tk" }, AV.tokNode(t.text)), isNext ? el("span", { class: "tag", style: "margin-left:6px", text: "actual next" }) : null),
          el("td", { class: "num", text: fmt(t.logit) }),
          el("td", { class: "num", text: fmtPct(t.prob) }),
          el("td", {}, probBar(t.prob))));
      });
      body.replaceChildren(el("table", { class: "tbl" },
        el("thead", {}, el("tr", {}, ...["rank", "next token", "logit", "probability", ""].map((h) => el("th", { text: h })))), tbody));
      if (entry.next) {
        foot.replaceChildren("The token that actually comes next is ", el("span", { class: "tk mono" }, AV.tokNode(entry.next.text)),
          `: rank ${entry.next.rank}, probability ${fmtPct(entry.next.prob)}.`);
      } else foot.textContent = "This is the last token, so this distribution is the model's guess at how the prompt continues.";
    });
    return c;
  };

  B.predictions = (block, app) => {
    const c = card(app, block.title);
    const tbody = el("tbody");
    const rows = app.trace.m.extra.topk.map((e, i) => {
      const top = e.top[0], hit = e.next && e.next.id === top.id;
      const tr = el("tr", { class: "clickable", onclick: () => app.select({ pos: i }) },
        el("td", { class: "num", text: String(i) }),
        el("td", {}, el("span", { class: "tk" }, AV.tokNode(app.trace.tokens[i].text))),
        el("td", { class: hit ? "hit" : "" }, el("span", { class: "tk" }, AV.tokNode(top.text))),
        el("td", { class: "num", text: fmtPct(top.prob) }),
        el("td", {}, e.next ? el("span", { class: "tk" }, AV.tokNode(e.next.text)) : "—"),
        el("td", { class: "num", text: e.next ? fmtPct(e.next.prob) : "" }),
        el("td", { class: "num", text: e.next ? String(e.next.rank) : "" }));
      tbody.append(tr);
      return tr;
    });
    reactive(app, ["pos"], () => rows.forEach((r, i) => r.classList.toggle("is-sel", i === app.state.pos)));
    c.append(el("div", { class: "tbl-wrap" }, el("table", { class: "tbl" },
      el("thead", {}, el("tr", {}, ...["pos", "token", "top prediction", "p", "actual next", "p", "rank"].map((h) => el("th", { text: h })))), tbody)));
    return c;
  };

  // A 1-D function (e.g. GELU) with the actual inputs/outputs of every element plotted on it.
  B.fnplot = (block, app) => {
    const c = card(app, block.title);
    const wrap = el("div", { class: "fnplot" });
    const cv = el("canvas", { role: "img" });
    const note = el("p", { class: "eq-note" });
    wrap.append(cv);
    c.append(wrap, note);
    const tr = app.trace;
    const cx = tr.tensor(block.curve.x).data, cy = tr.tensor(block.curve.y).data;
    const px = tr.tensor(block.points.x), py = tr.tensor(block.points.y);
    const axisIdx = px.axes.indexOf(block.points.axis), n = px.shape[axisIdx], per = px.data.length / n;
    const x0 = cx[0], x1 = cx[cx.length - 1];
    let y0 = Infinity, y1 = -Infinity;
    for (const y of cy) (y0 = Math.min(y0, y)), (y1 = Math.max(y1, y));
    y0 -= 0.4;
    y1 += 0.4;
    let geo = null, hover = -1;
    // Element j of slice k along the points axis (that axis is first or last).
    const at = (arr, k, j) => (axisIdx === 0 ? arr[k * per + j] : arr[j * n + k]);
    const draw = () => {
      const W = wrap.clientWidth || 600, H = Math.round(Math.min(340, Math.max(200, W * 0.5)));
      const dpr = window.devicePixelRatio || 1, pal = AV.color.get();
      cv.width = W * dpr;
      cv.height = H * dpr;
      cv.style.height = H + "px";
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const m = { l: 36, r: 10, t: 10, b: 34 };
      const sx = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
      const sy = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
      geo = { sx, sy };
      ctx.font = '10.5px "IBM Plex Mono", ui-monospace, monospace';
      ctx.lineWidth = 1;
      for (let x = Math.ceil(x0); x <= x1; x += 2) {
        ctx.strokeStyle = x === 0 ? pal.muted : pal.rule;
        ctx.beginPath();
        ctx.moveTo(Math.round(sx(x)) + 0.5, m.t);
        ctx.lineTo(Math.round(sx(x)) + 0.5, H - m.b);
        ctx.stroke();
        ctx.fillStyle = pal.muted;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(fmt(x), sx(x), H - m.b + 5);
      }
      for (let y = Math.ceil(y0); y <= y1; y += 2) {
        ctx.strokeStyle = y === 0 ? pal.muted : pal.rule;
        ctx.beginPath();
        ctx.moveTo(m.l, Math.round(sy(y)) + 0.5);
        ctx.lineTo(W - m.r, Math.round(sy(y)) + 0.5);
        ctx.stroke();
        ctx.fillStyle = pal.muted;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(fmt(y), m.l - 6, sy(y));
      }
      ctx.fillStyle = pal.ink2;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(block.xlabel || "x", (m.l + W - m.r) / 2, H - 2);
      ctx.strokeStyle = pal.ink2;
      ctx.lineWidth = 2;
      ctx.beginPath();
      cx.forEach((x, i) => (i ? ctx.lineTo(sx(x), sy(cy[i])) : ctx.moveTo(sx(x), sy(cy[i]))));
      ctx.stroke();
      const sel = app.state.pos;
      let outside = 0;
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = pal.muted;
      for (let k = 0; k < n; k++) {
        if (k === sel) continue;
        for (let j = 0; j < per; j++) {
          const x = at(px.data, k, j);
          if (x >= x0 && x <= x1) ctx.fillRect(sx(x) - 1, sy(at(py.data, k, j)) - 1, 2, 2);
        }
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = pal.accent;
      ctx.strokeStyle = pal.ring;
      ctx.lineWidth = 0.75;
      for (let j = 0; j < per; j++) {
        const x = at(px.data, sel, j);
        if (x < x0 || x > x1) {
          outside++;
          continue;
        }
        ctx.beginPath();
        ctx.arc(sx(x), sy(at(py.data, sel, j)), 2.6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      }
      if (hover >= 0) {
        ctx.strokeStyle = pal.ink;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sx(at(px.data, sel, hover)), sy(at(py.data, sel, hover)), 5, 0, 2 * Math.PI);
        ctx.stroke();
      }
      const tok = app.template("{pos}");
      cv.setAttribute("aria-label", `${block.ylabel || "f(x)"} against ${block.xlabel || "x"}, with token ${tok}'s ${per} values`);
      note.textContent = `Yellow dots: token ${tok}'s ${per} neurons. Gray: every other token's.` +
        (outside ? ` ${outside} of this token's neurons fall outside the plotted range.` : "");
    };
    cv.addEventListener("pointermove", (e) => {
      if (!geo) return;
      const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      let best = -1, bd = 64;
      for (let j = 0; j < per; j++) {
        const x = at(px.data, app.state.pos, j);
        if (x < x0 || x > x1) continue;
        const dx = geo.sx(x) - mx, dy = geo.sy(at(py.data, app.state.pos, j)) - my, d = dx * dx + dy * dy;
        if (d < bd) (bd = d), (best = j);
      }
      if (best !== hover) (hover = best), draw();
      if (best >= 0) {
        const x = at(px.data, app.state.pos, best), y = at(py.data, app.state.pos, best);
        AV.tip.show(`<div class="tl">neuron ${best} of token ${AV.escapeHtml(app.template("{pos}"))}</div><div class="tv">${fmt(x)} → ${fmt(y)}</div>`, e.clientX, e.clientY);
      } else AV.tip.hide();
    });
    cv.addEventListener("pointerleave", () => {
      hover = -1;
      draw();
      AV.tip.hide();
    });
    const ro = new ResizeObserver(() => draw());
    ro.observe(wrap);
    app.register({ onState: (ch) => ch.has("pos") && draw(), refresh: draw, destroy: () => ro.disconnect() });
    return c;
  };

  // Every tensor id a page's blocks refer to, so the page can load its chunks up front.
  AV.blockTensors = function (blocks) {
    const ids = new Set();
    const walk = (x) => {
      if (Array.isArray(x)) x.forEach(walk);
      else if (x && typeof x === "object") {
        if (typeof x.tensor === "string") ids.add(x.tensor);
        for (const [k, v] of Object.entries(x)) {
          if ((k === "x" || k === "y") && typeof v === "string") ids.add(v);
          else walk(v);
        }
      }
    };
    walk(blocks);
    return [...ids];
  };
})();
