"use strict";
/* PXA Control: the Live tab. Rolling charts of what the servers and the cards are doing right now.
   Loaded by index.html after its own script, so it uses that script's helpers ($, $$, el, api, toast, store, segInit, fmtNum,
   showTab, S). It reads GET /api/live, which PXA Control fills from nvidia-smi and from each server's own
   /slots, /props and /pxa/stats, and for the ranges past one hour (v3.1) GET /api/telemetry/history: the same numbers from
   the on-disk history PXA Control records in the background (tools/pxa_telemetry.py), one point per bucket. Plain SVG, no library. Everything that came from a server is put on the page as text.
   Colour follows the entity: a server keeps its colour (the same one on the Servers tab), a card keeps the colour of its
   index. Axes always start where the quantity starts (0, or a fixed range for percent and temperature), so a wobble
   never looks like a spike. One y axis per chart. */

const LV = {
  range: store.get("lv.range", 900), cards: new Map(), servers: new Map(), ghosts: [], seen: new Set(),
  ci: null, ts: 0, lastT: 0, histFrom: 0, timer: null, busy: false, built: false, err: "", fails: 0, open: false, host: null,
  hist: null, histAt: 0, histRange: 0, tel: null, telAt: 0,
};
let R = LV;                                   // what the charts draw: LV (live, in memory) or LV.hist (the on-disk history)
const LV_KEEP = 3700;                        // seconds of history held in the page
let LV_CLIP = 0;                              // chart clip ids
const LV_HOT_C = 80;                         // the card temperature the rig page also warns at
const lvColor = i => `var(--series-${((i | 0) % 8) + 1})`;
const LV_NS = "http://www.w3.org/2000/svg";
function lvSvg(tag, attrs, ...kids) {
  const e = document.createElementNS(LV_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null && v !== false) e.setAttribute(k, v === true ? "" : v);
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
}
const lvNum = v => v != null && isFinite(v);
function lvFmt(v, d) { return !lvNum(v) ? "-" : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en-US") : v.toFixed(d == null ? (Math.abs(v) >= 100 ? 0 : 1) : d); }
function lvClock(t, sec) {
  const d = new Date(t * 1000), p = n => String(n).padStart(2, "0");
  return p(d.getHours()) + ":" + p(d.getMinutes()) + (sec ? ":" + p(d.getSeconds()) : "");
}
const LV_MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function lvDay(t) { const d = new Date(t * 1000); return LV_MON[d.getMonth()] + " " + d.getDate(); }
function lvWhen(t, sec) { return R.hist ? lvDay(t) + " " + lvClock(t, false) : lvClock(t, sec); }      // a tooltip's time
function lvDur(s) { return s >= 172800 ? Math.round(s / 86400) + " days" : s >= 5400 ? Math.round(s / 3600) + " h" : fmtDur(s); }
function lvShort(s, n) { s = String(s == null ? "" : s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
function lvTok(n) { return n == null ? "-" : n >= 10000 ? Math.round(n / 1000) + "k" : n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(Math.round(n)); }
function lvNice(max) { return 4 * niceMax(Math.max(1e-9, max) * 1.08 / 4); }          // four even, round steps
function lvTimeTicks(t0, t1, maxN) {
  const span = t1 - t0, steps = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800, 604800];
  const step = steps.find(s => span / s <= maxN) || 604800, off = -new Date().getTimezoneOffset() * 60, out = [];
  for (let t = Math.ceil((t0 + off) / step) * step - off; t <= t1; t += step) out.push(t);
  return {ticks: out, sec: step < 60, day: step >= 86400, mixed: step < 86400 && span > 86400};
}
function lvTick(t, tk) {                      // an axis label: 14:05, or Oct 6, or Oct 6 18:00 on a multi-day range
  if (tk.day) return lvDay(t);
  if (tk.mixed) { const d = new Date(t * 1000); return d.getHours() === 0 && d.getMinutes() === 0 ? lvDay(t) : lvClock(t, false); }
  return lvClock(t, tk.sec);
}
function lvSearch(pts, t) {                    // index of the point nearest to t (pts sorted by time)
  let lo = 0, hi = pts.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (pts[m][0] < t) lo = m; else hi = m; }
  return Math.abs(pts[lo][0] - t) <= Math.abs(pts[hi][0] - t) ? lo : hi;
}

/* ------------------------------------------------------------------ line chart
   spec: {series:[{id,name,color,pts:[[t,v|null]...]}], t0, t1, y:{min,max,unit,digits}, mini (no axes, for a tile or a strip), height,
          area (a 10 % wash under a single line), gap (seconds without a point that break the line), dots (one mark per point),
          marks:[{v,label}] (a reference level), direct:false (no end labels), label (for screen readers), empty, tipNote} */
function lvChart(host, spec) {
  const W = Math.floor(host.clientWidth);
  if (W < 60) return;
  const mini = !!spec.mini, H = spec.height || (mini ? 52 : 220), t0 = spec.t0, t1 = spec.t1;
  const series = spec.series.map(s => Object.assign({}, s, {pts: s.pts.filter(p => p[0] >= t0 - 60 && p[0] <= t1 + 5)})).filter(s => s.pts.some(p => p[1] != null));
  const L = mini ? 4 : 46, T = mini ? 7 : 12, B = mini ? 7 : 24, y = spec.y || {}, ymin = y.min == null ? 0 : y.min;
  let ymax = y.max;
  if (ymax == null) { let m = 0; for (const s of series) for (const p of s.pts) if (p[1] != null && p[0] >= t0 && p[1] > m) m = p[1]; ymax = lvNice(m || 1); }
  const ph = H - T - B, Y = v => T + ph * (1 - (Math.max(ymin, Math.min(ymax, v)) - ymin) / (ymax - ymin));
  // direct labels sit at the line ends; they are used only when they would not collide (the legend and the tooltip carry the rest)
  const lastVals = series.map(s => { for (let i = s.pts.length - 1; i >= 0; i--) if (s.pts[i][1] != null) return s.pts[i][1]; return null; }).filter(v => v != null).map(Y).sort((a, b) => a - b);
  const labelled = !mini && W >= 520 && series.length >= 1 && series.length <= 4 && spec.direct !== false && lastVals.every((v, i) => !i || v - lastVals[i - 1] >= 16);
  const R = mini ? 8 : labelled ? 120 : 14, pw = W - L - R;
  const X = t => L + pw * (t - t0) / (t1 - t0);
  const svg = lvSvg("svg", {viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "lv-svg", role: "img", tabindex: mini ? null : "0",
                         "aria-label": spec.label || ""});
  // grid and axes: hairlines, solid
  if (!mini) {
    for (let i = 0; i <= 4; i++) {
      const v = ymin + (ymax - ymin) * i / 4, yy = Math.round(Y(v)) + .5;
      svg.append(lvSvg("line", {x1: L, x2: W - R, y1: yy, y2: yy, class: i === 0 ? "lv-axis" : "lv-grid"}));
      svg.append(lvSvg("text", {x: L - 8, y: yy + 4, class: "lv-tick", "text-anchor": "end"}, lvFmt(v, ymax - ymin >= 10 ? 0 : 1)));
    }
    const tk = lvTimeTicks(t0, t1, W < 420 ? 3 : 6);
    tk.ticks.forEach((t, i) => {
      const xx = Math.round(X(t)) + .5;
      svg.append(lvSvg("line", {x1: xx, x2: xx, y1: T, y2: T + ph, class: "lv-grid"}));
      svg.append(lvSvg("text", {x: xx, y: H - 7, class: "lv-tick", "text-anchor": i === 0 && X(t) - L < 20 ? "start" : "middle"}, lvTick(t, tk)));
    });
  } else {
    svg.append(lvSvg("line", {x1: L, x2: W - R, y1: Math.round(Y(ymin)) + .5, y2: Math.round(Y(ymin)) + .5, class: "lv-axis"}));
    if (y.max != null) svg.append(lvSvg("line", {x1: L, x2: W - R, y1: Math.round(Y(ymax)) + .5, y2: Math.round(Y(ymax)) + .5, class: "lv-grid"}));
  }
  for (const m of spec.marks || []) if (m.v > ymin && m.v < ymax) {      // a reference level (solid, status colour, named)
    const yy = Math.round(Y(m.v)) + .5;
    svg.append(lvSvg("line", {x1: L, x2: W - R, y1: yy, y2: yy, class: "lv-mark"}));
    if (!mini && m.label) svg.append(lvSvg("text", {x: L + 4, y: yy - 4, class: "lv-marklabel"}, m.label));
  }
  // the lines, clipped to the plot (a point from just before the window must not spill over the axis)
  const g = lvSvg("g", {"clip-path": `url(#lvc${++LV_CLIP})`});
  svg.append(lvSvg("defs", {}, lvSvg("clipPath", {id: "lvc" + LV_CLIP}, lvSvg("rect", {x: L, y: 0, width: pw + 7, height: H}))), g);
  const ends = [];
  for (const s of series) {
    const segs = [];
    let cur = [], px = null, acc = null;
    const gap = spec.gap == null ? 25 : spec.gap, flush = () => { if (acc) cur.push([px, acc[0] / acc[1], acc[2] / acc[1]]); acc = null; px = null; };
    let prev = null;
    for (const p of s.pts) {
      if (p[1] == null || p[0] < t0 - 30) { flush(); if (cur.length) segs.push(cur); cur = []; prev = null; continue; }
      if (prev != null && p[0] - prev > gap) { flush(); if (cur.length) segs.push(cur); cur = []; }
      prev = p[0];
      const xr = Math.round(X(p[0]));
      if (px !== xr) { flush(); px = xr; acc = [0, 0, 0]; }
      acc[0] += p[1]; acc[1] += 1; acc[2] += p[0];
    }
    flush(); if (cur.length) segs.push(cur);
    let d = "", last = null, areaD = "";
    for (const seg of segs) {
      const pts = seg.map(q => [X(q[2]), Y(q[1])]);
      d += pts.map((q, i) => (i ? "L" : "M") + q[0].toFixed(1) + " " + q[1].toFixed(1)).join("");
      if (pts.length === 1) d += "l0.01 0";
      if (spec.area) areaD += pts.map((q, i) => (i ? "L" : "M") + q[0].toFixed(1) + " " + q[1].toFixed(1)).join("") + `L${pts[pts.length - 1][0].toFixed(1)} ${Y(ymin).toFixed(1)}L${pts[0][0].toFixed(1)} ${Y(ymin).toFixed(1)}Z`;
      last = [pts[pts.length - 1][0], pts[pts.length - 1][1], seg[seg.length - 1][1]];
    }
    if (spec.area && areaD) g.append(lvSvg("path", {d: areaD, class: "lv-area", style: `fill:${s.color}`}));
    g.append(lvSvg("path", {d, class: "lv-line", style: `stroke:${s.color}`}));
    if (spec.dots) for (const seg of segs) for (const q of seg) g.append(lvSvg("circle", {cx: X(q[2]).toFixed(1), cy: Y(q[1]).toFixed(1), r: 4, class: "lv-dot", style: `fill:${s.color}`}));
    if (last) ends.push({s, x: last[0], y: last[1], v: last[2]});
  }
  for (const e of ends) {
    if (!spec.dots) g.append(lvSvg("circle", {cx: e.x.toFixed(1), cy: e.y.toFixed(1), r: 4, class: "lv-end", style: `fill:${e.s.color}`}));
    if (labelled) svg.append(lvSvg("text", {x: Math.min(W - R + 8, e.x + 10), y: e.y + 4, class: "lv-dl"},
      lvSvg("tspan", {class: "lv-dv"}, lvFmt(e.v, y.digits) + (y.unit ? " " + y.unit : "")),
      series.length > 1 ? lvSvg("tspan", {class: "lv-dn", dx: 6}, lvShort(e.s.name, 11)) : null));
  }
  if (!series.length && !mini) svg.append(lvSvg("text", {x: L + pw / 2, y: T + ph / 2, class: "lv-empty", "text-anchor": "middle"}, spec.empty || "no data yet"));
  // hover layer: a crosshair, a dot per series, one tooltip listing every series at that time
  const cross = lvSvg("line", {class: "lv-cross", y1: T, y2: T + ph, visibility: "hidden"});
  const hov = series.map(s => lvSvg("circle", {r: 4, class: "lv-hov", style: `fill:${s.color}`, visibility: "hidden"}));
  const hit = lvSvg("rect", {x: L, y: 0, width: Math.max(1, pw), height: H, fill: "transparent", class: "lv-hit"});
  svg.append(cross, ...hov, hit);
  let tip = host._tip;
  if (!tip) { tip = host._tip = el("div", {class: "lv-tip", role: "status", hidden: true}); }
  const show = (clientX, key) => {
    if (!series.length) return;
    const r = svg.getBoundingClientRect(), px = key != null ? key : clientX - r.left;
    const t = t0 + (t1 - t0) * Math.min(1, Math.max(0, (px - L) / pw));
    const near = series.map(s => { const i = lvSearch(s.pts.filter(p => p[1] != null), t), q = s.pts.filter(p => p[1] != null)[i];
      return q && Math.abs(q[0] - t) <= Math.max((t1 - t0) / 40, 12) ? q : null; });
    const tt = near.find(Boolean); if (!tt) { hide(); return; }
    const xx = Math.round(X(tt[0])) + .5;
    cross.setAttribute("x1", xx); cross.setAttribute("x2", xx); cross.setAttribute("visibility", "visible");
    hov.forEach((c, i) => { const q = near[i]; if (q) { c.setAttribute("cx", X(q[0])); c.setAttribute("cy", Y(q[1])); c.setAttribute("visibility", "visible"); } else c.setAttribute("visibility", "hidden"); });
    tip.replaceChildren(el("div", {class: "tt-h", text: lvWhen(tt[0], true) + (spec.tipNote ? " · " + spec.tipNote : "")}),
      ...series.map((s, i) => near[i] ? el("div", {class: "tt-r"}, el("i", {class: "tt-k", style: `background:${s.color}`}), el("b", {text: lvFmt(near[i][1], y.digits) + (y.unit ? " " + y.unit : "")}), el("span", {text: s.name})) : null).filter(Boolean));
    tip.hidden = false;
    const w = tip.offsetWidth, hh = tip.offsetHeight;
    let left = xx + 14; if (left + w > W - 4) left = xx - w - 14;
    tip.style.left = Math.max(4, left) + "px"; tip.style.top = Math.max(4, Math.min(H - hh - 4, T + 4)) + "px";
  };
  const hide = () => { cross.setAttribute("visibility", "hidden"); hov.forEach(c => c.setAttribute("visibility", "hidden")); tip.hidden = true; };
  hit.addEventListener("pointermove", e => { host.dataset.hold = "1"; show(e.clientX); });
  hit.addEventListener("pointerleave", () => { delete host.dataset.hold; hide(); });
  svg.addEventListener("keydown", e => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault(); host.dataset.hold = "1";
    host._kx = Math.min(W - R, Math.max(L, (host._kx == null ? W - R : host._kx) + (e.key === "ArrowLeft" ? -1 : 1) * pw / 40)); show(0, host._kx);
  });
  svg.addEventListener("blur", () => { delete host.dataset.hold; host._kx = null; hide(); });
  const kids = [svg, tip];
  if (!mini && spec.series.length >= 2) {
    kids.push(el("div", {class: "lv-legend"}, ...spec.series.map(s => el("span", {}, el("i", {style: `background:${s.color}`}), s.name))));
  }
  host.replaceChildren(...kids);
  if (host._kx != null && host.dataset.hold) show(0, host._kx);
}

/* ------------------------------------------------------------------ request timeline
   lanes: [{id, label, color, bars:[{a, b, c, live, tip:[...]}]}] a = start, b = decode start, c = end (unix s) */
function lvTimeline(host, spec) {
  const W = Math.floor(host.clientWidth);
  if (W < 60) return;
  const lanes = spec.lanes, G = Math.min(210, Math.round(W * .3)), R = 12, T = 8, B = 24, LH = 26, t0 = spec.t0, t1 = spec.t1;
  const H = T + B + Math.max(1, lanes.length) * LH, pw = W - G - R, X = t => G + pw * (t - t0) / (t1 - t0);
  const svg = lvSvg("svg", {viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "lv-svg", role: "img", "aria-label": spec.label || ""});
  const tk = lvTimeTicks(t0, t1, W < 420 ? 3 : 6);
  svg.append(lvSvg("line", {x1: G, x2: W - R, y1: H - B + .5, y2: H - B + .5, class: "lv-axis"}));
  tk.ticks.forEach(t => {
    const xx = Math.round(X(t)) + .5;
    svg.append(lvSvg("line", {x1: xx, x2: xx, y1: T, y2: H - B, class: "lv-grid"}));
    svg.append(lvSvg("text", {x: xx, y: H - 7, class: "lv-tick", "text-anchor": "middle"}, lvTick(t, tk)));
  });
  let tip = host._tip; if (!tip) tip = host._tip = el("div", {class: "lv-tip", role: "status", hidden: true});
  const showTip = (b, e) => {
    tip.replaceChildren(...b.tip.map((l, i) => el("div", {class: i ? "tt-r" : "tt-h", text: l})));
    tip.hidden = false;
    const r = host.getBoundingClientRect(), w = tip.offsetWidth;
    let left = e.clientX - r.left + 14; if (left + w > W - 4) left = e.clientX - r.left - w - 14;
    tip.style.left = Math.max(4, left) + "px"; tip.style.top = Math.max(4, Math.min(H - tip.offsetHeight - 4, e.clientY - r.top - 10)) + "px";
  };
  lanes.forEach((ln, i) => {
    const y0 = T + i * LH, by = y0 + (LH - 14) / 2;
    if (i % 2 === 0) svg.append(lvSvg("rect", {x: 0, y: y0, width: W, height: LH, class: "lv-band"}));
    svg.append(lvSvg("rect", {x: 6, y: y0 + LH / 2 - 5, width: 10, height: 10, rx: 2, style: `fill:${ln.color}`}));
    svg.append(lvSvg("text", {x: 22, y: y0 + LH / 2 + 4, class: "lv-lane"}, lvShort(ln.label, Math.max(4, Math.max(8, Math.floor((G - 26) / 6.4)) - (ln.suffix || "").length)) + (ln.suffix || "")));
    for (const b of ln.bars) {
      const xa = Math.max(G, X(b.a)), xb = Math.min(W - R, X(b.c)), xd = Math.min(xb, Math.max(xa, X(b.b)));
      if (xb <= G || xa >= W - R) continue;
      const wTot = Math.max(3, xb - xa);
      if (xd - xa >= 1) svg.append(lvSvg("rect", {x: xa.toFixed(1), y: by, width: Math.max(2, xd - xa - 1).toFixed(1), height: 14, rx: 3, class: "lv-pre", style: `fill:${ln.color}`}));
      if (xb - xd >= 1 || xd - xa < 1) svg.append(lvSvg("rect", {x: xd.toFixed(1), y: by, width: Math.max(2, wTot - Math.max(0, xd - xa)).toFixed(1), height: 14, rx: 3, class: b.live ? "lv-dec lv-run" : "lv-dec", style: `fill:${ln.color}`}));
      const hit = lvSvg("rect", {x: xa - 2, y: y0, width: wTot + 4, height: LH, fill: "transparent", class: "lv-hit", tabindex: "0", "aria-label": b.tip.join(", ")});
      hit.addEventListener("pointermove", e => { host.dataset.hold = "1"; showTip(b, e); });
      hit.addEventListener("pointerleave", () => { delete host.dataset.hold; tip.hidden = true; });
      hit.addEventListener("focus", () => { const r = hit.getBoundingClientRect(); showTip(b, {clientX: r.left + r.width / 2, clientY: r.top}); });
      hit.addEventListener("blur", () => { tip.hidden = true; });
      svg.append(hit);
    }
  });
  if (!lanes.length) svg.append(lvSvg("text", {x: G + pw / 2, y: T + LH / 2 + 4, class: "lv-empty", "text-anchor": "middle"}, spec.empty || "no requests in this range yet"));
  host.replaceChildren(svg, tip, el("div", {class: "lv-legend"},
    el("span", {}, el("i", {class: "k-pre"}), "prefill: reading the prompt"), el("span", {}, el("i", {class: "k-dec"}), "decode: writing the answer")));
}

/* ------------------------------------------------------------------ data
   The page keeps every row the server sent, indexed by column name. */
function lvMerge(d) {
  const idx = c => Object.fromEntries(c.map((n, i) => [n, i]));
  LV.ci = {card: idx(d.cols.card), server: idx(d.cols.server), req: idx(d.cols.req), host: idx(d.cols.host)};
  LV.ts = d.ts; LV.histFrom = d.history_from; LV.every = d.every; LV.background = !!d.background;
  const cut = d.ts - LV_KEEP;
  const take = (e, rows) => { const last = e.rows.length ? e.rows[e.rows.length - 1][0] : 0; for (const r of rows) if (r[0] > last) e.rows.push(r); while (e.rows.length && e.rows[0][0] < cut) e.rows.shift(); if (e.rows.length) LV.lastT = Math.max(LV.lastT, e.rows[e.rows.length - 1][0]); };
  const have = new Set();
  if (d.host) { if (!LV.host) LV.host = {rows: []}; LV.host.meta = d.host; take(LV.host, d.host.rows); }
  for (const c of d.cards) { let e = LV.cards.get(c.index); if (!e) LV.cards.set(c.index, e = {rows: []}); e.meta = c; take(e, c.rows); }
  for (const s of d.servers) {
    have.add(s.key);
    let e = LV.servers.get(s.key);
    if (!e) LV.servers.set(s.key, e = {rows: [], reqs: [], prevActive: []});
    const wasActive = e.meta ? (e.meta.active || []) : [];
    e.meta = s; take(e, s.rows);
    for (const r of s.reqs) { const k = s.key + "|" + r[0] + "|" + r[1]; if (!LV.seen.has(k)) { LV.seen.add(k); e.reqs.push(r); } }
    e.reqs.sort((a, b) => a[0] - b[0]); while (e.reqs.length && e.reqs[0][0] < cut) e.reqs.shift();
    // a request that just left the live list is kept as a bar until the server's own record of it arrives
    const nowIds = new Set((s.active || []).map(a => a.id + "|" + a.t0));
    for (const a of wasActive) if (!nowIds.has(a.id + "|" + a.t0)) LV.ghosts.push({key: s.key, slot: a.id, a: a.t0, b: a.t1, c: d.ts - (d.every || 2) / 2, until: d.ts + 20});
  }
  for (const k of [...LV.servers.keys()]) if (!have.has(k)) LV.servers.delete(k);
  LV.ghosts = LV.ghosts.filter(g => g.until > d.ts && !(LV.servers.get(g.key) || {reqs: []}).reqs.some(r => r[LV.ci.req.slot] === g.slot && Math.abs(r[0] - g.c) < 8));
  if (LV.seen.size > 20000) LV.seen.clear();
}
async function lvFetch() {
  const now = Date.now() / 1000, first = !LV.lastT, gap = now - LV.lastT;
  const since = first ? now - 3600 : LV.lastT, step = first ? 5 : (gap > 120 ? Math.ceil(gap / 300) : 0);
  const d = await api("/api/live?since=" + since.toFixed(1) + (step ? "&step=" + step : ""), {timeout: 15000});
  lvMerge(d); LV.err = ""; LV.fails = 0;
}
/* ---- the on-disk history (ranges past one hour): fetched again when the range changes or every max(20 s, one bucket) */
const lvStoreKey = m => String(m.key).startsWith("p:") && m.port ? "p@" + m.port : m.key;      // as PXA Control files it
async function lvTel(force) {
  if (!force && LV.tel && Date.now() / 1000 - LV.telAt < 60) return LV.tel;
  try { LV.tel = await api("/api/telemetry", {timeout: 15000}); } catch (e) { LV.tel = {available: false, error: e.message}; }
  LV.telAt = Date.now() / 1000;
  const on = !!(LV.tel && LV.tel.available && (LV.tel.recording || (LV.tel.series || 0) > 0));
  $$("#lv-range .lv-long").forEach(b => { b.hidden = !on; });
  if (!on && LV.range > LV_KEEP) { LV.range = 3600; $("#lv-range").setv("3600"); }
  return LV.tel;
}
function lvHistSource(d) {
  const idx = c => Object.fromEntries(c.map((n, i) => [n, i]));
  const ci = {card: idx(d.cols.card), server: idx(d.cols.server), host: idx(d.cols.host), req: LV.ci ? LV.ci.req : {}};
  const live = new Map([...LV.servers.values()].map(s => [lvStoreKey(s.meta), s]));
  const cards = new Map(), servers = new Map();
  for (const c of d.cards) {
    const i = +c.key, lc = LV.cards.get(i);
    cards.set(i, {ci, rows: c.rows, meta: {index: i, name: c.label, class: (c.info || {}).class, mem_total_mib: (c.info || {}).mem_total_mib,
                                         slot: i % 8, throttle: lc ? lc.meta.throttle : []}});
  }
  d.servers.forEach((s, n) => {
    const ls = live.get(s.key), info = s.info || {};
    servers.set(s.key, {ci, rows: s.rows, reqs: [], meta: {key: s.key, name: s.label || s.key, slot: ls ? ls.meta.slot : (n + 4) % 8, up: !!ls,
      phase: "idle", gpus: info.gpus || [], port: info.port, model_file: info.model_file,
      has_spec: s.rows.some(r => r[ci.server.acc] != null), has_xcache: s.rows.some(r => r[ci.server.xhit] != null)}});
  });
  const host = d.host ? {ci, rows: d.host.rows, meta: {ram_total_mib: (d.host.info || {}).ram_total_mib, swap_total_mib: LV.host && LV.host.meta ? LV.host.meta.swap_total_mib : 0}} : null;
  return {hist: true, ci, cards, servers, host, ts: d.until, histFrom: d.since, step: d.step, source: d.source, every: LV.every};
}
async function lvHistFetch() {
  const now = Date.now() / 1000, W = Math.max(300, ($("#lv-root") || {}).clientWidth || 900);
  if (LV.hist && LV.histRange === LV.range && now - LV.histAt < Math.max(20, LV.hist.step || 0)) return;
  const d = await api(`/api/telemetry/history?since=${Math.floor(now - LV.range)}&points=${Math.min(1500, Math.round(W / 2))}`, {timeout: 30000});
  LV.hist = lvHistSource(d); LV.histAt = now; LV.histRange = LV.range;
}
async function lvLoop() {
  clearTimeout(LV.timer);
  if (!LV.open) return;
  if (!document.hidden && S.tab === "live" && !LV.busy) {
    LV.busy = true;
    try { await lvFetch(); if (LV.range > LV_KEEP) await lvHistFetch(); lvRender(); }
    catch (e) { LV.err = e.message; if (++LV.fails === 3) toast("Live: " + e.message, true); lvStatus(); }
    finally { LV.busy = false; }
  }
  LV.timer = setTimeout(lvLoop, (LV.every || 2) * 1000);
}
function liveOpen() { if (!LV.built) lvBuild(); LV.open = true; lvTel(true).then(() => lvRender()); lvRender(); lvLoop(); }
function liveClose() { LV.open = false; clearTimeout(LV.timer); }

/* ------------------------------------------------------------------ page
   The skeleton is built once; every refresh only redraws the numbers and the SVG inside it. */
const LV_CHARTS = [
  {id: "dec", title: "Decode speed", sub: "tokens per second, all slots", unit: "t/s", always: true},
  {id: "pre", title: "Prefill speed", sub: "tokens per second, one point per request", unit: "t/s", always: true},
  {id: "acc", title: "Speculation", sub: "share of drafted tokens the model accepted", unit: "%", pct: true, when: s => s.meta.has_spec},
  {id: "xhit", title: "Expert cache", sub: "share of expert lookups served from VRAM", unit: "%", pct: true, when: s => s.meta.has_xcache},
];
function lvBuild() {
  LV.built = true;
  const root = $("#lv-root");
  root.replaceChildren(
    el("div", {id: "lv-hist", class: "tiny lv-histbar"}),
    el("div", {id: "lv-tiles", class: "lv-tiles"}),
    el("div", {id: "lv-charts", class: "lv-charts"}, ...LV_CHARTS.map(c =>
      el("div", {class: "card lv-card", id: "lc-" + c.id},
        el("h3", {}, c.title, el("span", {class: "tiny", text: c.sub})),
        el("div", {class: "lv-chart", id: "lh-" + c.id}),
        el("details", {class: "lv-table"}, el("summary", {text: "table"}), el("div", {class: "tw", id: "lt-" + c.id}))))),
    el("div", {class: "card lv-card"},
      el("h3", {}, "Requests", el("span", {class: "tiny", text: "one bar per request: prompt, prefill, decode (a lane is one server slot)"})),
      el("div", {class: "tiny lv-sum", id: "lv-tl-sum"}),
      el("div", {class: "lv-chart", id: "lv-tl"}),
      el("details", {class: "lv-table"}, el("summary", {text: "table"}), el("div", {class: "tw", id: "lt-tl"}))),
    el("h3", {class: "lv-h"}, "Hardware", el("span", {class: "tiny", text: "per card: memory, load, temperature, power; then this machine's memory and CPU"})),
    el("div", {id: "lv-cards", class: "lv-cards"}),
    el("div", {id: "lv-host", class: "lv-cards lv-hostbox"}),
    el("p", {class: "tiny lv-note", id: "lv-note"}));
  segInit($("#lv-range"), v => {
    LV.range = +v; store.set("lv.range", LV.range); lvRender();
    if (LV.range > LV_KEEP) { LV.busy = true; lvHistFetch().then(lvRender, e => { LV.err = e.message; lvStatus(); }).finally(() => { LV.busy = false; }); }
  });
  $("#lv-range").setv(String(LV.range));
  window.addEventListener("resize", () => { if (LV.open) lvRender(); });
}
function lvStatus() {
  const sub = $("#lv-sub"); if (!sub) return;
  if (LV.err) { sub.textContent = "Cannot reach PXA Control: " + LV.err; return; }
  const span = LV.ts - LV.histFrom, tel = LV.tel || {}, bar = $("#lv-hist");
  if (R.hist) {
    const first = tel.first_ts ? lvDay(tel.first_ts) + " " + lvClock(tel.first_ts, false) : null;
    sub.textContent = `Last ${lvDur(LV.range)} from the history PXA Control records on this machine · one point per ${lvDur(R.step)}` +
      (first ? ` · recorded since ${first}` : "");
  } else {
    sub.textContent = !LV.ts ? "Reading the first sample ..." :
      `Updated ${lvClock(LV.ts, true)} · a sample every ${LV.every || 2} s · ` + (span < LV.range - 10 ?
        (LV.background ? `history starts ${lvClock(LV.histFrom, false)}, when PXA Control started recording` : `history starts ${lvClock(LV.histFrom, false)}, when PXA Control first had a viewer`) :
        "last " + fmtDur(LV.range));
  }
  if (!bar) return;
  if (tel.available && tel.recording) {
    const since = Math.floor(Date.now() / 1000 - LV.range), q = k => `/api/telemetry/csv?kind=${k}&since=${since}`;
    bar.replaceChildren("Recorded in the background, also with this page closed" + (tel.settings ? ` (kept ${tel.settings.raw_days} days, minute averages ${tel.settings.rollup_days} days)` : "") + " · CSV of this range: ",
      el("a", {href: q("card"), download: "", text: "cards"}), " · ", el("a", {href: q("server"), download: "", text: "servers"}), " · ",
      el("a", {href: q("requests"), download: "", text: "requests"}), " · ", el("a", {href: q("host"), download: "", text: "this machine"}));
  } else if (tel.available && tel.writer_pid) {
    bar.textContent = `History is recorded by another PXA Control on this machine (pid ${tel.writer_pid}); this one reads it.`;
  } else if (tel.available === false || (tel.settings && tel.settings.enabled === false)) {
    bar.textContent = "History is off: the charts keep the last hour in memory only" + (tel.error ? ` (${tel.error})` : "") + ".";
  } else bar.textContent = "";
}
function lvSeries(e, kind, col, t0, scale) {
  const i = (e.ci || LV.ci)[kind][col];
  return e.rows.filter(r => r[0] >= t0 - 60).map(r => [r[0], r[i] == null ? null : (scale ? r[i] * scale : r[i])]);
}
function lvLast(e, kind, col, maxAge) {
  const i = LV.ci[kind][col], now = LV.ts;
  for (let k = e.rows.length - 1; k >= 0; k--) { const r = e.rows[k]; if (now - r[0] > maxAge) return null; if (r[i] != null) return r[i]; }
  return null;
}
function lvRender() {
  if (!LV.ci) { lvStatus(); return; }
  R = LV.range > LV_KEEP && LV.hist && LV.histRange === LV.range ? LV.hist : LV;
  const t1 = R.hist ? Math.max(R.ts, LV.ts) : LV.ts, t0 = t1 - LV.range;
  lvStatus();
  const liveServers = [...LV.servers.values()].sort((a, b) => a.meta.slot - b.meta.slot);
  lvTiles(liveServers, LV.ts - 180, LV.ts);                        // the tiles are always "now"
  const servers = R.hist ? [...R.servers.values()].sort((a, b) => a.meta.slot - b.meta.slot) : liveServers;
  const gap = R.hist ? Math.max(25, 2.5 * (R.step || 60)) : 25;
  // the charts: one series per server, coloured by server
  const hold = id => $("#lh-" + id).dataset.hold;
  for (const c of LV_CHARTS) {
    const host = $("#lh-" + c.id), card = $("#lc-" + c.id), use = servers.filter(s => c.always || (c.when && c.when(s)));
    card.hidden = !use.length && !c.always;
    if (card.hidden) continue;
    const mk = (s, pts) => ({id: s.meta.key, name: s.meta.name, color: lvColor(s.meta.slot), pts});
    let series;
    if (c.id === "dec") series = use.map(s => mk(s, lvSeries(s, "server", "dec", t0)));
    else if (c.id === "pre" && R.hist) series = use.map(s => mk(s, lvSeries(s, "server", "pre_req", t0)));
    else if (c.id === "pre") series = use.map(s => mk(s, s.reqs.filter(r => (r[LV.ci.req.prefill_tps] || 0) > 0 && (r[LV.ci.req.prompt_n] || 0) >= 64).map(r => [r[0] - (r[LV.ci.req.gen_ms] || 0) / 1000, r[LV.ci.req.prefill_tps]])));
    else if (c.id === "acc") series = use.map(s => mk(s, lvSeries(s, "server", "acc", t0, 100)));
    else series = use.map(s => mk(s, lvSeries(s, "server", "xhit", t0, 100)));
    const spec = {series, t0, t1, y: c.pct ? {min: 0, max: 100, unit: "%", digits: 0} : {min: 0, unit: c.unit, digits: 1}, area: false,
                  gap: c.id === "pre" && !R.hist ? 1e9 : gap, dots: c.id === "pre" && !R.hist, label: c.title + ", " + c.sub,
                  empty: !servers.length ? "no server is running" : c.id === "pre" ? "no finished request in this range yet" : "no data in this range yet"};
    if (!hold(c.id)) lvChart(host, spec);
    lvTable($("#lt-" + c.id), series, c.pct ? "%" : c.unit, c.pct ? 0 : 1);
  }
  if (R.hist) lvHistSummary(servers, t0); else lvTimelineRender(servers, t0, t1);
  lvCards(t0, t1, gap);
  lvHost(t0, t1, gap);
  $("#lv-note").textContent = "Decode speed is the tokens each server wrote per second, averaged over every sample (exact engine counters when the server runs with --metrics, " +
    "otherwise counted from slot progress). Prefill speed and the request bars come from the server's own record of each finished request. " +
    "Speculation and expert-cache charts appear when a server uses them. A red line marks " + LV_HOT_C + " °C, where cards begin to throttle.";
}
function lvTable(host, series, unit, d) {
  const rows = series.filter(s => s.pts.some(p => p[1] != null)).map(s => {
    const v = s.pts.map(p => p[1]).filter(lvNum), sum = v.reduce((a, b) => a + b, 0);
    return el("tr", {}, el("td", {text: s.name}), el("td", {text: lvFmt(v[v.length - 1], d) + " " + unit}), el("td", {text: lvFmt(sum / v.length, d)}), el("td", {text: lvFmt(Math.max(...v), d)}), el("td", {text: v.length}));
  });
  host.replaceChildren(rows.length ? el("table", {}, el("thead", {}, el("tr", {}, ...["series", "latest", "mean", "peak", "points"].map(h => el("th", {text: h})))), el("tbody", {}, ...rows)) : el("div", {class: "tiny", text: "nothing to list yet"}));
}

/* ---- one tile per server. The tile's frame and its sparkline stay in the page between refreshes (so a hover is not lost);
        only the numbers are rewritten. */
const LV_PHASE = {idle: ["", "idle"], prefill: ["warn", "reading the prompt"], decode: ["ok", "writing"]};
function lvTileMake() {
  const t = {
    sw: el("i", {class: "lv-sw"}), name: el("span", {class: "lv-name"}), chip: el("span", {class: "chip"}), sub: el("div", {class: "tiny lv-sub"}),
    decL: el("span", {class: "l"}), decV: el("span", {class: "v"}), preL: el("span", {class: "l"}), preV: el("span", {class: "v"}),
    spark: el("div", {class: "lv-spark"}), kvs: el("div", {class: "lv-kvs"}), act: el("div", {class: "lv-act"}),
  };
  t.root = el("div", {class: "card lv-tile"}, el("h3", {}, t.sw, t.name, el("span", {class: "sp"}), t.chip), t.sub,
    el("div", {class: "lv-big"}, el("div", {class: "stat"}, t.decL, t.decV), el("div", {class: "stat"}, t.preL, t.preV)), t.spark, t.kvs, t.act);
  return t;
}
function lvTiles(servers, t0, t1) {
  const box = $("#lv-tiles");
  LV.tiles = LV.tiles || new Map();
  if (!servers.length) {
    LV.tiles.clear();
    box.replaceChildren(el("div", {class: "card lv-emptycard"}, el("b", {text: "No server is running"}),
      el("span", {class: "muted", text: "Start one on the Launch tab and its speed, slots and caches appear here. The cards below are live either way."}),
      el("div", {}, el("button", {class: "b pri", onclick: () => showTab("launch"), text: "Open Launch"}))));
    return;
  }
  const keys = servers.map(s => s.meta.key);
  for (const k of [...LV.tiles.keys()]) if (!keys.includes(k)) LV.tiles.delete(k);
  for (const k of keys) if (!LV.tiles.has(k)) LV.tiles.set(k, lvTileMake());
  const same = box.children.length === keys.length && keys.every((k, i) => box.children[i] === LV.tiles.get(k).root);
  if (!same) box.replaceChildren(...keys.map(k => LV.tiles.get(k).root));
  const ci = LV.ci.server, rc = LV.ci.req;
  for (const s of servers) {
    const m = s.meta, t = LV.tiles.get(m.key), last = s.rows.length ? s.rows[s.rows.length - 1] : null;
    const live = last && t1 - last[0] < 10, req = s.reqs.length ? s.reqs[s.reqs.length - 1] : null;
    const busy = live && m.phase !== "idle";
    const dec = busy ? lvLast(s, "server", "dec", 10) : (req && req[rc.decode_tps] ? req[rc.decode_tps] : null);
    const preNow = lvLast(s, "server", "pre", 10), preReq = req && req[rc.prefill_tps] && (req[rc.prompt_n] || 0) >= 64 ? req[rc.prefill_tps] : null;
    const ph = !m.up ? ["", "stopped"] : !live ? ["warn", "no reading"] : LV_PHASE[m.phase] || LV_PHASE.idle;
    const ctx = lvLast(s, "server", "ctx", 20), acc = lvLast(s, "server", "ema", 30) != null ? lvLast(s, "server", "ema", 30) : lvLast(s, "server", "acc", 30);
    const xh = lvLast(s, "server", "xhit", 30), xs = lvLast(s, "server", "xswap", 30);
    t.sw.style.background = lvColor(m.slot);
    t.name.textContent = m.name || m.key;
    t.chip.className = "chip " + ph[0];
    t.chip.replaceChildren(el("span", {class: "lv-dot2 " + ph[0], "aria-hidden": "true"}), ph[1]);
    t.sub.textContent = [m.model_file || m.model, m.gpus && m.gpus.length ? "cards " + m.gpus.join(",") : null, m.port ? ":" + m.port : null].filter(Boolean).join(" · ");
    t.decL.textContent = busy ? "decode now" : "decode, last request";
    t.decV.replaceChildren(dec != null ? lvFmt(dec) : "-", el("small", {text: "t/s"}));
    t.preL.textContent = preNow != null ? "prefill now" : "prefill, last request";
    t.preV.replaceChildren(lvFmt(preNow != null ? preNow : preReq, 0), el("small", {text: "t/s"}));
    const kv = (k, v, sub) => el("div", {class: "kv"}, k, el("b", {}, v, sub ? el("small", {text: " " + sub}) : null));
    t.kvs.replaceChildren(...[
      kv("slots busy", m.total_slots != null ? `${last && last[ci.busy] != null ? last[ci.busy] : 0} / ${m.total_slots}` : "-"),
      kv("KV cache in use", ctx != null ? Math.round(ctx * 100) + " %" : "-", ctx != null && m.n_ctx ? `${lvTok(ctx * m.n_ctx)} of ${lvTok(m.n_ctx)}` : ""),
      ctx != null ? el("div", {class: "bar lv-meter", role: "img", "aria-label": "KV cache in use"}, el("i", {style: `width:${Math.round(ctx * 100)}%;background:${lvColor(m.slot)}`})) : null,
      m.has_spec ? kv("draft acceptance", acc != null ? Math.round(acc * 100) + " %" : "-") : null,
      m.has_xcache ? kv("expert cache hits", xh != null ? Math.round(xh * 100) + " %" : "-", xs != null && xs > 0 ? lvFmt(xs) + " swaps/s" : "") : null,
      req ? kv("last request", `${lvTok(req[rc.n_gen])} tokens`, lvFmt(req[rc.decode_tps]) + " t/s · " + fmtAgo(req[0])) : null].filter(Boolean));
    t.act.replaceChildren(...(m.active || []).map(a => {
      const tot = a.n_remain != null ? a.n_decoded + a.n_remain : null, pct = tot ? Math.min(100, 100 * a.n_decoded / tot) : (a.phase === "prefill" ? 0 : 100);
      return el("div", {class: "lv-slot"}, el("div", {class: "kv"}, `slot ${a.id} · ${a.phase === "decode" ? "writing" : "reading the prompt"}`,
        el("b", {text: a.phase === "decode" ? lvTok(a.n_decoded) + (tot ? " / " + lvTok(tot) : "") + " tokens" : "prefill"})),
        el("div", {class: "bar"}, el("i", {style: `width:${pct}%;background:${lvColor(m.slot)}`})));
    }));
    if (!t.spark.dataset.hold) lvChart(t.spark, {series: [{id: m.key, name: m.name, color: lvColor(m.slot), pts: lvSeries(s, "server", "dec", t1 - 180)}], t0: t1 - 180, t1,
      y: {min: 0, unit: "t/s", digits: 1}, mini: true, area: true, label: "decode tokens per second, last 3 minutes", tipNote: "decode"});
  }
}

/* ---- a long range: no bars (thousands of requests), the totals of what the store counted per bucket instead */
function lvHistSummary(servers, t0) {
  const ci = R.ci.server, sum = {n: 0, read: 0, readMs: 0, gen: 0, genMs: 0};
  for (const s of servers) for (const r of s.rows) if (r[0] >= t0) {
    sum.n += r[ci.req] || 0; sum.read += r[ci.prompt_tok] || 0; sum.readMs += r[ci.prompt_ms] || 0; sum.gen += r[ci.gen_tok] || 0; sum.genMs += r[ci.gen_ms] || 0;
  }
  $("#lv-tl-sum").textContent = !sum.n ? "no finished request recorded in this range" :
    `${Math.round(sum.n).toLocaleString("en-US")} requests in the last ${lvDur(LV.range)} \u00b7 ${lvTok(sum.read)} prompt tokens read` +
    (sum.readMs > 0 && sum.read >= 64 ? ` at ${lvFmt(sum.read / (sum.readMs / 1000), 0)} t/s` : "") + ` \u00b7 ${lvTok(sum.gen)} tokens written` +
    (sum.genMs > 0 ? ` at ${lvFmt(sum.gen / (sum.genMs / 1000))} t/s` : "");
  $("#lv-tl").replaceChildren(el("div", {class: "tiny", text: "One bar per request is drawn for ranges up to 1 h. Every request of the last days is in the requests CSV above."}));
  $("#lt-tl").replaceChildren();
}

/* ---- the request timeline */
function lvTimelineRender(servers, t0, t1) {
  const host = $("#lv-tl");
  const rc = LV.ci.req, lanes = new Map(), tableRows = [];
  const lane = (s, slot) => {
    const id = s.meta.key + "|" + slot;
    if (!lanes.has(id)) lanes.set(id, {id, order: s.meta.slot * 1000 + (slot < 0 ? 999 : slot), color: lvColor(s.meta.slot),
      label: s.meta.name, suffix: (s.meta.total_slots || 2) > 1 ? ` · slot ${slot < 0 ? "?" : slot}` : "", bars: []});
    return lanes.get(id);
  };
  for (const s of servers) {
    for (const r of s.reqs) {
      const end = r[0], tg = (r[rc.gen_ms] || 0) / 1000, tp = (r[rc.prompt_ms] || 0) / 1000;
      if (end < t0) continue;
      const tip = [`${s.meta.name} · slot ${r[rc.slot]} · ${lvClock(end - tg - tp, true)} to ${lvClock(end, true)}`,
        `prompt ${lvTok(r[rc.n_prompt])} tokens` + (r[rc.n_cached] ? ` (${lvTok(r[rc.n_cached])} reused)` : "") +
          (r[rc.prompt_n] ? `, read ${lvTok(r[rc.prompt_n])} in ${lvFmt(tp)} s` + (r[rc.prefill_tps] ? ` = ${lvFmt(r[rc.prefill_tps], 0)} t/s` : "") : ""),
        `wrote ${lvTok(r[rc.n_gen])} tokens in ${lvFmt(tg)} s` + (r[rc.decode_tps] ? ` = ${lvFmt(r[rc.decode_tps])} t/s` : "")];
      if (r[rc.draft_n]) tip.push(`draft acceptance ${Math.round(100 * r[rc.draft_acc] / r[rc.draft_n])} % (${r[rc.draft_acc]} of ${r[rc.draft_n]})`);
      lane(s, r[rc.slot]).bars.push({a: end - tg - tp, b: end - tg, c: end, tip});
      tableRows.push([end, s.meta.name, r[rc.slot], r[rc.n_prompt], r[rc.n_gen], r[rc.prefill_tps], r[rc.decode_tps]]);
    }
    for (const a of s.meta.active || []) {
      lane(s, a.id).bars.push({a: a.t0, b: a.t1 == null ? t1 : a.t1, c: t1, live: true,
        tip: [`${s.meta.name} · slot ${a.id} · running since ${lvClock(a.t0, true)}`, a.phase === "decode" ? `wrote ${lvTok(a.n_decoded)} tokens so far` : "reading the prompt"]});
    }
  }
  for (const g of LV.ghosts) {
    const s = LV.servers.get(g.key);
    if (s) lane(s, g.slot).bars.push({a: g.a, b: g.b == null ? g.c : g.b, c: g.c, tip: [`${s.meta.name} · slot ${g.slot} · just finished`, "the server's own record of it is on its way"]});
  }
  let list = [...lanes.values()].sort((a, b) => a.order - b.order), more = 0;
  if (list.length > 14) {
    list.sort((a, b) => Math.max(...b.bars.map(x => x.c)) - Math.max(...a.bars.map(x => x.c)));
    more = list.length - 14; list = list.slice(0, 14).sort((a, b) => a.order - b.order);
  }
  if (!host.dataset.hold) {
    lvTimeline(host, {lanes: list, t0, t1, label: "request timeline", empty: servers.length ? "no requests in this range yet: send a prompt from the Chat tab" : "no server is running"});
    if (more) host.append(el("div", {class: "tiny", text: `${more} quieter lane(s) hidden`}));
  }
  const sum = {n: tableRows.length, read: 0, readMs: 0, reused: 0, gen: 0, genMs: 0};
  for (const s of servers) for (const r of s.reqs) if (r[0] >= t0) {
    sum.read += r[rc.prompt_n] || 0; sum.readMs += r[rc.prompt_ms] || 0; sum.reused += r[rc.n_cached] || 0; sum.gen += r[rc.n_gen] || 0; sum.genMs += r[rc.gen_ms] || 0;
  }
  $("#lv-tl-sum").textContent = !sum.n ? "" : `${sum.n} request${sum.n === 1 ? "" : "s"} in this range \u00b7 ${lvTok(sum.read)} prompt tokens read` + (sum.readMs > 0 && sum.read >= 64 ? ` at ${lvFmt(sum.read / (sum.readMs / 1000), 0)} t/s` : "") +
    (sum.reused ? ` (${lvTok(sum.reused)} reused)` : "") + ` \u00b7 ${lvTok(sum.gen)} tokens written` + (sum.genMs > 0 ? ` at ${lvFmt(sum.gen / (sum.genMs / 1000))} t/s` : "");
  tableRows.sort((a, b) => b[0] - a[0]);
  $("#lt-tl").replaceChildren(tableRows.length ? el("table", {}, el("thead", {}, el("tr", {}, ...["finished", "server", "slot", "prompt", "written", "prefill t/s", "decode t/s"].map(h => el("th", {text: h})))),
    el("tbody", {}, ...tableRows.slice(0, 60).map(r => el("tr", {}, el("td", {text: lvClock(r[0], true)}), el("td", {text: r[1]}), el("td", {text: r[2]}),
      el("td", {text: r[3] == null ? "-" : r[3]}), el("td", {text: r[4] == null ? "-" : r[4]}), el("td", {text: lvFmt(r[5], 0)}), el("td", {text: lvFmt(r[6])}))))) :
    el("div", {class: "tiny", text: "no finished request in this range"}));
}

/* ---- the cards: one strip each; the strips stay in the page and only their numbers and lines are rewritten */
function lvStripMake(label, titles) {
  const mk = title => { const p = {root: el("div", {class: "lv-panel"}), l: el("div", {class: "l", text: title}), v: el("div", {class: "v"}), s: el("div", {class: "tiny"}), host: el("div", {class: "lv-mini"}), title};
    p.root.append(p.l, p.v, p.s, p.host); return p; };
  const st = {sw: el("i", {class: "lv-sw"}), name: el("span", {class: "lv-name"}), cls: el("div", {class: "tiny"}), chips: el("div", {class: "row", style: "gap:6px"}), panels: {}};
  for (const [k, title] of Object.entries(titles)) st.panels[k] = mk(title);
  st.root = el("div", {class: "card lv-strip"}, el("div", {class: "lv-sh"}, el("h3", {}, st.sw, el("span", {class: "mono", text: label}), st.name), st.cls, st.chips),
    el("div", {class: "lv-panels"}, ...Object.values(st.panels).map(p => p.root)));
  return st;
}
function lvFill(p, color, big, unit, sub, pts, o, t0, t1, what) {       // the number, its caption and the line of one panel
  p.v.replaceChildren(big, ...(unit ? [el("small", {text: " " + unit})] : [])); p.s.textContent = sub || "\u00a0";
  if (!p.host.dataset.hold) lvChart(p.host, Object.assign({series: [{id: p.title, name: p.title, color, pts}], t0, t1, mini: true, area: true, label: `${what} ${p.title}`, tipNote: p.title}, o));
}
function lvCards(t0, t1, gap) {
  const box = $("#lv-cards"), ci = R.ci.card, fresh = R.hist ? Math.max(60, 3 * (R.step || 60)) : 15;
  const cards = [...R.cards.values()].sort((a, b) => a.meta.index - b.meta.index);
  LV.strips = LV.strips || new Map();
  if (!cards.length) { LV.strips.clear(); box.replaceChildren(el("div", {class: "empty", text: "no NVIDIA cards visible"})); return; }
  const idx = cards.map(c => c.meta.index);
  for (const k of [...LV.strips.keys()]) if (!idx.includes(k)) LV.strips.delete(k);
  for (const k of idx) if (!LV.strips.has(k)) LV.strips.set(k, lvStripMake("#" + k, {mem: "memory", util: "load", temp: "temperature", power: "power"}));
  if (!(box.children.length === idx.length && idx.every((k, i) => box.children[i] === LV.strips.get(k).root))) box.replaceChildren(...idx.map(k => LV.strips.get(k).root));
  const holders = new Map();
  for (const s of LV.servers.values()) for (const g of s.meta.gpus || []) { if (!holders.has(g)) holders.set(g, []); holders.get(g).push(s.meta.name); }
  for (const c of cards) {
    const m = c.meta, st = LV.strips.get(m.index), color = lvColor(m.slot), tot = m.mem_total_mib || null;
    const lastv = k => { for (let i = c.rows.length - 1; i >= 0; i--) { if (t1 - c.rows[i][0] > fresh) return null; if (c.rows[i][ci[k]] != null) return c.rows[i][ci[k]]; } return null; };
    const mem = lastv("mem"), util = lastv("util"), temp = lastv("temp"), pow = lastv("power"), lim = lastv("limit"), clk = lastv("clock");
    const hot = temp != null && temp >= LV_HOT_C, used = holders.get(m.index);
    st.sw.style.background = color; st.name.textContent = m.name || ""; st.cls.textContent = m.class || "";
    st.chips.replaceChildren(used ? el("span", {class: "chip", title: used.join(", "), text: lvShort(used.join(", "), 24)}) : el("span", {class: "chip", text: "free"}),
      ...(m.throttle || []).map(r => el("span", {class: "chip warn", text: "⚠ " + r.replace(/^(sw_|hw_)/, "").replace(/_/g, " ")})));
    const pts = (k, f) => c.rows.filter(r => r[0] >= t0 - 60).map(r => [r[0], r[ci[k]] == null ? null : f(r[ci[k]])]);
    const fill = (p, big, unit, sub, spec, o) => lvFill(p, color, big, unit, sub, spec, Object.assign({gap}, o), t0, t1, `card ${m.index}`);
    const P = st.panels, pmax = Math.ceil((lim || Math.max(100, ...c.rows.map(r => r[ci.power] || 0))) / 50) * 50;
    fill(P.mem, mem == null ? "-" : lvFmt(mem / 1024), "GiB", tot ? `of ${lvFmt(tot / 1024, tot >= 10240 ? 0 : 1)} GiB · ${mem == null ? "-" : Math.round(100 * mem / tot)} %` : "",
      pts("mem", v => tot ? 100 * v / tot : v), {y: {min: 0, max: 100, unit: "%", digits: 0}});
    fill(P.util, util == null ? "-" : lvFmt(util, 0), "%", clk != null ? lvFmt(clk, 0) + " MHz" : "", pts("util", v => v), {y: {min: 0, max: 100, unit: "%", digits: 0}});
    fill(P.temp, temp == null ? "-" : lvFmt(temp, 0), "°C", hot ? "hot: expect throttling" : "", pts("temp", v => v), {y: {min: 20, max: 100, unit: "°C", digits: 0}, marks: [{v: LV_HOT_C}]});
    fill(P.power, pow == null ? "-" : lvFmt(pow, 0), "W", lim != null ? `of ${lvFmt(lim, 0)} W limit` : "", pts("power", v => v), {y: {min: 0, max: pmax, unit: "W", digits: 0}});
  }
}
/* ---- this machine: memory and CPU (where a model's host-side tables and offloaded experts live) */
function lvHost(t0, t1, gap) {
  const box = $("#lv-host"), h = R.host, fresh = R.hist ? Math.max(60, 3 * (R.step || 60)) : 15;
  if (!h || !h.rows.length) { box.replaceChildren(); return; }
  const hi = R.ci.host, color = lvColor(6), tot = h.meta.ram_total_mib, swapTot = h.meta.swap_total_mib || 0;
  if (!LV.hostStrip) { LV.hostStrip = lvStripMake("", {ram: "memory", cpu: "CPU", swap: "swap"}); LV.hostStrip.name.textContent = "This machine"; LV.hostStrip.sw.style.background = color; }
  const st = LV.hostStrip;
  if (box.firstChild !== st.root) box.replaceChildren(st.root);
  const lastv = k => { for (let i = h.rows.length - 1; i >= 0; i--) { if (t1 - h.rows[i][0] > fresh) return null; if (h.rows[i][hi[k]] != null) return h.rows[i][hi[k]]; } return null; };
  const ram = lastv("ram"), cpu = lastv("cpu"), swap = lastv("swap");
  const pts = (k, f) => h.rows.filter(r => r[0] >= t0 - 60).map(r => [r[0], r[hi[k]] == null ? null : f(r[hi[k]])]);
  const full = ram != null && tot && ram / tot > 0.92, swapping = swap != null && swap > 1024;
  st.chips.replaceChildren(...[full ? el("span", {class: "chip warn", text: "\u26A0 memory almost full"}) : null, swapping ? el("span", {class: "chip warn", text: "\u26A0 swapping"}) : null].filter(Boolean));
  const P = st.panels, GiB = v => lvFmt(v / 1024, v / 1024 >= 100 ? 0 : 1);
  lvFill(P.ram, color, ram == null ? "-" : GiB(ram), "GiB", tot ? `of ${GiB(tot)} GiB \u00b7 ${ram == null ? "-" : Math.round(100 * ram / tot)} % in use` : "", pts("ram", v => tot ? 100 * v / tot : v), {y: {min: 0, max: 100, unit: "%", digits: 0}, gap}, t0, t1, "this machine");
  lvFill(P.cpu, color, cpu == null ? "-" : lvFmt(cpu, 0), "%", "all cores", pts("cpu", v => v), {y: {min: 0, max: 100, unit: "%", digits: 0}, gap}, t0, t1, "this machine");
  lvFill(P.swap, color, swapTot ? (swap == null ? "-" : GiB(swap)) : "none", swapTot ? "GiB" : "", swapTot ? `of ${GiB(swapTot)} GiB` : "no swap on this machine", swapTot ? pts("swap", v => 100 * v / swapTot) : [], {y: {min: 0, max: 100, unit: "%", digits: 0}, gap}, t0, t1, "this machine");
}
if (typeof S !== "undefined" && S.tab === "live") liveOpen();
