/* many-batch-align — interactive figures. Every number comes from data/site.json
   (scripts/40_export_site_data.py). Colours are CSS tokens, so the theme toggle needs no redraw
   except for charts that compute colours (heatmaps). */
(() => {
  'use strict';
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const DUR = reduceMotion ? 0 : 750;
  const renderers = [];          // redrawn on resize and theme change
  const fmt2 = d3.format('.2f'), fmt3 = d3.format('.3f'), fmtInt = d3.format(',');
  const series = k => `var(--s${(k % 5) + 1})`;
  const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ── Tooltip (DOM built with textContent: labels are data) ────────────────
  const tip = document.getElementById('tooltip');
  function showTip(evt, head, rows) {
    tip.replaceChildren();
    if (head) { const h = document.createElement('div'); h.className = 'tt-head'; h.textContent = head; tip.append(h); }
    for (const r of rows) {
      const row = document.createElement('div'); row.className = 'row';
      const key = document.createElement('span'); key.className = 'key';
      if (r.color) { const i = document.createElement('i'); i.style.background = r.color; key.append(i); }
      key.append(document.createTextNode(r.label));
      const v = document.createElement('b'); v.textContent = r.value;
      row.append(key, v); tip.append(row);
    }
    const x = evt.clientX, y = evt.clientY, w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.min(x + 14, window.innerWidth - w - 8)}px`;
    tip.style.top = `${y + 14 + h > window.innerHeight ? y - h - 10 : y + 14}px`;
    tip.classList.add('on');
  }
  const hideTip = () => tip.classList.remove('on');

  // ── Chart scaffolding ─────────────────────────────────────────────────────
  function frame(id, height, margin = { top: 12, right: 16, bottom: 34, left: 48 }) {
    const el = document.getElementById(id);
    el.replaceChildren();
    const width = Math.max(300, el.clientWidth);
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    return { el, svg, g, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom, width, margin };
  }
  function axes(g, x, y, w, h, { xTitle, yTitle, xTicks = 8, yTicks = 5, yFmt } = {}) {
    g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(yTicks).tickSize(-w).tickFormat(''));
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(xTicks).tickSizeOuter(0));
    g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(yTicks).tickSizeOuter(0).tickFormat(yFmt || null))
      .call(s => s.select('.domain').remove());
    if (xTitle) g.append('text').attr('class', 'axis-title').attr('x', w).attr('y', h + 30).attr('text-anchor', 'end').text(xTitle);
    if (yTitle) g.append('text').attr('class', 'axis-title').attr('x', 0).attr('y', -2).attr('text-anchor', 'start').text(yTitle);
  }
  function legend(id, items) {
    const el = document.getElementById(id); if (!el) return;
    el.replaceChildren();
    for (const it of items) {
      const s = document.createElement('span'); const i = document.createElement('i');
      if (it.kind) i.className = it.kind;
      if (it.kind !== 'ring') i.style.background = it.color;
      if (it.dash) { i.style.background = `repeating-linear-gradient(90deg, ${it.color} 0 4px, transparent 4px 7px)`; }
      s.append(i, document.createTextNode(it.label)); el.append(s);
    }
  }
  function segGroup(selector, attr, onChange) {
    const btns = [...document.querySelectorAll(selector)];
    btns.forEach(b => b.addEventListener('click', () => {
      btns.forEach(o => o.setAttribute('aria-pressed', String(o === b)));
      onChange(b.dataset[attr]);
    }));
  }
  function chips(containerId, items, initial, onChange) {
    const el = document.getElementById(containerId); el.replaceChildren();
    const btns = items.map(it => {
      const b = document.createElement('button'); b.className = 'chip'; b.textContent = it.label;
      b.setAttribute('aria-pressed', String(it.key === initial));
      b.addEventListener('click', () => { btns.forEach(o => o.setAttribute('aria-pressed', String(o === b))); onChange(it.key); });
      el.append(b); return b;
    });
  }
  // text colour for a filled cell: whichever of near-black / white has the higher WCAG contrast
  const relLum = c => { const { r, g, b } = d3.rgb(c); const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const INK_L = relLum('#0b0b0b');
  const inkOn = c => { const L = relLum(c); return (L + 0.05) / (INK_L + 0.05) >= 1.05 / (L + 0.05) ? '#0b0b0b' : '#ffffff'; };

  // ── Boot ──────────────────────────────────────────────────────────────────
  // data/site.js sets window.SITE_DATA (works from file://); fall back to fetching the JSON.
  const load = window.SITE_DATA ? Promise.resolve(window.SITE_DATA) : fetch('data/site.json').then(r => r.json());
  load.then(D => {
    const K = D.batches.length, T = D.meta.n_bins, BIN = D.meta.bin_min, REF = D.meta.ref;
    const names = D.batches.map(b => b.name);
    const tMin = d3.range(T).map(i => i * BIN);
    const tMax = tMin[T - 1];
    const pairLabel = (a, b) => `${names[a]} × ${names[b]}`;
    const shiftArr = (obj, a, b) => obj[`${a}-${b}`];                  // bins, length T
    const interp = (arr, x) => {                                         // arr on 0..T-1 at fractional bin x; null = clamped
      if (x == null) return null;
      if (x <= 0) return arr[0]; if (x >= T - 1) return arr[T - 1];
      const i = Math.floor(x), f = x - i;
      return arr[i] == null || arr[i + 1] == null ? null : arr[i] * (1 - f) + arr[i + 1] * f;
    };
    const mapBins = (a, b, x, model = 'joint') => { const s = interp(shiftArr(D.warps[model], a, b), x); return s == null ? null : x + s; };
    const ok = v => v != null && Number.isFinite(v);
    const fmtMin = v => (ok(v) ? `${fmt2(v)} min` : '— (outside the run)');
    const scaleArr = arr => arr.map(v => (ok(v) ? v * BIN : null));
    // keep end-of-line labels apart (min gap in px), preserving order
    function dodge(ys, gap = 13) {
      const idx = ys.map((y, i) => i).sort((p, q) => ys[p] - ys[q]); const out = ys.slice();
      for (let n = 1; n < idx.length; n++) if (out[idx[n]] - out[idx[n - 1]] < gap) out[idx[n]] = out[idx[n - 1]] + gap;
      return out;
    }

    // ── Bindings and counters ──
    const cyc = D.cycle;
    const binds = {
      n_anchors: fmtInt(D.meta.n_anchors), n_batches: String(K), ref_name: names[REF],
      eval_start_min: `${Math.round(D.meta.eval_start_bin * BIN)}`,
      cycle_joint: fmt3(d3.mean(cyc, d => d.joint)), cycle_pairwise: fmt3(d3.mean(cyc, d => d.pairwise)),
      coverage_pct: D.coverage.predictive != null ? `${Math.round(D.coverage.predictive * 100)} %` : 'n/a',
      gcmsalign: D.meta.gcmsalign, generated: D.meta.generated,
    };
    document.querySelectorAll('[data-bind]').forEach(el => { el.textContent = binds[el.dataset.bind] ?? ''; });
    const counts = { n_samples: d3.sum(D.batches, b => b.n), n_sample_pairs: D.meta.n_sample_pairs, n_anchors: D.meta.n_anchors };
    const countObs = new IntersectionObserver(es => es.forEach(e => {
      if (!e.isIntersecting) return; countObs.unobserve(e.target);
      const target = counts[e.target.dataset.count];
      d3.select(e.target).transition().duration(reduceMotion ? 0 : 1600).ease(d3.easeCubicOut)
        .tween('text', function () { const i = d3.interpolateNumber(0, target); return t => { this.textContent = fmtInt(Math.round(i(t))); }; });
    }), { threshold: 0.4 });
    document.querySelectorAll('[data-count]').forEach(el => countObs.observe(el));

    const batchLegend = id => legend(id, names.map((n, k) => ({ label: `${n} (n=${D.batches[k].n})${k === REF ? ', reference' : ''}`, color: series(k) })));

    // ══ Ridgeline (hero + chapter 01) ══════════════════════════════════════
    function ridge(id, { height, interactive, getFrame, getOverlay }) {
      const ampF = interactive ? 1.45 : 1.6;
      const inner = height - (interactive ? 44 : 28);                    // plot height before headroom
      const head = (inner / K) * (ampF - 1) + 6;                         // room for the top row's peaks
      const m = interactive ? { top: head, right: 92, bottom: 34, left: 16 } : { top: head, right: 84, bottom: 22, left: 6 };
      const f = frame(id, height + head, m);
      const x = d3.scaleLinear().domain([0, tMax]).range([0, f.w]);
      const rowH = f.h / K, amp = rowH * ampF;
      const y = d3.scaleLinear().domain([0, 1]).range([0, -amp]).clamp(true);
      const line = d3.line().x((v, i) => x(tMin[i])).y(v => y(Math.min(v, 1.15))).curve(d3.curveMonotoneX);
      f.g.append('rect').attr('x', 0).attr('y', -6).attr('width', x(D.meta.eval_start_bin * BIN)).attr('height', f.h + 6)
        .style('fill', 'var(--surface-2)').attr('rx', 4);
      if (interactive) f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`)
        .call(d3.axisBottom(x).ticks(9).tickSizeOuter(0).tickFormat(d => `${d}`))
        .append('text').attr('class', 'axis-title').attr('x', f.w).attr('y', 30).attr('text-anchor', 'end').text('retention time (min)');
      const rowY = (k, overlay) => overlay ? f.h : (k + 1) * rowH;
      const rows = f.g.selectAll('g.row').data(D.tic).join('g').attr('class', 'row')
        .attr('transform', (d, k) => `translate(0,${rowY(k, getOverlay())})`);
      rows.append('line').attr('x1', 0).attr('x2', f.w).style('stroke', 'var(--grid)');
      rows.selectAll('path').data((d, k) => d[getFrame()].map(s => ({ s, k })))
        .join('path').attr('d', d => line(d.s)).style('fill', 'none').style('stroke', d => series(d.k))
        .style('stroke-width', interactive ? 1.5 : 1.3).style('stroke-opacity', 0.85).style('stroke-linejoin', 'round');
      const labels = rows.append('text').attr('class', 'direct-label').attr('x', f.w + 8).attr('y', -4)
        .text((d, k) => names[k]).style('fill', (d, k) => series(k)).style('opacity', getOverlay() ? 0 : 1);
      const api = {
        update(dur = DUR) {
          const fr = getFrame(), ov = getOverlay();
          rows.transition().duration(dur).ease(d3.easeCubicInOut).attr('transform', (d, k) => `translate(0,${rowY(k, ov)})`);
          rows.each(function (d) {
            d3.select(this).selectAll('path').data(d[fr]).transition().duration(dur).ease(d3.easeCubicInOut).attr('d', s => line(s));
          });
          labels.transition().duration(dur).style('opacity', ov ? 0 : 1);
        },
      };
      if (interactive) {
        const cross = f.g.append('line').attr('class', 'crosshair').attr('y1', -6).attr('y2', f.h).style('opacity', 0);
        f.g.append('rect').attr('width', f.w).attr('height', f.h).style('fill', 'transparent')
          .on('pointermove', evt => {
            const [mx] = d3.pointer(evt); const tb = Math.max(0, Math.min(T - 1, x.invert(mx) / BIN));
            cross.attr('x1', x(tb * BIN)).attr('x2', x(tb * BIN)).style('opacity', 1);
            const shared = getFrame() === 'aligned';
            // shared frame = reference batch's clock; show where each batch has this moment on its own clock
            const refT = shared ? tb : null;
            const rowsOut = names.map((n, k) => {
              const own = shared ? mapBins(REF, k, refT) : tb;
              const inRef = shared ? refT : mapBins(k, REF, tb);
              const v = shared ? own : inRef;
              return { label: n, color: series(k),
                value: !ok(v) ? '— (outside the run)' : shared ? `${fmt2(own * BIN)} min on its clock` : `≙ ${fmt2(inRef * BIN)} min in ${names[REF]}` };
            });
            showTip(evt, shared ? `${fmt2(tb * BIN)} min in the shared frame (${names[REF]})` : `${fmt2(tb * BIN)} min on each batch's own clock`, rowsOut);
          })
          .on('pointerleave', () => { cross.style('opacity', 0); hideTip(); });
      }
      return api;
    }

    // Hero
    let heroFrame = 'raw', heroPlay = !reduceMotion, heroTimer = null, hero;
    const heroLabel = document.getElementById('hero-state');
    const setHeroLabel = () => {
      heroLabel.classList.toggle('aligned', heroFrame === 'aligned');
      heroLabel.lastElementChild.textContent = heroFrame === 'aligned'
        ? `Aligned: every batch warped into ${names[REF]}'s clock` : 'Unaligned: each batch on its own clock';
    };
    function drawHero() { hero = ridge('hero-chart', { height: 330, interactive: false, getFrame: () => heroFrame, getOverlay: () => false }); }
    function heroTick() { heroFrame = heroFrame === 'raw' ? 'aligned' : 'raw'; setHeroLabel(); hero.update(1400); }
    function setPlay(on) {
      heroPlay = on; clearInterval(heroTimer);
      if (on) heroTimer = setInterval(heroTick, 3400);
      document.getElementById('hero-play').setAttribute('aria-pressed', String(on));
    }
    document.getElementById('hero-play').addEventListener('click', () => { setPlay(!heroPlay); document.getElementById('hero-raw').setAttribute('aria-pressed', 'false'); document.getElementById('hero-al').setAttribute('aria-pressed', 'false'); });
    for (const [id, fr] of [['hero-raw', 'raw'], ['hero-al', 'aligned']]) {
      document.getElementById(id).addEventListener('click', () => {
        setPlay(false); heroFrame = fr; setHeroLabel(); hero.update(1100);
        document.getElementById('hero-raw').setAttribute('aria-pressed', String(fr === 'raw'));
        document.getElementById('hero-al').setAttribute('aria-pressed', String(fr === 'aligned'));
      });
    }
    drawHero(); renderers.push(drawHero); setHeroLabel(); batchLegend('hero-legend'); setPlay(heroPlay);

    // Chapter 01
    let rFrame = 'raw', rOverlay = false, rApi;
    const drawRidge = () => { rApi = ridge('ridge-chart', { height: 470, interactive: true, getFrame: () => rFrame, getOverlay: () => rOverlay }); };
    drawRidge(); renderers.push(drawRidge); batchLegend('ridge-legend');
    segGroup('[data-frame]', 'frame', v => { rFrame = v; rApi.update(); });
    document.getElementById('ridge-overlay').addEventListener('change', e => { rOverlay = e.target.checked; rApi.update(); });

    // ══ 02 Cycle consistency ════════════════════════════════════════════════
    const worst = cyc.reduce((a, b) => (b.pairwise > a.pairwise ? b : a));
    let route = worst.t.slice(), cycModel = 'pairwise', pending = [];
    function drawCycChart() {
      const [a, b, c] = route;
      document.getElementById('cyc-route').textContent = `${names[a]} → ${names[b]} → ${names[c]}`;
      const f = frame('cyc-chart', 330, { top: 14, right: 18, bottom: 36, left: 52 });
      const sub = (v, i) => (ok(v) ? v - i : null);
      const direct = d3.range(T).map(i => sub(mapBins(a, c, i, cycModel), i));
      const composed = d3.range(T).map(i => sub(mapBins(b, c, mapBins(a, b, i, cycModel), cycModel), i));
      const all = direct.concat(composed).filter(ok).map(v => v * BIN);
      const x = d3.scaleLinear().domain([0, tMax]).range([0, f.w]);
      const pad = Math.max(0.3, (d3.max(all) - d3.min(all)) * 0.25);
      const y = d3.scaleLinear().domain([d3.min(all) - pad, d3.max(all) + pad]).nice().range([f.h, 0]);
      axes(f.g, x, y, f.w, f.h, { xTitle: `retention time in ${names[a]} (min)`, yTitle: `shift into ${names[c]} (min)` });
      f.g.append('rect').attr('x', x(5)).attr('width', x(40) - x(5)).attr('y', 0).attr('height', f.h)
        .style('fill', 'var(--surface-2)').style('opacity', 0.5).lower();
      const area = d3.area().defined((d, i) => ok(direct[i]) && ok(composed[i])).x((d, i) => x(tMin[i]))
        .y0((d, i) => y(direct[i] * BIN)).y1((d, i) => y(composed[i] * BIN)).curve(d3.curveMonotoneX);
      const line = arr => d3.line().defined(ok).x((d, i) => x(tMin[i])).y(d => y(d * BIN)).curve(d3.curveMonotoneX)(arr);
      f.g.append('path').attr('d', area(direct)).style('fill', 'var(--s2)').style('opacity', 0.22);
      f.g.append('path').attr('d', line(direct)).style('fill', 'none').style('stroke', 'var(--s1)').style('stroke-width', 2);
      const p = f.g.append('path').attr('d', line(composed)).style('fill', 'none').style('stroke', 'var(--s2)')
        .style('stroke-width', 2).style('stroke-dasharray', '6 4');
      if (!reduceMotion) { const L = p.node().getTotalLength(); p.style('stroke-dasharray', `${L} ${L}`).style('stroke-dashoffset', L)
        .transition().duration(1100).ease(d3.easeCubicOut).style('stroke-dashoffset', 0).on('end', () => p.style('stroke-dasharray', '6 4')); }
      const cross = f.g.append('line').attr('class', 'crosshair').attr('y1', 0).attr('y2', f.h).style('opacity', 0);
      f.g.append('rect').attr('width', f.w).attr('height', f.h).style('fill', 'transparent')
        .on('pointermove', evt => {
          const i = Math.round(Math.max(0, Math.min(T - 1, x.invert(d3.pointer(evt)[0]) / BIN)));
          cross.attr('x1', x(tMin[i])).attr('x2', x(tMin[i])).style('opacity', 1);
          showTip(evt, `${fmt2(tMin[i])} min in ${names[a]}`, [
            { label: `direct ${names[a]}→${names[c]}`, color: 'var(--s1)', value: fmtMin(ok(direct[i]) ? direct[i] * BIN : null) },
            { label: `via ${names[b]}`, color: 'var(--s2)', value: fmtMin(ok(composed[i]) ? composed[i] * BIN : null) },
            { label: 'disagreement', value: ok(direct[i]) && ok(composed[i]) ? `${fmt3(Math.abs(direct[i] - composed[i]) * BIN)} min` : '—' }]);
        }).on('pointerleave', () => { cross.style('opacity', 0); hideTip(); });
      legend('cyc-legend', [{ label: `direct ${names[a]} → ${names[c]}`, color: 'var(--s1)' },
        { label: `composed via ${names[b]}`, color: 'var(--s2)', dash: true }, { label: 'disagreement', color: 'var(--s2)', kind: 'area' }]);
    }
    function drawCycGraph() {
      const f = frame('cyc-graph', 250, { top: 10, right: 10, bottom: 10, left: 10 });
      const R = Math.min(f.w, f.h) / 2 - 26, cx = f.w / 2, cy = f.h / 2 + 4;
      const pos = names.map((n, k) => [cx + R * Math.cos(-Math.PI / 2 + k * 2 * Math.PI / K), cy + R * Math.sin(-Math.PI / 2 + k * 2 * Math.PI / K)]);
      const maxN = d3.max(D.n_pair.flat());
      const wScale = d3.scaleSqrt().domain([0, maxN]).range([0.5, 9]);
      const onRoute = (i, j) => [[route[0], route[1]], [route[1], route[2]], [route[0], route[2]]].some(([p, q]) => (p === i && q === j) || (p === j && q === i));
      for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
        f.g.append('line').attr('x1', pos[i][0]).attr('y1', pos[i][1]).attr('x2', pos[j][0]).attr('y2', pos[j][1])
          .style('stroke', onRoute(i, j) ? 'var(--ink)' : 'var(--axis)').style('stroke-width', wScale(D.n_pair[i][j]))
          .style('stroke-linecap', 'round').style('opacity', onRoute(i, j) ? 0.85 : 0.6)
          .on('pointermove', evt => showTip(evt, pairLabel(i, j), [{ label: 'direct anchors', value: fmtInt(D.n_pair[i][j]) }]))
          .on('pointerleave', hideTip);
      }
      const node = f.g.selectAll('g.node').data(names).join('g').attr('class', 'node')
        .attr('transform', (d, k) => `translate(${pos[k]})`).style('cursor', 'pointer').attr('tabindex', 0)
        .attr('role', 'button').attr('aria-label', d => `Add ${d} to the route`);
      node.append('circle').attr('r', 17).style('fill', 'var(--surface)').style('stroke', (d, k) => series(k)).style('stroke-width', 3);
      node.append('text').attr('text-anchor', 'middle').attr('dy', '0.35em').style('font', '700 11px var(--sans)')
        .style('fill', 'var(--ink)').text((d, k) => { const p = (pending.length ? pending : route).indexOf(k); return p >= 0 ? 'ABC'[p] : ''; });
      node.append('text').attr('class', 'direct-label').attr('text-anchor', 'middle')
        .attr('dy', (d, k) => (pos[k][1] < cy ? -24 : 32)).text(d => d);
      const pick = k => {
        if (pending.includes(k)) return;
        pending.push(k);
        if (pending.length === 3) { route = pending; pending = []; drawCycChart(); drawCycBars(); }
        drawCycGraph();
      };
      node.on('click', (evt, d) => pick(names.indexOf(d))).on('keydown', (evt, d) => { if (evt.key === 'Enter' || evt.key === ' ') { evt.preventDefault(); pick(names.indexOf(d)); } });
    }
    function drawCycBars() {
      const f = frame('cyc-bars', 190, { top: 8, right: 8, bottom: 22, left: 42 });
      const data = cyc.slice().sort((p, q) => q.pairwise - p.pairwise);
      const x = d3.scaleBand().domain(d3.range(data.length)).range([0, f.w]).padding(0.18);
      const y = d3.scaleLinear().domain([0, d3.max(data, d => d.pairwise) * 1.08]).nice().range([f.h, 0]);
      f.g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(4).tickSize(-f.w).tickFormat(''));
      f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickFormat(d => `${d}`)).call(s => s.select('.domain').remove());
      f.g.append('text').attr('class', 'axis-title').attr('x', 0).attr('y', f.h + 16).text('routes, worst first →');
      const same = d => d.t.join() === route.join();
      f.g.selectAll('rect.b').data(data).join('rect').attr('class', 'b').attr('x', (d, i) => x(i)).attr('width', x.bandwidth())
        .attr('y', f.h).attr('height', 0).attr('rx', Math.min(2, x.bandwidth() / 2))
        .style('fill', d => (same(d) ? 'var(--ink)' : 'var(--s2)')).style('cursor', 'pointer')
        .on('pointermove', (evt, d) => showTip(evt, d.t.map(k => names[k]).join(' → '), [
          { label: 'pairwise', color: 'var(--s2)', value: `${fmt3(d.pairwise)} min` }, { label: 'joint', color: 'var(--s1)', value: `${fmt3(d.joint)} min` }]))
        .on('pointerleave', hideTip)
        .on('click', (evt, d) => { route = d.t.slice(); pending = []; drawCycChart(); drawCycGraph(); drawCycBars(); })
        .transition().duration(DUR).delay((d, i) => (reduceMotion ? 0 : i * 8)).attr('y', d => y(d.pairwise)).attr('height', d => f.h - y(d.pairwise));
      f.g.append('path').attr('d', d3.line().x((d, i) => x(i) + x.bandwidth() / 2).y(d => y(d.joint))(data))
        .style('fill', 'none').style('stroke', 'var(--s1)').style('stroke-width', 2);
      legend('cyc-bars-legend', [{ label: 'pairwise warps', color: 'var(--s2)', kind: 'area' }, { label: 'joint warps (≈ 0)', color: 'var(--s1)' }]);
    }
    segGroup('[data-cyc]', 'cyc', v => { cycModel = v; drawCycChart(); });
    const drawCyc = () => { drawCycChart(); drawCycGraph(); drawCycBars(); };
    drawCyc(); renderers.push(drawCyc);

    // ══ 03 Method: λ sweep ══════════════════════════════════════════════════
    const sweep = D.lambda_sweep;
    const selIdx = sweep.findIndex(s => s.lam === D.meta.lam);
    const slider = document.getElementById('lam-slider');
    slider.max = String(sweep.length - 1); slider.value = String(selIdx);
    let lamIdx = selIdx, driftPaths = null;
    function drawDrift() {
      const f = frame('drift-chart', 320, { top: 12, right: 78, bottom: 36, left: 46 });
      const all = sweep.flatMap(s => s.drift.flat());
      const x = d3.scaleLinear().domain([0, tMax]).range([0, f.w]);
      const y = d3.scaleLinear().domain([d3.min(all) * BIN - 0.3, d3.max(all) * BIN + 0.3]).nice().range([f.h, 0]);
      axes(f.g, x, y, f.w, f.h, { xTitle: 'retention time in the batch (min)', yTitle: 'drift into the shared clock (min)' });
      const line = arr => d3.line().x((d, i) => x(tMin[i])).y(d => y(d * BIN)).curve(d3.curveMonotoneX)(arr);
      driftPaths = f.g.selectAll('path.dr').data(sweep[lamIdx].drift).join('path').attr('class', 'dr')
        .attr('d', line).style('fill', 'none').style('stroke', (d, k) => series(k)).style('stroke-width', 2);
      const labelY = idx => dodge(sweep[idx].drift.map(d => y(d[T - 1] * BIN)));
      const labels = f.g.selectAll('text.dl').data(labelY(lamIdx)).join('text').attr('class', 'direct-label dl')
        .attr('x', f.w + 6).attr('y', d => d).attr('dy', '0.35em').style('fill', (d, k) => series(k)).text((d, k) => names[k]);
      const cross = f.g.append('line').attr('class', 'crosshair').attr('y1', 0).attr('y2', f.h).style('opacity', 0);
      f.g.append('rect').attr('width', f.w).attr('height', f.h).style('fill', 'transparent')
        .on('pointermove', evt => {
          const i = Math.round(Math.max(0, Math.min(T - 1, x.invert(d3.pointer(evt)[0]) / BIN)));
          cross.attr('x1', x(tMin[i])).attr('x2', x(tMin[i])).style('opacity', 1);
          showTip(evt, `${fmt2(tMin[i])} min, λ = ${sweep[lamIdx].lam}`, names.map((n, k) => ({ label: n, color: series(k), value: `${fmt2(sweep[lamIdx].drift[k][i] * BIN)} min` })));
        }).on('pointerleave', () => { cross.style('opacity', 0); hideTip(); });
      driftPaths.update = idx => {
        driftPaths.data(sweep[idx].drift).transition().duration(DUR / 1.5).attr('d', line);
        labels.data(labelY(idx)).transition().duration(DUR / 1.5).attr('y', d => d);
      };
    }
    let cvMarker = null;
    function drawCV() {
      const f = frame('cv-chart', 210, { top: 14, right: 16, bottom: 36, left: 48 });
      const x = d3.scaleLog().domain(d3.extent(sweep, s => s.lam)).range([0, f.w]);
      const y = d3.scaleLinear().domain([0, d3.max(sweep, s => s.cv_min) * 1.15]).nice().range([f.h, 0]);
      f.g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(4).tickSize(-f.w).tickFormat(''));
      f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`).call(d3.axisBottom(x).tickValues(sweep.map(s => s.lam)).tickFormat(d3.format('~g')));
      f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4)).call(s => s.select('.domain').remove());
      f.g.append('text').attr('class', 'axis-title').attr('x', f.w).attr('y', f.h + 30).attr('text-anchor', 'end').text('λ (log scale)');
      f.g.append('path').attr('d', d3.line().x(s => x(s.lam)).y(s => y(s.cv_min))(sweep)).style('fill', 'none').style('stroke', 'var(--s1)').style('stroke-width', 2);
      f.g.selectAll('circle.pt').data(sweep).join('circle').attr('class', 'pt').attr('cx', s => x(s.lam)).attr('cy', s => y(s.cv_min)).attr('r', 4)
        .style('fill', 'var(--s1)').style('stroke', 'var(--surface)').style('stroke-width', 2);
      const best = sweep[selIdx];
      f.g.append('text').attr('class', 'direct-label').attr('x', x(best.lam)).attr('y', y(best.cv_min) + 22).attr('text-anchor', 'middle').text('selected');
      cvMarker = f.g.append('circle').attr('r', 9).style('fill', 'none').style('stroke', 'var(--ink)').style('stroke-width', 2);
      cvMarker.move = idx => cvMarker.transition().duration(DUR / 2).attr('cx', x(sweep[idx].lam)).attr('cy', y(sweep[idx].cv_min));
      cvMarker.attr('cx', x(sweep[lamIdx].lam)).attr('cy', y(sweep[lamIdx].cv_min));
      f.g.selectAll('rect.hit').data(sweep).join('rect').attr('class', 'hit').attr('x', s => x(s.lam) - 14).attr('width', 28).attr('y', 0).attr('height', f.h)
        .style('fill', 'transparent').style('cursor', 'pointer')
        .on('pointermove', (evt, s) => showTip(evt, `λ = ${s.lam}`, [{ label: 'held-out error', value: `${fmt3(s.cv_min)} min` }]))
        .on('pointerleave', hideTip).on('click', (evt, s) => { slider.value = String(sweep.indexOf(s)); slider.dispatchEvent(new Event('input')); });
    }
    function setLam(idx) {
      lamIdx = idx; const s = sweep[idx];
      document.getElementById('lam-val').textContent = `${s.lam}${idx === selIdx ? ' (selected)' : ''}`;
      document.getElementById('lam-cv').textContent = fmt3(s.cv_min);
      document.getElementById('lam-dr').textContent = fmt3(s.ticr_drifted);
      document.getElementById('lam-na').textContent = fmt3(s.ticr_near);
      if (driftPaths) driftPaths.update(idx); if (cvMarker) cvMarker.move(idx);
    }
    slider.addEventListener('input', () => setLam(+slider.value));
    const drawMethod = () => { drawDrift(); drawCV(); };
    drawMethod(); renderers.push(drawMethod); setLam(lamIdx);
    legend('drift-legend', names.map((n, k) => ({ label: n, color: series(k) })));

    // ══ 04 Anchors ══════════════════════════════════════════════════════════
    const pairsAB = [];
    for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) pairsAB.push([a, b]);
    let ancPair = pairsAB.reduce((p, q) => (D.anchors[`${q[0]}-${q[1]}`].n > D.anchors[`${p[0]}-${p[1]}`].n ? q : p));
    const drifted30 = D.ticr30.pairs.filter(p => p.drifted);
    if (drifted30.length) { const p = drifted30[0]; ancPair = [Math.min(p.a, p.b), Math.max(p.a, p.b)]; }
    function drawAnchors() {
      const [a, b] = ancPair, key = `${a}-${b}`, A = D.anchors[key];
      const f = frame('anc-chart', 400, { top: 14, right: 18, bottom: 38, left: 52 });
      const x = d3.scaleLinear().domain([0, tMax]).range([0, f.w]);
      const pts = A.ta.map((ta, i) => ({ ta: ta * BIN, d: (A.tb[i] - ta) * BIN, w: A.w[i] }));
      const joint = scaleArr(D.warps.joint[key]), pw = scaleArr(D.warps.pairwise[key]);
      const band = D.band[key];
      const kept = pts.filter(p => p.w > 0).map(p => p.d);
      const lo = Math.min(d3.quantile(kept.sort(d3.ascending), 0.01), d3.min(joint)) - 0.8;
      const hi = Math.max(d3.quantile(kept, 0.99), d3.max(joint)) + 0.8;
      const y = d3.scaleLinear().domain([lo, hi]).nice().range([f.h, 0]);
      axes(f.g, x, y, f.w, f.h, { xTitle: `retention time in ${names[a]} (min)`, yTitle: `shift ${names[a]} → ${names[b]} (min)` });
      f.svg.append('defs').append('clipPath').attr('id', 'anc-clip').append('rect').attr('width', f.w).attr('height', f.h);
      const plot = f.g.append('g').attr('clip-path', 'url(#anc-clip)');
      if (document.getElementById('anc-band').checked)
        plot.append('path').attr('d', d3.area().defined((v, i) => ok(band[0][i]) && ok(band[1][i])).x((v, i) => x(tMin[i]))
          .y0((v, i) => y(band[0][i] * BIN)).y1((v, i) => y(band[1][i] * BIN))(joint))
          .style('fill', 'var(--s1)').style('opacity', 0.2);
      const shown = pts.filter(p => p.d >= y.domain()[0] && p.d <= y.domain()[1]);
      plot.selectAll('circle').data(shown).join('circle').attr('cx', p => x(p.ta)).attr('cy', p => y(p.d)).attr('r', 0)
        .style('fill', p => (p.w > 0 ? 'var(--s1)' : 'none')).style('fill-opacity', 0.28)
        .style('stroke', p => (p.w > 0 ? 'none' : 'var(--muted)')).style('stroke-width', 1.2)
        .transition().duration(DUR).delay(() => (reduceMotion ? 0 : Math.random() * 500)).attr('r', p => (p.w > 0 ? 2.6 : 3));
      const line = arr => d3.line().defined(ok).x((v, i) => x(tMin[i])).y(v => y(v)).curve(d3.curveMonotoneX)(arr);
      if (document.getElementById('anc-pw').checked)
        plot.append('path').attr('d', line(pw)).style('fill', 'none').style('stroke', 'var(--s2)').style('stroke-width', 2).style('stroke-dasharray', '6 4');
      const jp = plot.append('path').attr('d', line(joint)).style('fill', 'none').style('stroke', 'var(--ink)').style('stroke-width', 2.4);
      if (!reduceMotion) { const L = jp.node().getTotalLength(); jp.style('stroke-dasharray', `${L} ${L}`).style('stroke-dashoffset', L).transition().duration(1200).style('stroke-dashoffset', 0); }
      const outside = pts.length - shown.length;
      if (outside) f.g.append('text').attr('class', 'axis-title').attr('x', f.w).attr('y', -2).attr('text-anchor', 'end').text(`${outside} far-off mismatches outside the view`);
      const delaunay = d3.Delaunay.from(shown, p => x(p.ta), p => y(p.d));
      const ring = f.g.append('circle').attr('r', 6).style('fill', 'none').style('stroke', 'var(--ink)').style('stroke-width', 1.5).style('opacity', 0);
      f.g.append('rect').attr('width', f.w).attr('height', f.h).style('fill', 'transparent')
        .on('pointermove', evt => {
          const [mx, my] = d3.pointer(evt); const i = delaunay.find(mx, my); const p = shown[i];
          if (!p || Math.hypot(x(p.ta) - mx, y(p.d) - my) > 24) {
            ring.style('opacity', 0);
            const ti = Math.round(Math.max(0, Math.min(T - 1, x.invert(mx) / BIN)));
            showTip(evt, `${fmt2(tMin[ti])} min in ${names[a]}`, [{ label: 'joint shift', color: 'var(--ink)', value: fmtMin(joint[ti]) },
              { label: 'pairwise shift', color: 'var(--s2)', value: fmtMin(pw[ti]) },
              { label: '90 % band', value: ok(band[0][ti]) ? `${fmt2(band[0][ti] * BIN)} to ${fmt2(band[1][ti] * BIN)}` : '—' }]);
            return;
          }
          ring.attr('cx', x(p.ta)).attr('cy', y(p.d)).style('opacity', 1);
          showTip(evt, p.w > 0 ? 'anchor (used)' : 'anchor (rejected)', [
            { label: `in ${names[a]}`, value: `${fmt2(p.ta)} min` }, { label: `in ${names[b]}`, value: `${fmt2(p.ta + p.d)} min` },
            { label: 'robust weight', value: fmt2(p.w) }]);
        }).on('pointerleave', () => { ring.style('opacity', 0); hideTip(); });
      document.getElementById('anc-n').textContent = fmtInt(A.n);
      document.getElementById('anc-rej').textContent = `${(A.rejected * 100).toFixed(1)} %`;
      document.getElementById('anc-shift').textContent = fmt2(d3.median(kept));
      legend('anc-legend', [{ label: 'anchor, used', color: 'var(--s1)', kind: 'dot' }, { label: 'anchor, rejected (weight 0)', kind: 'ring' },
        { label: 'joint warp', color: 'var(--ink)' }, { label: 'pairwise warp', color: 'var(--s2)', dash: true }, { label: '90 % band', color: 'var(--s1)', kind: 'area' }]);
    }
    chips('anc-pairs', pairsAB.map(([a, b]) => ({ key: `${a}-${b}`, label: pairLabel(a, b) })), `${ancPair[0]}-${ancPair[1]}`,
      key => { ancPair = key.split('-').map(Number); drawAnchors(); });
    ['anc-pw', 'anc-band'].forEach(id => document.getElementById(id).addEventListener('change', drawAnchors));
    drawAnchors(); renderers.push(drawAnchors);

    // ══ 05 Results ══════════════════════════════════════════════════════════
    const R = D.ticr30;
    const groupMean = (rows, m) => d3.mean(rows, p => p.mean[m]);
    const dr = R.pairs.filter(p => p.drifted), na = R.pairs.filter(p => !p.drifted);
    const others = R.methods.filter(m => !m.startsWith('Joint') && m !== 'Unaligned');
    const bestOther = others.reduce((p, q) => (groupMean(dr, q) > groupMean(dr, p) ? q : p));
    document.getElementById('res-joint-dr').textContent = fmt3(groupMean(dr, 'Joint'));
    document.getElementById('res-best-other').textContent = fmt3(groupMean(dr, bestOther));
    document.getElementById('res-best-other-k').textContent = `${bestOther}, drifted pairs`;
    document.getElementById('res-joint-na').textContent = fmt3(groupMean(na, 'Joint'));
    document.getElementById('res-unal-na').textContent = fmt3(groupMean(na, 'Unaligned'));

    function heatmap(containerId, rows, methods, getVal, asTable, groupOf) {
      const el = document.getElementById(containerId); el.replaceChildren();
      const ramp = d3.interpolateRgbBasis(['--seq-0', '--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5'].map(cssVar));
      const color = d3.scaleSequential(ramp).domain([0, 0.92]).clamp(true);
      const table = document.createElement('table'); table.className = asTable ? 'data' : 'heat';
      table.style.width = '100%'; if (!asTable) { table.style.borderSpacing = '0'; table.style.tableLayout = 'fixed'; }
      const thead = table.createTHead().insertRow();
      const th0 = document.createElement('th'); th0.textContent = 'A × B'; th0.className = 'row'; thead.append(th0);
      methods.forEach(m => { const th = document.createElement('th'); th.textContent = m; thead.append(th); });
      const body = table.createTBody();
      let lastGroup = null;
      rows.forEach(p => {
        const g = groupOf(p);
        if (g !== lastGroup) {
          const tr = body.insertRow(); const td = tr.insertCell(); td.colSpan = methods.length + 1;
          const tag = document.createElement('div'); tag.className = 'group-tag'; tag.textContent = g; tag.style.padding = '12px 0 4px';
          td.append(tag); lastGroup = g;
        }
        const tr = body.insertRow();
        const th = document.createElement('th'); th.className = 'row'; th.textContent = pairLabel(p.a, p.b); tr.append(th);
        const vals = methods.map(m => getVal(p, m));
        const best = d3.max(vals.filter(v => v != null));
        methods.forEach((m, j) => {
          const v = vals[j]; const td = tr.insertCell();
          if (asTable) { td.textContent = v == null ? '—' : fmt3(v); if (v === best) td.style.fontWeight = '700'; return; }
          td.className = 'cell' + (v === best ? ' best' : '');
          const d = document.createElement('div'); d.textContent = v == null ? '—' : fmt3(v);
          if (v != null) { const c = color(v); d.style.background = c; d.style.color = inkOn(c); }
          d.addEventListener('pointermove', evt => showTip(evt, pairLabel(p.a, p.b), [{ label: m, value: v == null ? 'n/a' : fmt3(v) }]));
          d.addEventListener('pointerleave', hideTip);
          td.append(d);
        });
      });
      el.append(table);
    }
    let heatAsTable = false;
    const groupDr = p => (p.drifted ? 'Drifted pairs (unaligned r < 0.5)' : 'Near-aligned pairs');
    const orderedRows = R.pairs.slice().sort((p, q) => Number(q.drifted) - Number(p.drifted));
    const drawHeat = () => heatmap('heat', orderedRows, R.methods, (p, m) => p.mean[m], heatAsTable, groupDr);
    document.getElementById('heat-table-btn').addEventListener('click', e => {
      heatAsTable = !heatAsTable; e.currentTarget.setAttribute('aria-pressed', String(heatAsTable));
      e.currentTarget.textContent = heatAsTable ? 'Show as heatmap' : 'Show as table'; drawHeat();
    });
    drawHeat(); renderers.push(drawHeat);

    const selA = document.getElementById('cmp-a'), selB = document.getElementById('cmp-b');
    R.methods.forEach(m => { selA.add(new Option(m, m)); selB.add(new Option(m, m)); });
    selA.value = 'Joint'; selB.value = bestOther;
    function drawCompare() {
      const mA = selA.value, mB = selB.value;
      const f = frame('cmp-chart', 330, { top: 14, right: 12, bottom: 62, left: 44 });
      const x = d3.scaleBand().domain(orderedRows.map((p, i) => i)).range([0, f.w]).paddingInner(0.25).paddingOuter(0.1);
      const all = orderedRows.flatMap(p => p.per_a[mA].concat(p.per_a[mB]));
      const y = d3.scaleLinear().domain([Math.min(0, d3.min(all)), 1]).nice().range([f.h, 0]);
      f.g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(5).tickSize(-f.w).tickFormat(''));
      f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5)).call(s => s.select('.domain').remove());
      f.g.append('text').attr('class', 'axis-title').attr('x', 0).attr('y', -2).text('TIC r per batch-A sample');
      orderedRows.forEach((p, i) => {
        const g = f.g.append('g').attr('transform', `translate(${x(i)},0)`);
        const bw = x.bandwidth(), xa = bw * 0.25, xb = bw * 0.75;
        g.append('rect').attr('width', bw).attr('height', f.h).attr('rx', 6).style('fill', 'var(--surface-2)').style('opacity', p.drifted ? 0.9 : 0.4);
        const va = p.per_a[mA], vb = p.per_a[mB];
        va.forEach((v, s) => g.append('line').attr('x1', xa).attr('x2', xb).attr('y1', y(v)).attr('y2', y(vb[s]))
          .style('stroke', vb[s] >= v ? 'var(--s2)' : 'var(--s1)').style('opacity', 0.35).style('stroke-width', 1));
        const dots = (vals, cx, col, m) => g.selectAll(null).data(vals).join('circle').attr('cx', cx).attr('cy', f.h).attr('r', 3.5)
          .style('fill', col).style('stroke', 'var(--surface)').style('stroke-width', 1.5)
          .on('pointermove', (evt, v) => showTip(evt, pairLabel(p.a, p.b), [{ label: m, color: col, value: fmt3(v) }]))
          .on('pointerleave', hideTip)
          .transition().duration(DUR).delay(reduceMotion ? 0 : i * 40).attr('cy', v => y(v));
        dots(va, xa, 'var(--s1)', mA); dots(vb, xb, 'var(--s2)', mB);
        [[xa, d3.mean(va)], [xb, d3.mean(vb)]].forEach(([cx, mv]) => g.append('line').attr('x1', cx - 10).attr('x2', cx + 10)
          .attr('y1', y(mv)).attr('y2', y(mv)).style('stroke', 'var(--ink)').style('stroke-width', 2.5).style('stroke-linecap', 'round'));
        g.append('text').attr('class', 'axis-title').attr('x', bw / 2).attr('y', f.h + 16).attr('text-anchor', 'middle').text(names[p.a]);
        g.append('text').attr('class', 'axis-title').attr('x', bw / 2).attr('y', f.h + 30).attr('text-anchor', 'middle').text(`× ${names[p.b]}`);
      });
      legend('cmp-legend', [{ label: mA, color: 'var(--s1)', kind: 'dot' }, { label: mB, color: 'var(--s2)', kind: 'dot' },
        { label: 'pair mean', color: 'var(--ink)' }, { label: `shaded = drifted pair`, color: 'var(--muted)', kind: 'area' }]);
    }
    [selA, selB].forEach(s => s.addEventListener('change', drawCompare));
    drawCompare(); renderers.push(drawCompare);

    // ══ 06 Missing correspondences ══════════════════════════════════════════
    const lopoKeys = Object.keys(D.lopo);
    let lopoKey = lopoKeys[0], lopoOn = false, lopoApi = null;
    const rowFor = (a, b) => R.pairs.find(p => p.a === a && p.b === b);
    function drawLopo() {
      const [a, b] = lopoKey.split('-').map(Number), L = D.lopo[lopoKey];
      const f = frame('lopo-chart', 340, { top: 14, right: 16, bottom: 38, left: 52 });
      const joint = scaleArr(D.warps.joint[lopoKey]), lop = scaleArr(L.map);
      const A = D.anchors[`${Math.min(a, b)}-${Math.max(a, b)}`];
      const pts = A.ta.map((ta, i) => (a < b ? { t: ta, d: A.tb[i] - ta, w: A.w[i] } : { t: A.tb[i], d: ta - A.tb[i], w: A.w[i] }))
        .filter(p => p.w > 0).map(p => ({ t: p.t * BIN, d: p.d * BIN }));
      const x = d3.scaleLinear().domain([0, tMax]).range([0, f.w]);
      const vals = joint.concat(lop).filter(ok);
      const y = d3.scaleLinear().domain([d3.min(vals) - 0.8, d3.max(vals) + 0.8]).nice().range([f.h, 0]);
      axes(f.g, x, y, f.w, f.h, { xTitle: `retention time in ${names[a]} (min)`, yTitle: `shift ${names[a]} → ${names[b]} (min)` });
      const dots = f.g.append('g').selectAll('circle').data(pts.filter(p => p.d >= y.domain()[0] && p.d <= y.domain()[1])).join('circle')
        .attr('cx', p => x(p.t)).attr('cy', p => y(p.d)).attr('r', 2.2).style('fill', 'var(--muted)').style('opacity', lopoOn ? 0.05 : 0.3);
      const line = arr => d3.line().defined(ok).x((v, i) => x(tMin[i])).y(v => y(v)).curve(d3.curveMonotoneX)(arr);
      f.g.append('path').attr('d', line(joint)).style('fill', 'none').style('stroke', 'var(--s1)').style('stroke-width', 2.4);
      const lp = f.g.append('path').attr('d', line(lopoOn ? lop : joint)).style('fill', 'none').style('stroke', 'var(--s2)')
        .style('stroke-width', 2.4).style('stroke-dasharray', '7 4').style('opacity', lopoOn ? 1 : 0);
      const cross = f.g.append('line').attr('class', 'crosshair').attr('y1', 0).attr('y2', f.h).style('opacity', 0);
      f.g.append('rect').attr('width', f.w).attr('height', f.h).style('fill', 'transparent')
        .on('pointermove', evt => {
          const i = Math.round(Math.max(0, Math.min(T - 1, x.invert(d3.pointer(evt)[0]) / BIN)));
          cross.attr('x1', x(tMin[i])).attr('x2', x(tMin[i])).style('opacity', 1);
          showTip(evt, `${fmt2(tMin[i])} min in ${names[a]}`, [{ label: 'all anchors', color: 'var(--s1)', value: fmtMin(joint[i]) },
            { label: 'pair removed', color: 'var(--s2)', value: fmtMin(lop[i]) },
            { label: 'difference', value: ok(lop[i]) && ok(joint[i]) ? `${fmt3(Math.abs(lop[i] - joint[i]))} min` : '—' }]);
        }).on('pointerleave', () => { cross.style('opacity', 0); hideTip(); });
      lopoApi = on => {
        lp.transition().duration(DUR * 1.4).ease(d3.easeCubicInOut).style('opacity', 1).attr('d', line(on ? lop : joint))
          .on('end', function () { if (!on) d3.select(this).style('opacity', 0); });
        dots.transition().duration(DUR).style('opacity', on ? 0.05 : 0.3);
      };
      const r = rowFor(a, b);
      document.getElementById('lopo-r-j').textContent = fmt3(r ? r.mean.Joint : NaN);
      document.getElementById('lopo-r-l').textContent = fmt3(L.ticr);
      document.getElementById('lopo-e-j').textContent = fmt3(L.err_joint_min);
      document.getElementById('lopo-e-l').textContent = fmt3(L.err_min);
      legend('lopo-legend', [{ label: 'joint, all anchors', color: 'var(--s1)' }, { label: "joint, this pair's anchors removed", color: 'var(--s2)', dash: true },
        { label: "the pair's own anchors", color: 'var(--muted)', kind: 'dot' }]);
    }
    function drawLopoAll() {
      const f = frame('lopo-all', 230, { top: 8, right: 14, bottom: 24, left: 118 });
      const rows = lopoKeys.map(k => { const [a, b] = k.split('-').map(Number); const r = rowFor(a, b); return { k, a, b, j: r.mean.Joint, l: D.lopo[k].ticr }; });
      const y = d3.scaleBand().domain(rows.map(r => r.k)).range([0, f.h]).padding(0.3);
      const x = d3.scaleLinear().domain([Math.min(0, d3.min(rows, r => Math.min(r.j, r.l))), 1]).range([0, f.w]);
      f.g.append('rect').attr('x', -f.margin.left).attr('width', f.w + f.margin.left).attr('y', y(lopoKey) - y.step() * y.padding() / 2)
        .attr('height', y.step()).attr('rx', 6).style('fill', 'var(--surface-2)');
      f.g.append('g').attr('class', 'gridline').attr('transform', `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(5).tickSize(-f.h).tickFormat(''));
      f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(5));
      f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).tickFormat(k => { const [a, b] = k.split('-').map(Number); return pairLabel(a, b); }).tickSize(0))
        .call(s => s.select('.domain').remove());
      const g = f.g.selectAll('g.r').data(rows).join('g').attr('transform', r => `translate(0,${y(r.k) + y.bandwidth() / 2})`)
        .style('cursor', 'pointer').on('click', (evt, r) => { lopoKey = r.k; chips('lopo-pairs', lopoChips, lopoKey, onLopoPair); drawLopo(); drawLopoAll(); })
        .on('pointermove', (evt, r) => showTip(evt, pairLabel(r.a, r.b), [{ label: 'all anchors', color: 'var(--s1)', value: fmt3(r.j) }, { label: 'pair removed', color: 'var(--s2)', value: fmt3(r.l) }]))
        .on('pointerleave', hideTip);
      g.append('rect').attr('x', -f.margin.left).attr('y', -y.step() / 2).attr('width', f.w + f.margin.left).attr('height', y.step()).style('fill', 'transparent');
      g.append('line').attr('x1', r => x(Math.min(r.j, r.l))).attr('x2', r => x(Math.max(r.j, r.l))).style('stroke', 'var(--axis)').style('stroke-width', 2);
      g.append('circle').attr('cx', r => x(r.l)).attr('r', 7).style('fill', 'none').style('stroke', 'var(--s2)').style('stroke-width', 2);
      g.append('circle').attr('cx', r => x(r.j)).attr('r', 4).style('fill', 'var(--s1)');
    }
    const lopoChips = lopoKeys.map(k => { const [a, b] = k.split('-').map(Number); return { key: k, label: pairLabel(a, b) }; });
    const onLopoPair = k => { lopoKey = k; drawLopo(); drawLopoAll(); };
    chips('lopo-pairs', lopoChips, lopoKey, onLopoPair);
    document.getElementById('lopo-toggle').addEventListener('click', e => {
      lopoOn = !lopoOn; e.currentTarget.setAttribute('aria-pressed', String(lopoOn));
      e.currentTarget.textContent = lopoOn ? '↺ Restore direct anchors' : '✂ Remove direct anchors'; lopoApi(lopoOn);
    });
    const drawMissing = () => { drawLopo(); drawLopoAll(); };
    drawMissing(); renderers.push(drawMissing);

    // ══ 07 Uncertainty ══════════════════════════════════════════════════════
    function drawBand() {
      const f = frame('band-chart', 300, { top: 8, right: 12, bottom: 36, left: 118 });
      const keys = pairsAB.map(([a, b]) => `${a}-${b}`);
      const step = 2;                                      // bins per cell
      const cols = d3.range(0, T, step);
      const x = d3.scaleLinear().domain([0, tMax]).range([0, f.w]);
      const y = d3.scaleBand().domain(keys).range([0, f.h]).padding(0.08);
      const widths = keys.map(k => cols.map(i => (ok(D.band[k][1][i]) && ok(D.band[k][0][i]) ? (D.band[k][1][i] - D.band[k][0][i]) * BIN : null)));
      const ramp = d3.interpolateRgbBasis(['--seq-0', '--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5'].map(cssVar));
      const color = d3.scaleSequential(ramp).domain([0, d3.quantile(widths.flat().filter(ok).sort(d3.ascending), 0.98)]).clamp(true);
      keys.forEach((k, r) => {
        const [a, b] = k.split('-').map(Number);
        f.g.append('rect').attr('y', y(k)).attr('width', f.w).attr('height', y.bandwidth()).attr('rx', 3).style('fill', 'var(--surface-2)');
        f.g.selectAll(null).data(cols.filter((i, j) => ok(widths[r][j]))).join('rect').attr('x', i => x(i * BIN)).attr('y', y(k)).attr('width', x(step * BIN) + 0.5)
          .attr('height', y.bandwidth()).style('fill', i => color(widths[r][cols.indexOf(i)]))
          .on('pointermove', (evt, i) => showTip(evt, pairLabel(a, b), [{ label: `at ${fmt2(i * BIN)} min`, value: `${fmt3(widths[r][cols.indexOf(i)])} min wide` }]))
          .on('pointerleave', hideTip);
      });
      f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(9))
        .append('text').attr('class', 'axis-title').attr('x', f.w).attr('y', 30).attr('text-anchor', 'end').text('retention time in A (min)');
      f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).tickSize(0).tickFormat(k => { const [a, b] = k.split('-').map(Number); return pairLabel(a, b); }))
        .call(s => s.select('.domain').remove());
      const lg = document.getElementById('band-legend'); lg.replaceChildren();
      const bar = document.createElement('span'); const i = document.createElement('i');
      i.style.width = '120px'; i.style.height = '10px'; i.style.borderRadius = '3px';
      i.style.background = `linear-gradient(90deg, ${d3.range(0, 1.01, 0.2).map(t => color(t * color.domain()[1])).join(',')})`;
      bar.append(document.createTextNode('0 '), i, document.createTextNode(` ${fmt2(color.domain()[1])}+ min band width`)); lg.append(bar);
    }
    function drawCov() {
      const f = frame('cov-chart', 150, { top: 10, right: 40, bottom: 30, left: 150 });
      const rows = [{ k: 'warp band only', v: D.coverage.epistemic }, { k: 'band ⊕ anchor scatter', v: D.coverage.predictive }].filter(r => r.v != null);
      const x = d3.scaleLinear().domain([0, 1]).range([0, f.w]);
      const y = d3.scaleBand().domain(rows.map(r => r.k)).range([0, f.h]).padding(0.35);
      f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(5, '%'));
      f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).tickSize(0)).call(s => s.select('.domain').remove());
      f.g.selectAll('rect.v').data(rows).join('rect').attr('class', 'v').attr('y', r => y(r.k)).attr('height', y.bandwidth()).attr('rx', 4)
        .attr('width', 0).style('fill', (r, i) => (i ? 'var(--s1)' : 'var(--muted)'))
        .on('pointermove', (evt, r) => showTip(evt, r.k, [{ label: 'coverage', value: `${(r.v * 100).toFixed(1)} %` }, { label: 'target', value: '90 %' }]))
        .on('pointerleave', hideTip)
        .transition().duration(DUR * 1.5).attr('width', r => x(r.v));
      const inside = r => r.v > 0.3;
      f.g.selectAll('text.v').data(rows).join('text').attr('class', 'direct-label v')
        .attr('x', r => (inside(r) ? x(r.v) - 8 : x(r.v) + 6)).attr('text-anchor', r => (inside(r) ? 'end' : 'start'))
        .attr('y', r => y(r.k) + y.bandwidth() / 2).attr('dy', '0.35em').style('fill', r => (inside(r) ? '#ffffff' : 'var(--ink-2)'))
        .text(r => `${(r.v * 100).toFixed(0)} %`);
      f.g.append('line').attr('x1', x(0.9)).attr('x2', x(0.9)).attr('y1', -4).attr('y2', f.h).style('stroke', 'var(--ink)').style('stroke-dasharray', '4 3');
      f.g.append('text').attr('class', 'direct-label').attr('x', x(0.9)).attr('y', -6).attr('text-anchor', 'middle').text('target 90 %');
    }
    const drawUnc = () => { drawBand(); drawCov(); };
    drawUnc(); renderers.push(drawUnc);

    // ══ 08 Benchmark ════════════════════════════════════════════════════════
    const B = D.benchmark;
    let proto = 'B-frame', benchTable = false;
    function drawBench() {
      if (!B) {
        document.getElementById('bench-heat').replaceChildren(Object.assign(document.createElement('div'),
          { className: 'pending', textContent: 'The fair benchmark (script 32) has not been exported yet. Run scripts/32_fair_benchmark.py, then scripts/40_export_site_data.py.' }));
        return;
      }
      const pairs = B.pairs.map(p => ({ ...p, drifted: (rowFor(p.a, p.b) || {}).drifted }));
      const group = p => (names[p.a] === 'aug2015' ? 'aug2015 pairs (two samples)' : p.drifted ? 'Drifted pairs' : 'Near-aligned pairs');
      const order = ['Drifted pairs', 'Near-aligned pairs', 'aug2015 pairs (two samples)'];
      const rows = pairs.slice().sort((p, q) => order.indexOf(group(p)) - order.indexOf(group(q)));
      heatmap('bench-heat', rows, B.methods, (p, m) => p.mean[`${m}|${proto}`] ?? null, benchTable, group);
      // group means as small multiples of horizontal bars
      const f = frame('bench-bars', 250, { top: 22, right: 20, bottom: 26, left: 130 });
      const gap = 44, panelW = (f.w - 2 * gap) / 3;
      order.forEach((gname, gi) => {
        const gp = rows.filter(p => group(p) === gname);
        const vals = B.methods.map(m => ({ m, v: d3.mean(gp, p => p.mean[`${m}|${proto}`]) })).filter(d => d.v != null);
        const g = f.g.append('g').attr('transform', `translate(${gi * (panelW + gap)},0)`);
        const x = d3.scaleLinear().domain([Math.min(0, d3.min(vals, d => d.v)), 1]).range([0, panelW]);
        const y = d3.scaleBand().domain(B.methods).range([0, f.h]).padding(0.25);
        g.append('text').attr('class', 'direct-label').attr('y', -8).text(gname);
        g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(4).tickSizeOuter(0));
        if (gi === 0) g.append('g').attr('class', 'axis').call(d3.axisLeft(y).tickSize(0)).call(s => s.select('.domain').remove());
        const best = d3.max(vals, d => d.v);
        g.selectAll('rect').data(vals).join('rect').attr('y', d => y(d.m)).attr('height', y.bandwidth()).attr('x', d => x(Math.min(0, d.v)))
          .attr('rx', 3).attr('width', 0).style('fill', d => (d.m === 'Joint' || d.m === 'Neural joint' ? 'var(--s1)' : 'var(--axis)'))
          .on('pointermove', (evt, d) => showTip(evt, gname, [{ label: d.m, value: fmt3(d.v) }])).on('pointerleave', hideTip)
          .transition().duration(DUR).attr('width', d => Math.abs(x(d.v) - x(0)));
        g.selectAll('text.v').data(vals).join('text').attr('class', 'axis-title v').attr('x', d => x(Math.max(0, d.v)) + 4)
          .attr('y', d => y(d.m) + y.bandwidth() / 2).attr('dy', '0.35em').style('font-weight', d => (d.v === best ? 700 : 400)).text(d => fmt2(d.v));
      });
      document.getElementById('bench-note').textContent = `Bars: mean TIC r per group under the ${proto} protocol; blue = this project's models. `
        + (B.gcmsalign ? `Benchmark run with ${B.gcmsalign}.` : '');
    }
    segGroup('[data-proto]', 'proto', v => { proto = v; drawBench(); });
    document.getElementById('bench-table-btn').addEventListener('click', e => {
      benchTable = !benchTable; e.currentTarget.setAttribute('aria-pressed', String(benchTable));
      e.currentTarget.textContent = benchTable ? 'Show as heatmap' : 'Show as table'; drawBench();
    });
    drawBench(); renderers.push(drawBench);
  }).catch(err => {
    console.error(err);
    const msg = window.SITE_DATA
      ? `A figure failed to render: ${err && err.message ? err.message : err}. The browser console has details.`
      : 'The page data (docs/data/site.js) is missing. It is written by python scripts/40_export_site_data.py and must be committed with the site.';
    document.querySelector('main').prepend(Object.assign(document.createElement('div'), { className: 'pending', textContent: msg }));
  });

  // ── Page chrome: theme, nav, reveal, progress, copy ─────────────────────
  document.getElementById('theme-btn').addEventListener('click', () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('mba-theme', root.dataset.theme); } catch (e) { /* storage unavailable */ }
    renderers.forEach(r => r());
  });
  const revealObs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); revealObs.unobserve(e.target); } }), { threshold: 0.12 });
  document.querySelectorAll('.reveal').forEach(el => revealObs.observe(el));
  const links = [...document.querySelectorAll('.nav-links a')];
  const secObs = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${e.target.id}`));
  }), { rootMargin: '-45% 0px -50% 0px' });
  document.querySelectorAll('section.chapter').forEach(s => secObs.observe(s));
  const prog = document.getElementById('progress');
  addEventListener('scroll', () => {
    const h = document.documentElement; prog.style.width = `${(h.scrollTop / (h.scrollHeight - h.clientHeight)) * 100}%`;
  }, { passive: true });
  let rz; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => renderers.forEach(r => r()), 180); });
  document.getElementById('copy-btn').addEventListener('click', e => {
    navigator.clipboard?.writeText(document.getElementById('code').textContent).then(() => {
      e.target.textContent = 'Copied'; setTimeout(() => { e.target.textContent = 'Copy'; }, 1400);
    });
  });
})();
