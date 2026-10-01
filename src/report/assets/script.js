
(() => {
  // seletores: gravam a escolha no data-* do alvo; o CSS mostra/esconde
  document.querySelectorAll('.seletor').forEach(g => {
    g.addEventListener('click', ev => {
      const b = ev.target.closest('button'); if (!b) return;
      const alvo = document.getElementById(g.dataset.alvo);
      alvo.dataset[g.dataset.chave] = b.dataset.valor;
      g.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b));
    });
  });

  const dados = JSON.parse(document.getElementById('dados-curvas').textContent);
  const pct = v => (v * 100).toFixed(1).replace('.', ',') + '%';
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, at, pai) => { const e = document.createElementNS(NS, tag);
    for (const k in at) e.setAttribute(k, at[k]); if (pai) pai.appendChild(e); return e; };
  const series = [['fwd', 'Img→cér', 'var(--s1)'], ['bwd', 'Cér→img', 'var(--s2)']];

  function passo(n) { // divisao limpa do eixo x
    for (const p of [1, 2, 5, 10, 20, 25, 50]) if (n / p <= 6) return p;
    return 100;
  }

  function desenha(caixa, d) {
    caixa.textContent = '';
    const W = caixa.clientWidth, H = caixa.clientHeight;
    if (W < 50) return;
    const m = {t: 10, r: 96, b: 26, l: 40};
    const e0 = d.epocas[0], e1 = d.epocas[d.epocas.length - 1];
    const x = e => m.l + (e1 === e0 ? 0 : (e - e0) / (e1 - e0)) * (W - m.l - m.r);
    const y = v => m.t + (1 - v) * (H - m.t - m.b);
    const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', tabindex: 0,
      'aria-label': 'Retrieval top-1 no teste por época'}, caixa);
    for (const v of [0, .25, .5, .75, 1]) {
      el('line', {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: 'var(--rule)',
        'stroke-width': 1}, svg);
      const t = el('text', {x: m.l - 7, y: y(v) + 4, 'text-anchor': 'end'}, svg);
      t.textContent = (v * 100) + '%';
    }
    const p = passo(e1 - e0 + 1);
    for (let e = Math.ceil((e0 + 1) / p) * p - 1; e <= e1; e += p) {
      if (e < e0) continue;
      const t = el('text', {x: x(e), y: H - 7, 'text-anchor': 'middle'}, svg);
      t.textContent = e + 1;
    }
    for (const [k, , cor] of series) {
      el('polyline', {points: d.epocas.map((e, i) => `${x(e)},${y(d[k][i])}`).join(' '),
        fill: 'none', stroke: cor, 'stroke-width': 2, 'stroke-linejoin': 'round',
        'stroke-linecap': 'round'}, svg);
    }
    // rotulo so no fim da linha; se colidirem, afasta com uma perna curta
    const fim = series.map(([k, nome, cor]) => ({k, nome, cor, v: d[k][d[k].length - 1]}));
    const yl = fim.map(f => y(f.v));
    if (Math.abs(yl[0] - yl[1]) < 14) {
      const meio = (yl[0] + yl[1]) / 2, s = yl[0] <= yl[1] ? -1 : 1;
      yl[0] = meio + s * 8; yl[1] = meio - s * 8;
    }
    fim.forEach((f, i) => {
      const cx = x(e1), cy = y(f.v);
      if (Math.abs(yl[i] - cy) > 1)
        el('line', {x1: cx + 6, y1: cy, x2: cx + 12, y2: yl[i], stroke: 'var(--muted)',
          'stroke-width': 1}, svg);
      el('circle', {cx, cy, r: 4, fill: f.cor, stroke: 'var(--surface)', 'stroke-width': 2}, svg);
      const t = el('text', {x: cx + 14, y: yl[i] + 4, class: 'fim'}, svg);
      t.textContent = `${f.nome} ${pct(f.v)}`;
    });
    // camada de leitura: crosshair que encaixa na epoca mais proxima
    const cruz = el('line', {y1: m.t, y2: H - m.b, stroke: 'var(--muted)', 'stroke-width': 1,
      visibility: 'hidden'}, svg);
    const pts = series.map(([, , cor]) => el('circle', {r: 4, fill: cor, stroke: 'var(--surface)',
      'stroke-width': 2, visibility: 'hidden'}, svg));
    const dica = document.createElement('div'); dica.className = 'dica'; dica.hidden = true;
    caixa.appendChild(dica);
    let atual = d.epocas.length - 1;
    function mostra(i) {
      atual = Math.max(0, Math.min(d.epocas.length - 1, i));
      const e = d.epocas[atual], cx = x(e);
      cruz.setAttribute('x1', cx); cruz.setAttribute('x2', cx); cruz.setAttribute('visibility', 'visible');
      series.forEach(([k], j) => { pts[j].setAttribute('cx', cx); pts[j].setAttribute('cy', y(d[k][atual]));
        pts[j].setAttribute('visibility', 'visible'); });
      dica.textContent = '';
      const ep = document.createElement('span'); ep.className = 'ep';
      ep.textContent = `época ${e + 1} de ${e1 + 1}`; dica.appendChild(ep);
      for (const [k, nome, cor] of series) {
        const linha = document.createElement('div');
        const c = document.createElement('span'); c.className = 'chave'; c.style.borderColor = cor;
        const b = document.createElement('b'); b.textContent = pct(d[k][atual]);
        const r = document.createElement('span'); r.className = 'rot'; r.textContent = nome;
        linha.append(c, b, r); dica.appendChild(linha);
      }
      dica.hidden = false;
      const esq = cx + 12 + dica.offsetWidth > W ? cx - 12 - dica.offsetWidth : cx + 12;
      dica.style.left = esq + 'px'; dica.style.top = m.t + 'px';
    }
    function esconde() { cruz.setAttribute('visibility', 'hidden');
      pts.forEach(p => p.setAttribute('visibility', 'hidden')); dica.hidden = true; }
    svg.addEventListener('pointermove', ev => {
      const r = svg.getBoundingClientRect(), px = ev.clientX - r.left;
      let melhor = 0, dist = Infinity;
      d.epocas.forEach((e, i) => { const dd = Math.abs(x(e) - px); if (dd < dist) { dist = dd; melhor = i; } });
      mostra(melhor);
    });
    svg.addEventListener('pointerleave', esconde);
    svg.addEventListener('focus', () => mostra(atual));
    svg.addEventListener('blur', esconde);
    svg.addEventListener('keydown', ev => {
      if (ev.key === 'ArrowLeft') { mostra(atual - 1); ev.preventDefault(); }
      if (ev.key === 'ArrowRight') { mostra(atual + 1); ev.preventDefault(); }
    });
  }

  document.querySelectorAll('.curva .grafico').forEach(caixa => {
    const d = dados[+caixa.dataset.i];
    const redesenha = () => desenha(caixa, d);
    redesenha();
    if ('ResizeObserver' in window) new ResizeObserver(redesenha).observe(caixa);
  });
})();
