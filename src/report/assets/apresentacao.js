/* A apresentação (results/site/apresentacao.html): navegação entre slides, escala para a janela, as
   imagens recortadas das tiras da galeria e dos exemplos, os gráficos e os slides interativos do
   controle semântico. Os dados vêm de report/apresentacao.py, no bloco JSON #dados; a regra
   exclusiva e a decodificação das frações são as do simulador.js (o mesmo do site). */
(function () {
  'use strict';
  var D = JSON.parse(document.getElementById('dados').textContent);
  var raiz = document.documentElement;
  var NS = 'http://www.w3.org/2000/svg';
  var CHAVE_TEMA = 'benchmark-modelos/site/v1/tema';     // a mesma do site: o tema escolhido vale nos dois
  var G = D.galeria, E = D.exemplos;
  var DEC = D.semantica && window.Simulador ? Simulador.decodifica(D.semantica) : null;
  var COR = {frr: 'var(--frr)', mindeye2: 'var(--me2)', artigo: 'var(--me2)', mindeye1: 'var(--me1)'};

  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; });
  }
  function virg(x, casas) { return x.toFixed(casas == null ? 1 : casas).replace('.', ','); }
  function milhar(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, '.'); }
  function pct(x, casas) { return x == null ? '—' : virg(100 * x, casas) + '%'; }      // x de 0 a 1
  function pctTela(v) { return (v >= 10 ? virg(v, 0) : v >= 1 ? virg(v, 1) : virg(v, 1)) + '%'; }
  function tempo(s) {
    if (s == null) return '—';
    if (s < 90) return Math.round(s) + ' s';
    var m = Math.round(s / 60), h = Math.floor(m / 60);
    return h ? h + ' h ' + String(m % 60).padStart(2, '0') : m + ' min';
  }
  function svg(tag, at, pai) {
    var e = document.createElementNS(NS, tag);
    for (var k in at) if (at[k] != null) e.setAttribute(k, at[k]);
    if (pai) pai.appendChild(e);
    return e;
  }
  function texto(pai, x, y, t, at) {
    var e = svg('text', Object.assign({x: x, y: y}, at || {}), pai);
    e.textContent = t;
    return e;
  }
  function nome(sup) { return (D.nomes && D.nomes[sup]) || sup; }

  // ------------------------------------------------------------------ tema
  try { var t0 = localStorage.getItem(CHAVE_TEMA); if (t0) raiz.setAttribute('data-theme', t0); } catch (e) { /* sem armazenamento */ }
  function escuro() {
    var t = raiz.getAttribute('data-theme');
    if (t) return t === 'dark';
    return !!(window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches);
  }
  function trocaTema() {
    var novo = escuro() ? 'light' : 'dark';
    raiz.setAttribute('data-theme', novo);
    try { localStorage.setItem(CHAVE_TEMA, novo); } catch (e) { /* sem armazenamento */ }
    rotulaTema();
  }
  function rotulaTema() { var b = $('[data-cmd="tema"]'); if (b) b.textContent = escuro() ? 'Tema claro' : 'Tema escuro'; }

  // ------------------------------------------------------------------ imagens das tiras
  /* 'vista:9', 'modelo:<id>:<tipo>:9' (galeria) ou 'ex:<supercategoria>:<k>' (exemplos) -> a tira e a posição */
  function fonte(spec) {
    var p = spec.split(':');
    if (p[0] === 'vista' && G) return {src: G.vista, n: G.n, k: +p[1], lado: G.lado};
    if (p[0] === 'modelo' && G && G.modelos[p[1]] && G.modelos[p[1]][p[2]]) {
      return {src: G.modelos[p[1]][p[2]], n: G.n, k: +p[3], lado: G.lado};
    }
    if (p[0] === 'ex' && E && E.classes[p[1]]) {
      var c = E.classes[p[1]];
      return {src: c.fotos, mascara: c.mascaras, n: c.exemplos.length, k: +p[2], lado: E.lado};
    }
    return null;
  }
  /* um <div class="tile"> de `tam` px com a imagem (e a camada da máscara, nos exemplos) */
  function tile(spec, tam, el) {
    var f = fonte(spec);
    el = el || document.createElement('div');
    el.className = 'tile';
    el.style.width = el.style.height = tam + 'px';
    if (!f) { el.classList.add('tile-vazio'); el.textContent = '—'; return el; }
    var pos = function (d) {
      d.style.backgroundSize = (f.n * tam) + 'px ' + tam + 'px';
      d.style.backgroundPosition = (-f.k * tam) + 'px 0';
    };
    el.style.backgroundImage = 'url("' + f.src + '")';
    pos(el);
    if (f.mascara) {
      var m = document.createElement('div');
      m.className = 'mascara';
      m.style.backgroundImage = 'url("' + f.mascara + '")';
      pos(m);
      el.appendChild(m);
    }
    return el;
  }
  /* <svg data-tile> dentro de um diagrama: a tira como <image>, recortada pelo viewBox */
  function tileSvg(s) {
    var f = fonte(s.getAttribute('data-tile'));
    var w = +s.getAttribute('width'), h = +s.getAttribute('height');
    if (!f) {
      s.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
      svg('rect', {'class': 'caixa-wash', width: w, height: h}, s);
      return;
    }
    s.setAttribute('viewBox', (f.k * f.lado) + ' 0 ' + f.lado + ' ' + f.lado);
    s.setAttribute('preserveAspectRatio', 'none');
    var im = svg('image', {href: f.src, width: f.n * f.lado, height: f.lado, preserveAspectRatio: 'none'}, s);
    var filtro = s.getAttribute('data-filtro');
    if (filtro) im.setAttribute('filter', 'url(#' + filtro + ')');
  }

  // ------------------------------------------------------------------ dica (tooltip) dentro do slide
  function dica(slide) {
    var d = $('.dica', slide);
    if (!d) { d = document.createElement('div'); d.className = 'dica'; d.hidden = true; slide.appendChild(d); }
    return {
      mostra: function (html, ev) {
        d.innerHTML = html; d.hidden = false;
        var r = slide.getBoundingClientRect(), s = r.width / 1280;
        var x = (ev.clientX - r.left) / s, y = (ev.clientY - r.top) / s;
        var w = d.offsetWidth, h = d.offsetHeight;
        d.style.left = Math.min(1280 - w - 12, x + 16) + 'px';
        d.style.top = Math.max(8, Math.min(720 - h - 8, y - h - 12)) + 'px';
      },
      some: function () { d.hidden = true; }
    };
  }

  // ------------------------------------------------------------------ desenhos pequenos dos diagramas
  function voxels(g) {
    var p = g.getAttribute('data-voxels').split(',').map(Number), cols = p[0], lins = p[1], t = p[2], v = p[3];
    for (var i = 0; i < lins; i++) {
      for (var j = 0; j < cols; j++) {
        var x = 0.5 + 0.32 * Math.sin(0.9 * i + 0.45 * j) * Math.cos(0.55 * j - 0.3 * i) +
          0.18 * Math.sin(12.9898 * i + 78.233 * j);
        var nivel = x < 0.35 ? 1 : x < 0.55 ? 2 : x < 0.75 ? 3 : 4;
        svg('rect', {'class': 'voxel', x: j * (t + v), y: i * (t + v), width: t, height: t, rx: 1.5,
          style: 'fill:var(--seq-' + nivel + ')'}, g);
      }
    }
  }
  function vetor(g) {
    var p = g.getAttribute('data-vetor').split(',').map(Number), n = p[0], larg = p[1];
    for (var i = 0; i < n; i++) {
      var w = larg * (0.2 + 0.8 * Math.abs(Math.sin(i * 2.3 + 1)));
      svg('rect', {'class': 'barra-vetor', x: 0, y: i * 10, width: w.toFixed(1), height: 7, rx: 1.5}, g);
    }
  }
  function sessoes(g) {
    var n = +g.getAttribute('data-sessoes'), w = (240 - (n - 1) * 2) / n;
    for (var i = 0; i < n; i++) {
      svg('rect', {x: (i * (w + 2)).toFixed(2), y: 22, width: w.toFixed(2), height: 36, rx: 1,
        style: 'fill:' + (i === 0 ? 'var(--ink)' : 'var(--rule-2)')}, g);
    }
  }
  function repeticoes(g) {
    var p = g.getAttribute('data-reps').split('x').map(Number), imgs = p[0], reps = p[1];
    var t = 34, x = 0, quais = [0, 3, 9, 10, 11, 18];
    for (var a = 0; a < imgs; a++) {
      for (var r = 0; r < reps; r++) {
        var s = svg('svg', {'data-tile': 'vista:' + quais[a % quais.length], x: x, y: 0, width: t, height: t}, g);
        tileSvg(s);
        x += t + (r < reps - 1 ? 2 : 0);
      }
      x += reps > 1 ? 14 : 6;
    }
  }
  function outrosSujeitos(g) {
    var n = +g.getAttribute('data-outros-sujeitos');
    for (var i = 1; i <= n; i++) {
      var y = 46 + i * 34;
      texto(g, 0, y + 5, 'sujeito ' + (i + 1));
      svg('rect', {'class': 'caixa', x: 86, y: y - 13, width: 96, height: 26, rx: 3}, g);
      texto(g, 134, y + 5, 'ridge', {'text-anchor': 'middle'});
      svg('path', {'class': 'linha-mu', d: 'M182,' + y + ' C220,' + y + ' 220,166 248,166'}, g);
    }
  }
  function gradeGama(g) {
    var p = g.getAttribute('data-grade-gama').split(',').map(Number), x0 = p[0], y1 = p[1], y0 = p[2], n = p[3];
    for (var i = 1; i <= n; i++) {
      var y = y0 - (y0 - y1) * (i / n);
      svg('line', {'class': 'linha-mu', x1: x0 - 7, x2: x0, y1: y.toFixed(1), y2: y.toFixed(1)}, g);
    }
  }
  function trio(div) {
    div.innerHTML = '';
    div.className = 'trio';
    div.getAttribute('data-trio').split(',').forEach(function (par) {
      var p = par.split(':'), sup = p[0], faixa = +p[1];
      var c = E && E.classes[sup];
      if (!c) return;
      var k = c.exemplos.findIndex(function (e) { return e.faixa === faixa; });
      var f = document.createElement('figure');
      f.appendChild(tile('ex:' + sup + ':' + Math.max(0, k), 74));
      var cap = document.createElement('figcaption');
      cap.textContent = nome(sup);
      f.appendChild(cap);
      div.appendChild(f);
    });
  }

  // ------------------------------------------------------------------ capa
  function capa(div) {
    var idx = div.getAttribute('data-capa').split(',').map(Number);
    var mod = div.getAttribute('data-capa-modelo').split(':');
    idx.forEach(function (k) {
      var par = document.createElement('div');
      par.className = 'par';
      par.appendChild(tile('vista:' + k, 133));
      par.appendChild(tile('modelo:' + mod[0] + ':' + mod[1] + ':' + k, 133));
      div.appendChild(par);
    });
  }

  // ------------------------------------------------------------------ gráfico: imagens por sessões
  function graficoSessoes(div, slide) {
    if (!DEC) { div.innerHTML = '<p class="nota">Sem os dados do simulador.</p>'; return; }
    var W = 660, H = 320, ml = 58, mr = 18, mt = 12, mb = 40;
    var serie = [];
    for (var N = 1; N <= 40; N++) {
      var c = [0, 0, 0, 0];
      for (var i = 0; i < DEC.n; i++) {
        var v = 0;
        for (var j = 0; j < 3; j++) { var s = DEC.sess[i * 3 + j]; if (s !== 255 && s < N) v++; }
        c[v]++;
      }
      serie.push({N: N, c1: c[1], c2: c[2], c3: c[3], tot: c[1] + c[2] + c[3]});
    }
    var X = function (n) { return ml + (W - ml - mr) * (n - 1) / 39; };
    var Y = function (v) { return mt + (H - mt - mb) * (1 - v / 9000); };
    var el = svg('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img',
      'aria-label': 'Imagens únicas de treino por número de sessões, empilhadas por número de repetições'});
    [0, 3000, 6000, 9000].forEach(function (v) {
      svg('line', {'class': v ? 'grade' : 'base', x1: ml, x2: W - mr, y1: Y(v), y2: Y(v)}, el);
      texto(el, ml - 10, Y(v) + 4, milhar(v), {'class': 'eixo-txt', 'text-anchor': 'end'});
    });
    [1, 10, 20, 30, 40].forEach(function (n) {
      texto(el, X(n), H - mb + 20, n, {'class': 'eixo-txt', 'text-anchor': 'middle'});
    });
    texto(el, W - mr, H - 4, 'sessões', {'class': 'eixo-txt', 'text-anchor': 'end'});
    var camadas = [['c3', 'var(--seq-4)'], ['c2', 'var(--seq-3)'], ['c1', 'var(--seq-2)']];
    var base = serie.map(function () { return 0; });
    camadas.forEach(function (cm) {
      var topo = serie.map(function (p, i) { return base[i] + p[cm[0]]; });
      var d = 'M' + serie.map(function (p, i) { return X(p.N) + ',' + Y(topo[i]); }).join(' L') +
        ' L' + serie.slice().reverse().map(function (p, i) { return X(p.N) + ',' + Y(base[serie.length - 1 - i]); }).join(' L') + ' Z';
      svg('path', {d: d, style: 'fill:' + cm[1] + ';stroke:var(--bg);stroke-width:2;stroke-linejoin:round'}, el);
      base = topo;
    });
    var guia = svg('line', {'class': 'base', y1: mt, y2: H - mb, style: 'opacity:0'}, el);
    var alvo = svg('rect', {'class': 'alvo', x: ml, y: mt, width: W - ml - mr, height: H - mt - mb}, el);
    var dc = dica(slide);
    alvo.addEventListener('mousemove', function (ev) {
      var r = el.getBoundingClientRect(), x = (ev.clientX - r.left) * W / r.width;
      var n = Math.max(1, Math.min(40, Math.round(1 + 39 * (x - ml) / (W - ml - mr)))), p = serie[n - 1];
      guia.setAttribute('x1', X(n)); guia.setAttribute('x2', X(n)); guia.style.opacity = 1;
      dc.mostra('<b>' + n + (n === 1 ? ' sessão' : ' sessões') + '</b>: ' + milhar(p.tot) + ' imagens<br>' +
        '<div class="linha-dica"><span><span class="sw" style="background:var(--seq-2)"></span>vistas 1 vez</span><span>' + milhar(p.c1) + '</span></div>' +
        '<div class="linha-dica"><span><span class="sw" style="background:var(--seq-3)"></span>2 vezes</span><span>' + milhar(p.c2) + '</span></div>' +
        '<div class="linha-dica"><span><span class="sw" style="background:var(--seq-4)"></span>3 vezes</span><span>' + milhar(p.c3) + '</span></div>', ev);
    });
    alvo.addEventListener('mouseleave', function () { guia.style.opacity = 0; dc.some(); });
    div.appendChild(el);
    var leg = $('[data-legenda="sessoes"]', slide);
    if (leg) leg.innerHTML = '<span><span class="sw" style="background:var(--seq-2)"></span>vistas 1 vez</span>' +
      '<span><span class="sw" style="background:var(--seq-3)"></span>2 vezes</span>' +
      '<span><span class="sw" style="background:var(--seq-4)"></span>3 vezes</span>';
  }

  // ------------------------------------------------------------------ barras agrupadas (FRR)
  function barrasAgrupadas(div, slide, grupos, series, opc) {
    var W = opc.W || 560, H = opc.H || 300, ml = 46, mr = 8, mt = 24, mb = 40;
    var max = opc.max || 100;
    var el = svg('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': opc.rotulo || ''});
    var Y = function (v) { return mt + (H - mt - mb) * (1 - v / max); };
    opc.ticks.forEach(function (v) {
      svg('line', {'class': v ? 'grade' : 'base', x1: ml, x2: W - mr, y1: Y(v), y2: Y(v)}, el);
      texto(el, ml - 8, Y(v) + 4, v + '%', {'class': 'eixo-txt', 'text-anchor': 'end'});
    });
    var gw = (W - ml - mr) / grupos.length, bw = Math.min(opc.bw || 34, (gw - 18) / series.length - 2);
    var dc = dica(slide);
    grupos.forEach(function (g, gi) {
      var cx = ml + gw * (gi + 0.5), x0 = cx - (series.length * (bw + 2) - 2) / 2;
      texto(el, cx, H - mb + 22, g, {'class': 'eixo-txt', 'text-anchor': 'middle', style: 'font-size:14px'});
      series.forEach(function (s, si) {
        var v = s.valores[gi];
        if (v == null) return;
        var x = x0 + si * (bw + 2), y = Y(v), h = Y(0) - y;
        var r = Math.min(4, h / 2, bw / 2);
        var d = 'M' + x + ',' + Y(0) + ' V' + (y + r) + ' Q' + x + ',' + y + ' ' + (x + r) + ',' + y +
          ' H' + (x + bw - r) + ' Q' + (x + bw) + ',' + y + ' ' + (x + bw) + ',' + (y + r) + ' V' + Y(0) + ' Z';
        if (h < 1) d = 'M' + x + ',' + (Y(0) - 1) + ' h' + bw + ' v1 h-' + bw + ' Z';
        svg('path', {d: d, style: 'fill:' + s.cor}, el);
        texto(el, x + bw / 2, y - 7, opc.fmt(v), {'class': 'valor', 'text-anchor': 'middle'});
        var a = svg('rect', {'class': 'alvo', x: x - 1, y: mt, width: bw + 2, height: H - mt - mb}, el);
        a.addEventListener('mousemove', function (ev) {
          dc.mostra('<b>' + esc(g) + '</b> · ' + esc(s.nome) + '<br>' + opc.fmt(v) + (opc.extra ? opc.extra(gi, si) : ''), ev);
        });
        a.addEventListener('mouseleave', dc.some);
      });
    });
    div.appendChild(el);
  }

  function graficoGama(div, slide) {
    var g = D.gama;
    if (!g) { div.innerHTML = '<p class="nota">Sem os registros do FRR.</p>'; return; }
    var series = [{nome: '1 sessão', cor: 'var(--seq-2)', valores: g.um}, {nome: '40 sessões', cor: 'var(--seq-3)', valores: g.quarenta}];
    barrasAgrupadas(div, slide, g.rotulos, series, {W: 548, H: 290, ticks: [0, 25, 50, 75, 100], bw: 30,
      fmt: function (v) { return v >= 1 ? virg(v, 0) + '%' : v > 0 ? '<1%' : ''; },
      rotulo: 'Fração escolhida por dimensão, em % das dimensões, com 1 e 40 sessões'});
    var leg = $('[data-legenda="gama"]', slide);
    if (leg) leg.innerHTML = series.map(function (s) {
      return '<span><span class="sw" style="background:' + s.cor + '"></span>' + s.nome + '</span>';
    }).join('');
  }

  function graficoHub(div, slide) {
    var h = D.hub;
    if (!h) { div.innerHTML = '<p class="nota">Sem os registros do FRR.</p>'; return; }
    var series = [{nome: 'como sai do modelo', cor: 'var(--seq-2)', valores: [h.um.bruto, h.quarenta.bruto]},
      {nome: 'sem a média do treino', cor: 'var(--seq-3)', valores: [h.um.centrado, h.quarenta.centrado]}];
    barrasAgrupadas(div, slide, ['1 sessão', '40 sessões'], series, {W: 640, H: 300, ticks: [0, 25, 50, 75, 100], bw: 72,
      fmt: function (v) { return virg(v, 1) + '%'; },
      rotulo: 'Retrieval cérebro para imagem do FRR, antes e depois de tirar a média do treino'});
    var leg = $('[data-legenda="hub"]', slide);
    if (leg) leg.innerHTML = series.map(function (s) {
      return '<span><span class="sw" style="background:' + s.cor + '"></span>' + s.nome + '</span>';
    }).join('');
  }

  // ------------------------------------------------------------------ retrieval: 1 -> 40 sessões
  function graficoRetrieval(div, slide) {
    var linhas = D.retrieval || [];
    var sentido = 'bwd';
    var dc = dica(slide);
    function desenha() {
      div.innerHTML = '';
      var W = 1136, lh = 54, mt = 8, mb = 36, ml = 270, mr = 70, H = mt + lh * linhas.length + mb;
      var X = function (v) { return ml + (W - ml - mr) * v; };
      var el = svg('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img',
        'aria-label': 'Retrieval com 1 e 40 sessões para cada modelo'});
      [0, .25, .5, .75, 1].forEach(function (v) {
        svg('line', {'class': v ? 'grade' : 'base', x1: X(v), x2: X(v), y1: mt, y2: H - mb}, el);
        texto(el, X(v), H - mb + 22, (100 * v) + '%', {'class': 'eixo-txt', 'text-anchor': 'middle'});
      });
      svg('line', {'class': 'acaso', x1: X(1 / 300), x2: X(1 / 300), y1: mt, y2: H - mb}, el);
      linhas.forEach(function (l, i) {
        var y = mt + lh * (i + 0.5), cor = COR[l.familia] || 'var(--ink)';
        var a = l.de[sentido], b = l.para[sentido];
        var g = svg('g', {}, el);
        texto(g, 0, y + 5, l.nome, {'class': 'rot-linha'});
        if (a != null && b != null) {
          svg('line', {x1: X(a), x2: X(b), y1: y, y2: y, style: 'stroke:' + cor + ';stroke-width:3;opacity:.45'}, g);
        }
        if (a != null) {
          svg('circle', {cx: X(a), cy: y, r: 7.5, style: 'fill:var(--bg);stroke:' + cor + ';stroke-width:2.5'}, g);
          texto(g, X(a) + (b != null && b < a ? 14 : -14), y + 5, pct(a), {'class': 'valor',
            'text-anchor': b != null && b < a ? 'start' : 'end'});
        }
        if (b != null) {
          svg('circle', {cx: X(b), cy: y, r: 8, style: 'fill:' + cor + ';stroke:var(--bg);stroke-width:2'}, g);
          texto(g, X(b) + (a != null && a > b ? -14 : 14), y + 5, pct(b), {'class': 'valor',
            'text-anchor': a != null && a > b ? 'end' : 'start'});
        }
        var alvo = svg('rect', {'class': 'alvo', x: 0, y: y - lh / 2, width: W, height: lh}, g);
        alvo.addEventListener('mousemove', function (ev) {
          var ic = function (p) {
            var c = p['ic_' + sentido];
            return c ? ' <span class="mudo">(IC 95%: ' + pct(c[0]) + ' a ' + pct(c[1]) + ')</span>' : '';
          };
          dc.mostra('<b>' + esc(l.nome) + '</b><br>1 sessão: ' + pct(a) + ic(l.de) + '<br>40 sessões: ' + pct(b) + ic(l.para), ev);
        });
        alvo.addEventListener('mouseleave', dc.some);
      });
      div.appendChild(el);
    }
    $$('[data-sentido] button', slide).forEach(function (b) {
      b.addEventListener('click', function () {
        sentido = b.getAttribute('data-v');
        $$('[data-sentido] button', slide).forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
        desenha();
      });
    });
    desenha();
  }

  // ------------------------------------------------------------------ custo x desempenho
  function graficoCusto(div, slide) {
    var pts = (D.custo || []).filter(function (p) { return p.tempo_s && p.bwd != null; });
    if (!pts.length) { div.innerHTML = '<p class="nota">Sem tempos de treino.</p>'; return; }
    var W = 1136, H = 356, ml = 56, mr = 24, mt = 14, mb = 40;
    var lmin = 0.8, lmax = 5.3;
    var X = function (s) { return ml + (W - ml - mr) * (Math.log10(s) - lmin) / (lmax - lmin); };
    var Y = function (v) { return mt + (H - mt - mb) * (1 - v); };
    var el = svg('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Tempo de treino contra retrieval cérebro para imagem'});
    [0, .25, .5, .75, 1].forEach(function (v) {
      svg('line', {'class': v ? 'grade' : 'base', x1: ml, x2: W - mr, y1: Y(v), y2: Y(v)}, el);
      texto(el, ml - 10, Y(v) + 4, (100 * v) + '%', {'class': 'eixo-txt', 'text-anchor': 'end'});
    });
    [[10, '10 s'], [60, '1 min'], [600, '10 min'], [3600, '1 h'], [36000, '10 h'], [86400, '1 dia']].forEach(function (t) {
      svg('line', {'class': 'grade', x1: X(t[0]), x2: X(t[0]), y1: mt, y2: H - mb, style: 'opacity:.6'}, el);
      texto(el, X(t[0]), H - mb + 22, t[1], {'class': 'eixo-txt', 'text-anchor': 'middle'});
    });
    var dc = dica(slide);
    pts.forEach(function (p) {
      var cor = COR[p.familia] || 'var(--ink)', x = X(p.tempo_s), y = Y(p.bwd);
      var g = svg('g', {}, el);
      if (p.sessoes === 1) svg('circle', {cx: x, cy: y, r: 7.5, style: 'fill:var(--bg);stroke:' + cor + ';stroke-width:2.5'}, g);
      else svg('circle', {cx: x, cy: y, r: 8, style: 'fill:' + cor + ';stroke:var(--bg);stroke-width:2'}, g);
      texto(g, x + (p.dx || 14), y + (p.dy || 5), p.rotulo, {'class': 'rot-linha', 'text-anchor': p.ancora || 'start',
        style: 'font-size:14.5px'});
      var a = svg('circle', {'class': 'alvo', cx: x, cy: y, r: 16}, g);
      a.addEventListener('mousemove', function (ev) {
        dc.mostra('<b>' + esc(p.nome) + '</b><br>treino: ' + tempo(p.tempo_s) + '<br>Cér→img: ' + pct(p.bwd), ev);
      });
      a.addEventListener('mouseleave', dc.some);
    });
    div.appendChild(el);
    var leg = $('[data-legenda="custo"]', slide);
    if (leg) leg.innerHTML = '<span><span class="sw" style="background:var(--frr)"></span>FRR</span>' +
      '<span><span class="sw" style="background:var(--me2)"></span>MindEye2 ridge-only</span>' +
      '<span><span class="sw" style="background:var(--me1)"></span>MindEye1</span>' +
      '<span><span class="sw sw-oco"></span>1 sessão</span><span><span class="sw sw-cheio"></span>40 sessões</span>';
  }

  // ------------------------------------------------------------------ matriz de retrieval (ilustração)
  function matriz(el) {
    var cols = [0, 3, 9, 10, 11, 18], n = cols.length, x0 = 100, y0 = 60, c = 46, gap = 5;
    // semelhanças inventadas: a diagonal ganha, menos na linha 4 (a previsão 4 erra)
    var S = [[.82, .31, .22, .40, .18, .27], [.30, .74, .25, .21, .33, .19], [.20, .28, .79, .35, .24, .41],
             [.26, .66, .30, .55, .22, .18], [.17, .29, .38, .25, .71, .30], [.33, .20, .44, .19, .27, .77]];
    cols.forEach(function (k, j) {
      tileSvg(svg('svg', {'data-tile': 'vista:' + k, x: x0 + j * (c + gap), y: 0, width: c, height: c}, el));
    });
    texto(el, x0 - 12, 28, 'imagens', {'class': 't-mu', 'text-anchor': 'end'});
    for (var i = 0; i < n; i++) {
      var y = y0 + i * (c + gap), melhor = S[i].indexOf(Math.max.apply(null, S[i]));
      texto(el, x0 - 12, y + c / 2 + 5, 'previsão ' + (i + 1), {'class': 't-2', 'text-anchor': 'end'});
      for (var j = 0; j < n; j++) {
        var v = S[i][j], nivel = v > .7 ? 4 : v > .5 ? 3 : v > .3 ? 2 : 1;
        svg('rect', {x: x0 + j * (c + gap), y: y, width: c, height: c, rx: 2, style: 'fill:var(--seq-' + nivel + ')'}, el);
      }
      svg('rect', {'class': 'destaque', x: x0 + melhor * (c + gap) - 3, y: y - 3, width: c + 6, height: c + 6, rx: 4}, el);
      texto(el, x0 + n * (c + gap) + 8, y + c / 2 + 7, melhor === i ? '✓' : '✗',
        {style: 'font-size:20px;font-weight:600;fill:' + (melhor === i ? 'var(--ink-2)' : 'var(--ink)')});
    }
  }

  // ------------------------------------------------------------------ exibições: 3.000 de três jeitos
  function exibicoes(el) {
    var linhas = [['3.000 imagens, 1 exibição cada', 1], ['1.500 imagens, 2 exibições cada', 2], ['1.000 imagens, 3 exibições cada', 3]];
    var t = 13, g = 2, entre = 6, total = 30;
    var tons = ['var(--seq-2)', 'var(--seq-3)', 'var(--seq-4)'];
    linhas.forEach(function (l, li) {
      var y = 26 + li * 80;
      texto(el, 0, y, l[0], {'class': 't-gr', style: 'font-size:16px'});
      var x = 0, img = 0;
      for (var q = 0; q < total; q++) {
        svg('rect', {x: x, y: y + 12, width: t, height: 30, rx: 2, style: 'fill:' + tons[img % 3]}, el);
        var fim = (q + 1) % l[1] === 0;
        x += t + (fim ? entre : g);
        if (fim) img++;
      }
      texto(el, x + 6, y + 33, '…', {'class': 't-mu', style: 'font-size:20px'});
    });
  }

  // ------------------------------------------------------------------ galeria
  function galeria(div, slide) {
    if (!G) { div.innerHTML = '<p class="nota">Sem a galeria.</p>'; return; }
    var cols = [{rotulo: 'Imagem vista', vista: true}].concat((D.galeria_colunas || []).filter(function (c) {
      return G.modelos[c.id] && G.modelos[c.id][c.tipo];
    }));
    var porPag = +div.getAttribute('data-galeria'), pag = 0, nPag = Math.ceil(G.n / porPag), T = 112;
    div.style.gridTemplateColumns = 'repeat(' + cols.length + ', ' + T + 'px)';
    function desenha() {
      div.innerHTML = '';
      cols.forEach(function (c) {
        var h = document.createElement('div');
        h.className = 'gal-cab';
        h.innerHTML = c.vista ? '<b>' + c.rotulo + '</b>' :
          '<i style="background:' + (COR[c.familia] || 'var(--ink)') + '"></i><b>' + esc(c.rotulo) + '</b>' + esc(c.sub || '');
        div.appendChild(h);
      });
      for (var r = 0; r < porPag; r++) {
        var k = pag * porPag + r;
        if (k >= G.n) break;
        cols.forEach(function (c) {
          div.appendChild(tile(c.vista ? 'vista:' + k : 'modelo:' + c.id + ':' + c.tipo + ':' + k, T));
        });
      }
      var txt = $('[data-pag-txt]', slide);
      if (txt) txt.textContent = (pag + 1) + ' de ' + nPag;
    }
    $$('[data-pag]', slide).forEach(function (b) {
      b.addEventListener('click', function () {
        pag = (pag + Number(b.getAttribute('data-pag')) + nPag) % nPag;
        desenha();
      });
    });
    desenha();
  }

  // ------------------------------------------------------------------ recorte do NSD
  function recorte(div) {
    var r = E && E.recorte;
    if (!r) { div.innerHTML = '<p class="nota">Sem o exemplo de recorte (make_site.py --exemplos).</p>'; return; }
    var c = E.classes[r.classe], ex = c.exemplos[r.exemplo];
    var col = D.semantica ? D.semantica.sup.indexOf(r.classe) : -1;
    var pix = col >= 0 ? ex.pix[col] : null, tela = (D.semantica && D.semantica.tela) || 180625;
    var H = 232, W = Math.round(H * r.largura / r.altura);
    var cx = r.caixa;
    div.innerHTML =
      '<figure><div class="coco" style="width:' + W + 'px">' +
      '<img src="' + r.arquivo + '" alt="A imagem COCO inteira, com o quadrado que o NSD recortou" width="' + W + '" height="' + H + '">' +
      '<div class="caixa-crop" style="left:' + (cx[0] * W).toFixed(1) + 'px; top:' + (cx[1] * H).toFixed(1) + 'px; width:' +
      (cx[2] * W).toFixed(1) + 'px; height:' + (cx[3] * H).toFixed(1) + 'px"></div></div>' +
      '<figcaption>a imagem COCO inteira (' + r.original[0] + ' × ' + r.original[1] + ')</figcaption></figure>' +
      '<div class="seta" aria-hidden="true">→</div>' +
      '<figure><div data-t1></div><figcaption>o que o NSD mostrou, em 425 × 425</figcaption></figure>' +
      '<div class="seta" aria-hidden="true">→</div>' +
      '<figure><div data-t2></div><figcaption>as máscaras de ' + esc(nome(r.classe)) + ', na tela</figcaption></figure>';
    var t1 = tile('ex:' + r.classe + ':' + r.exemplo, 232);
    t1.querySelector('.mascara').remove();
    $('[data-t1]', div).replaceWith(t1);
    $('[data-t2]', div).replaceWith(tile('ex:' + r.classe + ':' + r.exemplo, 232));
    if (pix != null) {
      var conta = document.createElement('div');
      conta.className = 'conta';
      conta.innerHTML = '<span>' + esc(nome(r.classe)) + '</span><b>' + virg(100 * pix / tela, 1) + '%</b><span>' +
        milhar(pix) + ' de ' + milhar(tela) + ' pixels da tela</span>';
      div.appendChild(conta);
    }
  }

  // ------------------------------------------------------------------ regra exclusiva sobre os pixels de um exemplo
  function fracoesEx(ex) {
    var tela = (D.semantica && D.semantica.tela) || 180625, out = {};
    D.semantica.sup.forEach(function (s, j) { out[s] = 100 * (ex.pix[j] / tela); });
    return out;
  }
  function rotuloEx(fr, classes, minimo, maximo) {      // semantica.rotulo_exclusivo, para uma imagem
    var f = classes.map(function (c) { return Math.min(100, c.sups.reduce(function (a, s) { return a + fr[s]; }, 0)); });
    var rot = null;
    for (var a = 0; a < classes.length; a++) {
      if (f[a] < minimo) continue;
      var ok = true;
      for (var b = 0; b < classes.length; b++) if (b !== a && f[b] >= maximo) { ok = false; break; }
      if (ok) rot = classes[a].nome;
    }
    return rot;
  }

  // ------------------------------------------------------------------ explorador: x% da tela da classe k
  function explorador(div, slide) {
    if (!E || !D.semantica) { div.innerHTML = '<p class="nota">Sem os exemplos (make_site.py --exemplos).</p>'; return; }
    var ordem = (D.ordem_classes || Object.keys(E.classes)).filter(function (s) { return E.classes[s]; });
    var estado = {sup: ordem.indexOf('animal') >= 0 ? 'animal' : ordem[0], faixa: 5, sel: 0};
    var bClasses = $('[data-classes]', div), range = $('[data-x]', div), hist = $('[data-hist]', div);
    var grade = $('[data-grade]', div), detalhe = $('[data-detalhe]', div);
    var presentes = {};
    if (DEC) ordem.forEach(function (s) {
      var j = DEC.col[s], n = 0;
      for (var i = 0; i < DEC.n; i++) if (DEC.frac[i * DEC.k + j] >= 0.5) n++;
      presentes[s] = n;
    });
    ordem.forEach(function (s) {
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = nome(s); b.setAttribute('data-sup', s);
      b.addEventListener('click', function () { estado.sup = s; estado.sel = 0; desenha(); });
      bClasses.appendChild(b);
    });
    E.faixas.forEach(function (f, i) {
      var b = document.createElement('button');
      b.type = 'button'; b.setAttribute('aria-label', 'Faixa de ' + pctTela(f[0]) + ' a ' + pctTela(f[1]));
      b.addEventListener('click', function () { estado.faixa = i; estado.sel = 0; range.value = i; desenha(); });
      hist.appendChild(b);
    });
    var marcas = $('[data-marcas]', div);
    if (marcas) marcas.innerHTML = E.paradas.map(function (x) { return '<span>' + x + '</span>'; }).join('');
    range.addEventListener('input', function () { estado.faixa = +range.value; estado.sel = 0; desenha(); });
    $('[data-mascara]', div).addEventListener('change', function (ev) { div.classList.toggle('sem-mascara', !ev.target.checked); });
    var classes5 = D.classes5 || [];

    function desenha() {
      var c = E.classes[estado.sup], cont = c.contagem, mx = Math.max.apply(null, cont);
      $$('button', bClasses).forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-sup') === estado.sup)); });
      $$('button', hist).forEach(function (b, i) {
        b.style.height = Math.max(2, 64 * Math.sqrt(cont[i] / (mx || 1))) + 'px';
        b.setAttribute('aria-pressed', String(i === estado.faixa));
        b.title = milhar(cont[i]) + ' imagens entre ' + pctTela(E.faixas[i][0]) + ' e ' + pctTela(E.faixas[i][1]);
      });
      var f = E.faixas[estado.faixa];
      $('[data-x-valor]', div).textContent = '≈ ' + E.paradas[estado.faixa] + '%';
      $('[data-x-faixa]', div).textContent = 'entre ' + pctTela(f[0]) + ' e ' + pctTela(f[1]);
      $('[data-x-conta]', div).innerHTML = '<b>' + milhar(cont[estado.faixa]) + '</b> das ' + milhar(D.semantica.n) +
        ' imagens de treino têm ' + esc(nome(estado.sup)) + ' nesta faixa' +
        (presentes[estado.sup] != null ? '; ' + esc(nome(estado.sup)) + ' aparece (≥ 0,5%) em ' + milhar(presentes[estado.sup]) + '.' : '.');
      grade.innerHTML = '';
      var lista = [];
      c.exemplos.forEach(function (e, k) { if (e.faixa === estado.faixa) lista.push(k); });
      if (!lista.length) {
        grade.innerHTML = '<div class="ex-vazio">Nenhuma imagem de treino com ' + esc(nome(estado.sup)) + ' nesta faixa.</div>';
        detalhe.innerHTML = '';
        return;
      }
      estado.sel = Math.min(estado.sel, lista.length - 1);
      lista.forEach(function (k, pos) {
        var e = c.exemplos[k], fr = fracoesEx(e);
        var fig = document.createElement('figure');
        fig.className = 'ex' + (pos === estado.sel ? ' sel' : '');
        var t = tile('ex:' + estado.sup + ':' + k, 216);
        t.setAttribute('role', 'button'); t.tabIndex = 0;
        t.setAttribute('aria-label', 'Detalhar a imagem NSD ' + e.nsd);
        var escolhe = function () { estado.sel = pos; desenha(); };
        t.addEventListener('click', escolhe);
        t.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); escolhe(); } });
        fig.appendChild(t);
        var cap = document.createElement('figcaption');
        cap.innerHTML = '<b>' + esc(nome(estado.sup)) + ' ' + pctTela(fr[estado.sup]) + '</b> · NSD ' + e.nsd;
        fig.appendChild(cap);
        grade.appendChild(fig);
      });
      // a imagem escolhida: o que mais há na tela e o que a regra exclusiva faria com ela
      var e = c.exemplos[lista[estado.sel]], fr = fracoesEx(e);
      var sups = Object.keys(fr).filter(function (s) { return fr[s] > 0; }).sort(function (a, b) { return fr[b] - fr[a]; }).slice(0, 7);
      var mxf = Math.max.apply(null, sups.map(function (s) { return fr[s]; }).concat([1]));
      var rot = classes5.length ? rotuloEx(fr, classes5, 10, 2) : null;
      var naLista = classes5.some(function (k) { return k.sups.indexOf(estado.sup) >= 0; });
      var porque = '';
      if (!rot && classes5.length) {
        var dela = classes5.filter(function (k) { return k.sups.indexOf(estado.sup) >= 0; })[0];
        var outras = classes5.filter(function (k) { return k !== dela; }).map(function (k) {
          return {nome: k.nome, v: Math.min(100, k.sups.reduce(function (a, s) { return a + fr[s]; }, 0))};
        }).filter(function (o) { return o.v >= 2; });
        if (!naLista) porque = esc(nome(estado.sup)) + ' não é uma das 5 classes';
        else if (fr[estado.sup] < 10) porque = esc(nome(estado.sup)) + ' ocupa menos de 10%';
        else if (outras.length) porque = outras.map(function (o) { return esc(o.nome) + ' ' + pctTela(o.v); }).join(', ') + ' passa de 2%';
      }
      detalhe.innerHTML = '<h3>Nesta imagem</h3><div class="comp">' + sups.map(function (s) {
        return '<div class="comp-l' + (s === estado.sup ? ' atual' : '') + '"><span>' + esc(nome(s)) + '</span>' +
          '<span class="comp-b"><span style="width:' + (100 * fr[s] / mxf).toFixed(1) + '%"></span></span>' +
          '<span class="num">' + pctTela(fr[s]) + '</span></div>';
      }).join('') + '</div>' +
        '<p class="comp-reg"><span class="mudo">Regra exclusiva com as 5 classes (≥ 10%, outras &lt; 2%):</span><br>' +
        (rot ? '<b>entra como ' + esc(rot) + '</b>' : '<b>fica de fora</b>' + (porque ? ': ' + porque : '')) + '</p>';
    }
    range.max = E.faixas.length - 1;
    range.value = estado.faixa;
    desenha();
  }

  // ------------------------------------------------------------------ regra exclusiva: quanto sobra
  function regra(div) {
    if (!DEC || !window.Simulador) { div.innerHTML = '<p class="nota">Sem os dados do simulador.</p>'; return; }
    var classes5 = D.classes5 || [], dicio = {};
    classes5.forEach(function (c) { dicio[c.nome] = c.sups.map(function (s) { return 'sup_' + s; }); });
    var rMin = $('[data-min]', div), rMax = $('[data-max]', div), linhas = $('[data-linhas]', div);
    // escala fixa: quantas imagens têm a classe em pelo menos 1% da tela (o teto de qualquer configuração)
    var presenca = {};
    classes5.forEach(function (c) {
      var n = 0;
      for (var i = 0; i < DEC.n; i++) {
        var v = 0;
        c.sups.forEach(function (s) { v += DEC.frac[i * DEC.k + DEC.col[s]]; });
        if (v >= 1) n++;
      }
      presenca[c.nome] = n;
    });
    var teto = Math.max.apply(null, classes5.map(function (c) { return presenca[c.nome]; }));
    function desenha() {
      var minimo = +rMin.value, maximo = +rMax.value / 2;
      $('[data-min-txt]', div).textContent = '≥ ' + minimo + '%';
      $('[data-max-txt]', div).textContent = '< ' + virg(maximo, maximo % 1 ? 1 : 0) + '%';
      var res = Simulador.simula(DEC, {sessoes: 40, classes: dicio, minimo: minimo, maximo_outros: maximo});
      var por = (res && res.elegiveis_por_classe) || {};
      var menor = classes5.reduce(function (a, c) { return (por[c.nome] || 0) < (por[a.nome] || 0) ? c : a; }, classes5[0]);
      var total = classes5.reduce(function (a, c) { return a + (por[c.nome] || 0); }, 0);
      linhas.innerHTML = '';
      classes5.forEach(function (c) {
        var n = por[c.nome] || 0;
        var l = document.createElement('div');
        l.className = 'linha-classe' + (c === menor ? ' gargalo' : '');
        var w = 100 * n / teto, wt = 100 * (presenca[c.nome] || 0) / teto;
        l.innerHTML = '<span class="nome">' + esc(c.nome) + '</span><span class="barra"><i style="width:' + wt.toFixed(1) +
          '%"></i><span style="width:' + w.toFixed(1) +
          '%"></span><em style="left:calc(' + w.toFixed(1) + '% + 8px)">' + milhar(n) + (c === menor ? ' · a menor' : '') +
          '</em></span><span class="miniaturas"></span>';
        var mini = $('.miniaturas', l), sup = c.sups[0], ce = E && E.classes[sup];
        if (ce) {
          var entram = [];
          ce.exemplos.forEach(function (e, k) {
            var fr = fracoesEx(e);
            if (rotuloEx(fr, classes5, minimo, maximo) === c.nome) entram.push({k: k, v: fr[sup]});
          });
          entram.sort(function (a, b) { return a.v - b.v; });
          entram.slice(0, 5).forEach(function (x) {
            var t = tile('ex:' + sup + ':' + x.k, 54);
            t.removeChild(t.querySelector('.mascara'));
            t.title = nome(sup) + ' ' + pctTela(x.v);
            mini.appendChild(t);
          });
          if (!entram.length) mini.innerHTML = '<span class="vazio">nenhum exemplo entra</span>';
        }
        linhas.appendChild(l);
      });
      $('[data-regra-resumo]', div).innerHTML = 'Entram <b>' + milhar(total) + '</b> das ' + milhar(DEC.n) + ' imagens (' +
        virg(100 * total / DEC.n, 0) + '%). Balanceado, cada classe fica com <b>' + milhar(por[menor.nome] || 0) +
        '</b> imagens, o tamanho de ' + esc(menor.nome) + '.';
    }
    rMin.addEventListener('input', desenha);
    rMax.addEventListener('input', desenha);
    desenha();
  }

  // ------------------------------------------------------------------ preenche tudo
  function prepara() {
    $$('svg[data-tile]').forEach(tileSvg);
    $$('[data-voxels]').forEach(voxels);
    $$('[data-vetor]').forEach(vetor);
    $$('[data-sessoes]').forEach(sessoes);
    $$('[data-reps]').forEach(repeticoes);
    $$('[data-outros-sujeitos]').forEach(outrosSujeitos);
    $$('[data-grade-gama]').forEach(gradeGama);
    $$('[data-trio]').forEach(trio);
    $$('[data-capa]').forEach(capa);
    $$('[data-matriz]').forEach(matriz);
    $$('[data-exibicoes]').forEach(exibicoes);
    $$('[data-recorte]').forEach(recorte);
    var porNome = {sessoes: graficoSessoes, gama: graficoGama, hub: graficoHub, retrieval: graficoRetrieval, custo: graficoCusto};
    $$('[data-grafico]').forEach(function (d) {
      var f = porNome[d.getAttribute('data-grafico')];
      if (f) f(d, d.closest('.slide'));
    });
    $$('[data-galeria]').forEach(function (d) { galeria(d, d.closest('.slide')); });
    $$('[data-explorador]').forEach(function (d) { explorador(d, d.closest('.slide')); });
    $$('[data-regra]').forEach(regra);
  }

  // ------------------------------------------------------------------ slides: escala, navegação, sumário
  var slides = $$('.slide'), atual = 0, doc = false;
  slides.forEach(function (s, i) {
    var pe = document.createElement('footer');
    pe.className = 'pe';
    pe.innerHTML = '<span class="marca">benchmark-modelos</span><span>' + (s.getAttribute('data-secao') || '') +
      '</span><span>' + (i + 1) + ' / ' + slides.length + '</span>';
    if (i > 0) s.appendChild(pe);
  });

  function escala() {
    var s = doc ? Math.min(1, (window.innerWidth - 32) / 1280) : Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    raiz.style.setProperty('--s', s.toFixed(4));
  }
  function indice(h) {
    h = (h || '').replace(/^#/, '');
    if (!h) return 0;
    if (/^\d+$/.test(h)) return Math.max(0, Math.min(slides.length - 1, +h - 1));
    var i = slides.findIndex(function (s) { return s.id === h; });
    return i < 0 ? 0 : i;
  }
  /* o slide atual: o visível no modo slides, o que está na tela no documento */
  function marca(i) {
    slides[atual].classList.remove('ativo');
    atual = i;
    slides[i].classList.add('ativo');
    $('.contador').textContent = (i + 1) + ' / ' + slides.length;
    $('.progresso').style.width = (100 * (i + 1) / slides.length) + '%';
    $$('.sumario a').forEach(function (a, k) { a.setAttribute('aria-current', String(k === i)); });
  }
  function vai(i, rola, imediato) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    marca(i);
    try { history.replaceState(null, '', '#' + slides[i].id); } catch (e) { /* file:// em alguns navegadores */ }
    if (doc && rola) slides[i].closest('.folha').scrollIntoView({behavior: imediato ? 'auto' : 'smooth', block: 'start'});
  }
  function modo(documento) {
    doc = documento;
    document.body.classList.toggle('documento', doc);
    var b = $('[data-cmd="modo"]');
    if (b) b.textContent = doc ? 'Slides' : 'Documento';
    escala();
    if (doc) slides[atual].closest('.folha').scrollIntoView({block: 'start'});
  }
  function sumario() {
    var div = $('.sumario'), html = '<ol>', secao = null;
    slides.forEach(function (s, i) {
      var sc = s.getAttribute('data-secao') || '';
      if (sc && sc !== secao) html += '<li class="secao-sum" aria-hidden="true">' + esc(sc) + '</li>';
      secao = sc;
      html += '<li><a href="#' + s.id + '" data-i="' + i + '">' + esc(s.getAttribute('data-titulo') || s.id) + '</a></li>';
    });
    div.innerHTML = html + '</ol>';
    $$('a', div).forEach(function (a) {
      a.addEventListener('click', function (ev) { ev.preventDefault(); vai(+a.getAttribute('data-i'), true); div.hidden = true; });
    });
  }
  function telaCheia() {
    if (document.fullscreenElement) document.exitFullscreen();
    else if (raiz.requestFullscreen) raiz.requestFullscreen().catch(function () { /* recusado */ });
  }

  document.addEventListener('keydown', function (ev) {
    if (ev.altKey || ev.ctrlKey || ev.metaKey) return;
    var t = ev.target, tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if ((tag === 'BUTTON' || (t.getAttribute && t.getAttribute('role') === 'button')) && (ev.key === ' ' || ev.key === 'Enter')) return;
    var k = ev.key;
    if (k === 'Escape') { $('.sumario').hidden = true; return; }
    if (k === 'd' || k === 'D') { modo(!doc); return; }
    if (k === 't' || k === 'T') { trocaTema(); return; }
    if (k === 'f' || k === 'F') { telaCheia(); return; }
    if (doc) return;                                   // no documento, as setas rolam a página
    if (k === 'ArrowRight' || k === 'PageDown' || k === ' ' || k === 'ArrowDown') { ev.preventDefault(); vai(atual + 1); }
    else if (k === 'ArrowLeft' || k === 'PageUp' || k === 'ArrowUp') { ev.preventDefault(); vai(atual - 1); }
    else if (k === 'Home') { ev.preventDefault(); vai(0); }
    else if (k === 'End') { ev.preventDefault(); vai(slides.length - 1); }
  });
  var toque = null;
  document.addEventListener('touchstart', function (ev) {
    if (doc || ev.touches.length !== 1 || ev.target.closest('input, button, [role="button"]')) { toque = null; return; }
    toque = {x: ev.touches[0].clientX, y: ev.touches[0].clientY};
  }, {passive: true});
  document.addEventListener('touchend', function (ev) {
    if (!toque) return;
    var dx = ev.changedTouches[0].clientX - toque.x, dy = ev.changedTouches[0].clientY - toque.y;
    if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy)) vai(atual + (dx < 0 ? 1 : -1));
    toque = null;
  });

  var cmds = $('.comandos'), quieto = null;
  function acorda() {
    cmds.classList.remove('quieto');
    clearTimeout(quieto);
    quieto = setTimeout(function () { if ($('.sumario').hidden && !doc) cmds.classList.add('quieto'); }, 2600);
  }
  document.addEventListener('mousemove', acorda);
  $$('[data-cmd]').forEach(function (b) {
    b.addEventListener('click', function () {
      var c = b.getAttribute('data-cmd');
      if (c === 'ant') vai(atual - 1, true);
      else if (c === 'prox') vai(atual + 1, true);
      else if (c === 'tema') trocaTema();
      else if (c === 'modo') modo(!doc);
      else if (c === 'cheia') telaCheia();
      else if (c === 'sumario') { var s = $('.sumario'); s.hidden = !s.hidden; }
    });
  });
  // no documento, o contador acompanha o slide que está na tela (o endereço não muda: trocar o
  // fragmento durante o carregamento faria o navegador rolar até ele)
  var pendente = false;
  window.addEventListener('scroll', function () {
    if (!doc || pendente) return;
    pendente = true;
    requestAnimationFrame(function () {
      pendente = false;
      var y = window.innerHeight * 0.3;
      var i = slides.findIndex(function (s) { return s.closest('.folha').getBoundingClientRect().bottom > y; });
      if (i >= 0 && i !== atual) marca(i);
    });
  }, {passive: true});
  window.addEventListener('resize', escala);
  window.addEventListener('hashchange', function () { var i = indice(location.hash); if (i !== atual) vai(i, true); });
  window.addEventListener('beforeprint', function () { slides.forEach(function (s) { s.classList.add('ativo'); }); });
  window.addEventListener('afterprint', function () { slides.forEach(function (s, i) { s.classList.toggle('ativo', i === atual); }); });

  try { prepara(); } catch (e) { if (window.console) console.error(e); document.body.setAttribute('data-erro', String(e && e.stack || e)); }
  sumario();
  rotulaTema();
  modo(window.innerWidth < 720);
  vai(indice(location.hash), !!location.hash, true);
  acorda();
})();
