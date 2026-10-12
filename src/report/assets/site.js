/* O site do benchmark: troca de paginas, simulador de dataset e resultados interativos. Os dados vem
   do <script id="dados"> que src/make_site.py grava na pagina; nada e buscado na rede. */
(function () {
  'use strict';
  var D = JSON.parse(document.getElementById('dados').textContent);
  var doc = document, raiz = doc.documentElement;
  var $ = function (s, r) { return (r || doc).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || doc).querySelectorAll(s)); };

  function h(tag, at) {
    var e = doc.createElement(tag);
    if (at) Object.keys(at).forEach(function (k) {
      var v = at[k];
      if (v == null || v === false) return;
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k === 'texto') e.textContent = v;
      else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    });
    for (var i = 2; i < arguments.length; i++) poe(e, arguments[i]);
    return e;
  }
  function poe(e, f) {
    if (f == null || f === false) return;
    if (Array.isArray(f)) { f.forEach(function (x) { poe(e, x); }); return; }
    e.append(f.nodeType ? f : String(f));
  }
  var NS = 'http://www.w3.org/2000/svg';
  function sv(tag, at, pai) {
    var e = doc.createElementNS(NS, tag);
    if (at) Object.keys(at).forEach(function (k) { if (at[k] != null) e.setAttribute(k, at[k]); });
    if (pai) pai.appendChild(e);
    return e;
  }
  function svTexto(at, txt, pai) { var t = sv('text', at, pai); t.textContent = txt; return t; }

  // ------------------------------------------------------------------ formatacao
  function virg(x, c) { return x.toFixed(c).replace('.', ',').replace('-', '−'); }
  function milhar(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, '.'); }
  function fmt(v, f) {
    if (v == null || isNaN(v)) return '—';
    if (f === 'pctc') return (Math.round(v * 1000) / 10).toString().replace('.', ',') + '%';   // eixos
    if (f === 'decc') return (Math.round(v * 1000) / 1000).toString().replace('.', ',').replace('-', '−');
    if (f === 'pct') return virg(v * 100, 1) + '%';
    if (f === 'dec') return virg(v, 3);
    if (f === 'int') return milhar(v);
    if (f === 'h') return v < 1 ? Math.round(v * 60) + ' min' : virg(v, 1) + ' h';
    return String(v);
  }
  function diferenca(v, f) { return f === 'pct' ? virg(v * 100, 1) + ' p.p.' : fmt(v, f); }
  function duracao(s) {
    if (s == null) return '—';
    if (s < 90) return Math.round(s) + ' s';
    var m = Math.round(s / 60), hh = Math.floor(m / 60);
    return hh ? hh + ' h ' + String(m % 60).padStart(2, '0') : m + ' min';
  }
  function mi(n) { return n >= 1e9 ? virg(n / 1e9, 2) + ' bi' : virg(n / 1e6, 1) + ' M'; }
  function plural(n, um, varios) { return milhar(n) + ' ' + (n === 1 ? um : varios); }

  // ------------------------------------------------------------------ preferencias (so deste navegador)
  var CHAVE = 'benchmark-modelos/site/v1';
  var salvo = {};
  try { salvo = JSON.parse(localStorage.getItem(CHAVE) || '{}') || {}; } catch (e) { salvo = {}; }
  try { var tema0 = localStorage.getItem(CHAVE + '/tema'); if (tema0) raiz.setAttribute('data-theme', tema0); } catch (e) { /* sem armazenamento */ }

  // ------------------------------------------------------------------ modelos, metricas e cores
  var MODELOS = D.modelos;
  var POR_ID = {};
  MODELOS.forEach(function (m) { POR_ID[m.id] = m; });
  var METRICAS = D.metricas, MET = {};
  METRICAS.forEach(function (m) { MET[m.id] = m; });
  var FAM = {};
  D.familias.forEach(function (f) { FAM[f.id] = f; });
  var DEGRAUS = [0, 13, -9, 23, -16, 7, 30, -4, 18, -12, 27, 3, -19, 11];
  (function () {
    var cont = {};
    MODELOS.forEach(function (m) {
      var chave = m.grupo === 'experimento' ? 'experimento' : m.familia;
      var f = FAM[chave] || {h: 0, s: 0, l: 45};
      var i = cont[chave] = (cont[chave] == null ? -1 : cont[chave]) + 1;
      var l = Math.max(20, Math.min(70, f.l + DEGRAUS[i % DEGRAUS.length]));
      m.cor = 'hsl(' + f.h + ' ' + f.s + '% calc(' + l + '% + var(--ajuste-l)))';
    });
  })();
  function temDados(m) { return !!(m.valores || m.curva); }

  var EXTRAS = {
    sessoes: {id: 'sessoes', rotulo: 'Sessões', nome: 'Sessões de treino', fmt: 'int', log: true,
      get: function (m) { return m.sessoes; }},
    exibicoes: {id: 'exibicoes', rotulo: 'Exibições', nome: 'Exibições de treino', fmt: 'int', log: true,
      get: function (m) { return m.resumo_dataset ? m.resumo_dataset.exibicoes : null; }},
    imagens: {id: 'imagens', rotulo: 'Imagens', nome: 'Imagens únicas de treino', fmt: 'int', log: true,
      get: function (m) { return m.resumo_dataset ? m.resumo_dataset.imagens_unicas : null; }},
    tempo_h: {id: 'tempo_h', rotulo: 'Tempo', nome: 'Tempo de treino (horas)', fmt: 'h', log: true,
      get: function (m) { return m.tempo_s != null ? m.tempo_s / 3600 : null; }}
  };
  function descritor(id) { return EXTRAS[id] || MET[id]; }
  function valor(m, id, tipo) {
    if (EXTRAS[id]) return EXTRAS[id].get(m);
    var v = m.valores && m.valores[tipo || E.tipo];
    return v && v[id] != null ? v[id] : null;
  }
  function intervalo(m, id) {
    var r = m.retrieval;
    if (!r) return null;
    var lado = id === 'FwdRetrieval' ? r.fwd : id === 'BwdRetrieval' ? r.bwd : null;
    return lado && lado.ic95 ? lado.ic95 : null;
  }
  function marcas(m, id) {
    var s = '';
    if (MET[id] && MET[id].grupo === 'retrieval') {
      if (m.espaco_clip === 'ViT-L/14') s += '†';
      if (m.retrieval && m.retrieval.fonte === 'treino') s += '‡';
    }
    return s;
  }

  // ------------------------------------------------------------------ estado
  var ABAS = [['resumo', 'Resumo'], ['tabela', 'Tabela'], ['graficos', 'Gráficos'], ['dispersao', 'Dispersão'],
              ['treino', 'Treino'], ['galeria', 'Galeria'], ['frr', 'FRR'], ['fichas', 'Fichas']];
  var NOMES_ABAS = {};
  ABAS.forEach(function (a) { NOMES_ABAS[a[0]] = a[1]; });
  var validos = (salvo.sel || []).filter(function (id) { return POR_ID[id] && temDados(POR_ID[id]); });
  var E = {
    pagina: 'inicio', aba: NOMES_ABAS[salvo.aba] ? salvo.aba : 'resumo',
    sel: new Set(salvo.sel ? validos : D.padrao),
    tipo: salvo.tipo === 'unclip' ? 'unclip' : 'refinada',
    colunas: Array.isArray(salvo.colunas) ? salvo.colunas : ['retrieval', 'baixo', 'alto'],
    grupo: salvo.grupo || 'retrieval', escala: salvo.escala || 'ajustada', ordenaGraf: !!salvo.ordenaGraf,
    x: descritor(salvo.x) ? salvo.x : 'exibicoes', y: descritor(salvo.y) ? salvo.y : 'BwdRetrieval',
    logx: salvo.logx !== false, rotulosDisp: salvo.rotulosDisp !== false,
    curva: salvo.curva || 'teste_bwd', eixoCurva: salvo.eixoCurva || 'epoca',
    galTipo: salvo.galTipo || 'refinada', galLeg: salvo.galLeg !== false, galTam: salvo.galTam || 'm',
    resumoMet: salvo.resumoMet || 'BwdRetrieval', ordem: salvo.ordem || null, ordemDesc: salvo.ordemDesc !== false,
    fichas: salvo.fichas || 'selecionados', busca: '',
    fechados: Array.isArray(salvo.fechados) ? salvo.fechados : []
  };
  function guarda() {
    var o = {};
    Object.keys(E).forEach(function (k) { if (k !== 'pagina' && k !== 'busca') o[k] = E[k]; });
    o.sel = Array.from(E.sel);
    try { localStorage.setItem(CHAVE, JSON.stringify(o)); } catch (e) { /* sem armazenamento */ }
  }
  function selecionados() { return MODELOS.filter(function (m) { return E.sel.has(m.id); }); }

  // ------------------------------------------------------------------ dica flutuante
  var dica = h('div', {class: 'dica', role: 'tooltip', hidden: true});
  doc.body.appendChild(dica);
  function mostraDica(ev, html) {
    dica.innerHTML = html;
    dica.hidden = false;
    var x = ev.clientX + 14, y = ev.clientY + 14;
    var r = dica.getBoundingClientRect();
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
    dica.style.left = Math.max(8, x) + 'px';
    dica.style.top = Math.max(8, y) + 'px';
  }
  function escondeDica() { dica.hidden = true; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }

  // ------------------------------------------------------------------ copiar
  function copia(texto, botao) {
    function ok() { if (botao) { var t = botao.textContent; botao.textContent = 'Copiado'; setTimeout(function () { botao.textContent = t; }, 1400); } }
    function manual() {
      var ta = h('textarea', {class: 'mono', rows: 6, style: 'width:100%'});
      ta.value = texto;
      if (botao && botao.parentNode) { botao.parentNode.insertBefore(ta, botao.nextSibling); ta.select(); }
    }
    try {
      navigator.clipboard.writeText(texto).then(ok, manual);
    } catch (e) { manual(); }
  }
  $$('pre').forEach(function (pre) {
    var cx = h('div', {class: 'cod'});
    pre.parentNode.insertBefore(cx, pre);
    cx.appendChild(pre);
    var b = h('button', {class: 'copiar', type: 'button', texto: 'Copiar'});
    b.addEventListener('click', function () { copia(pre.innerText.replace(/\n$/, ''), b); });
    cx.appendChild(b);
  });

  // ------------------------------------------------------------------ tema
  var botaoTema = $('.tema');
  function escuro() {
    var t = raiz.getAttribute('data-theme');
    if (t) return t === 'dark';
    return window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function rotuloTema() { if (botaoTema) botaoTema.textContent = escuro() ? 'Tema claro' : 'Tema escuro'; }
  if (botaoTema) {
    botaoTema.addEventListener('click', function () {
      var novo = escuro() ? 'light' : 'dark';
      raiz.setAttribute('data-theme', novo);
      try { localStorage.setItem(CHAVE + '/tema', novo); } catch (e) { /* sem armazenamento */ }
      rotuloTema();
    });
    rotuloTema();
  }

  // ------------------------------------------------------------------ paginas e endereco
  function rota() {
    var t = decodeURIComponent(location.hash.slice(1)) || 'inicio';
    var pag = 'inicio', alvo = null, aba = null;
    if (NOMES_ABAS[t]) { pag = 'resultados'; aba = t; }
    else if (t === 'resultados') pag = 'resultados';
    else if (t === 'uso' || t === 'inicio') pag = t;
    else {
      var el = doc.getElementById(t), p = el && el.closest('.pagina');
      if (p) { pag = p.getAttribute('data-pag'); alvo = t; }
    }
    mostraPagina(pag, alvo, aba);
  }
  function mostraPagina(pag, alvo, aba) {
    var trocou = E.pagina !== pag;
    E.pagina = pag;
    $$('.pagina').forEach(function (p) { p.classList.toggle('ativa', p.getAttribute('data-pag') === pag); });
    $$('.paginas a').forEach(function (a) { a.setAttribute('aria-current', a.getAttribute('data-pag') === pag ? 'page' : 'false'); });
    if (pag === 'resultados') { iniciaResultados(); trocaAba(aba || E.aba, false); }
    if (pag === 'uso') iniciaSimulador();
    if (alvo) {
      var el = doc.getElementById(alvo), suave = !primeira;
      if (el) requestAnimationFrame(function () { el.scrollIntoView({behavior: suave ? 'smooth' : 'instant', block: 'start'}); });
    } else if (trocou) {
      window.scrollTo(0, 0);
    }
  }
  var primeira = true;
  window.addEventListener('hashchange', rota);

  // indice lateral: marca a secao visivel
  if ('IntersectionObserver' in window) {
    $$('.indice').forEach(function (ind) {
      var links = $$('a', ind);
      var obs = new IntersectionObserver(function (ents) {
        ents.forEach(function (en) {
          if (!en.isIntersecting) return;
          links.forEach(function (a) { a.classList.toggle('atual', a.getAttribute('href') === '#' + en.target.id); });
        });
      }, {rootMargin: '-20% 0px -70% 0px'});
      links.forEach(function (a) { var s = doc.getElementById(a.getAttribute('href').slice(1)); if (s) obs.observe(s); });
    });
  }

  // ------------------------------------------------------------------ escalas
  function passoBonito(span, n) {
    var bruto = span / n, p = Math.pow(10, Math.floor(Math.log10(bruto))), f = bruto / p;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
  }
  function marcasEixo(lo, hi, n) {
    var p = passoBonito(hi - lo || 1, n || 4), out = [];
    for (var v = Math.ceil(lo / p - 1e-9) * p; v <= hi + p * 1e-6; v += p) out.push(+v.toFixed(10));
    return out;
  }
  function dominio(vals, opc) {
    opc = opc || {};
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (opc.zero) lo = Math.min(0, lo);
    if (opc.incluir != null) { lo = Math.min(lo, opc.incluir); hi = Math.max(hi, opc.incluir); }
    if (hi === lo) { hi += Math.abs(hi) * 0.05 || 0.01; lo -= opc.zero ? 0 : (Math.abs(lo) * 0.05 || 0.01); }
    var pad = (hi - lo) * 0.06;
    if (!opc.zero) lo -= pad;
    hi += pad;
    var p = passoBonito(hi - lo, 4);
    lo = Math.floor(lo / p) * p; hi = Math.ceil(hi / p) * p;
    if (opc.pct) { lo = Math.max(0, lo); hi = Math.min(1, hi); }
    if (opc.zero) lo = Math.max(0, lo) === 0 ? 0 : lo;
    return [lo, hi];
  }
  function marcasLog(lo, hi) {
    var out = [];
    for (var e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      [1, 2, 5].forEach(function (k) { var v = k * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); });
    }
    return out;
  }

  // ------------------------------------------------------------------ grafico de pontos/barras (HTML)
  /* itens: [{m, v, ic}]; mt: descritor da metrica. Devolve um .gb */
  function grafico(itens, mt, opc) {
    opc = opc || {};
    var barras = opc.barras, pct = mt.fmt === 'pct';
    var vals = [];
    itens.forEach(function (it) { vals.push(it.v); if (it.ic) vals.push(it.ic[0], it.ic[1]); });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), dom;
    if (barras) {
      dom = pct && mt.grupo === 'retrieval' ? [0, 1] : [0, dominio(vals, {zero: true, pct: pct})[1]];
    } else {
      // o acaso entra no eixo quando esta perto dos valores (no retrieval do FRR de 1 sessao, por exemplo)
      var perto = mt.acaso != null && mt.acaso < lo && lo - mt.acaso < (hi - lo) * 0.6;
      dom = dominio(vals, {pct: pct, incluir: perto ? mt.acaso : null});
    }
    var X = function (v) { return Math.max(0, Math.min(100, (v - dom[0]) / (dom[1] - dom[0]) * 100)); };
    var ticks = marcasEixo(dom[0], dom[1], 4).filter(function (t) { return t >= dom[0] - 1e-9 && t <= dom[1] + 1e-9; });
    var grade = ticks.map(function (t) {
      var p = X(t).toFixed(3);
      return 'linear-gradient(to right, transparent calc(' + p + '% - .5px), var(--rule) calc(' + p + '% - .5px), var(--rule) calc(' + p + '% + .5px), transparent calc(' + p + '% + .5px))';
    }).join(',');
    var melhor = null;
    var candidatos = itens.filter(function (it) { return !it.m.publicada; });
    if (candidatos.length > 1) {
      melhor = candidatos.reduce(function (a, b) { return (mt.maior ? b.v > a.v : b.v < a.v) ? b : a; });
    }
    var gb = h('div', {class: 'gb', style: '--grade:' + (grade || 'none')});
    itens.forEach(function (it) {
      var m = it.m, x = X(it.v), lin = it === melhor;
      var trilho = h('div', {class: 'gb-trilho', style: '--c:' + m.cor});
      if (barras) trilho.appendChild(h('span', {class: 'gb-barra', style: 'width:' + x.toFixed(2) + '%'}));
      if (it.ic) {
        var a = X(it.ic[0]), b = X(it.ic[1]);
        trilho.appendChild(h('span', {class: 'gb-ic', style: 'left:' + a.toFixed(2) + '%;width:' + Math.max(0.4, b - a).toFixed(2) + '%'}));
      }
      if (!barras) trilho.appendChild(h('span', {class: 'gb-ponto', style: 'left:' + x.toFixed(2) + '%'}));
      if (mt.acaso != null && mt.acaso >= dom[0] && mt.acaso <= dom[1]) {
        trilho.appendChild(h('span', {class: 'gb-acaso', style: 'left:' + X(mt.acaso).toFixed(2) + '%'}));
      }
      var dentro = barras && x > 80;
      var txt = fmt(it.v, mt.fmt) + (it.marca || '');
      trilho.appendChild(h('span', {class: 'gb-val' + (dentro ? ' dentro' : ''),
        style: 'left:' + (barras ? x : x + 1.2).toFixed(2) + '%', texto: txt}));
      var rot = h('div', {class: 'gb-rot', texto: m.rotulo});
      var linha = [rot, trilho];
      linha.forEach(function (el) {
        if (lin) el.classList.add('gb-linha-melhor');
        el.addEventListener('pointermove', function (ev) {
          var html = '<b>' + esc(m.rotulo) + '</b><br>' + esc(mt.nome || mt.rotulo) + ': ' + esc(txt);
          if (it.ic) html += '<br>IC 95%: ' + fmt(it.ic[0], mt.fmt) + ' a ' + fmt(it.ic[1], mt.fmt);
          if (it.nota) html += '<br>' + esc(it.nota);
          mostraDica(ev, html);
        });
        el.addEventListener('pointerleave', escondeDica);
      });
      if (lin) rot.classList.add('gb-linha-melhor');
      gb.appendChild(rot); gb.appendChild(trilho);
    });
    var eixo = h('div', {class: 'gb-eixo'});
    ticks.forEach(function (t, i) {
      // o ultimo rotulo fica alinhado pela direita; o penultimo sai se ficar colado nele
      if (i === ticks.length - 2 && X(ticks[i + 1]) - X(t) < 14) return;
      eixo.appendChild(h('span', {style: 'left:' + X(t).toFixed(2) + '%', texto: fmt(t, mt.fmt === 'pct' ? 'pctc' : mt.fmt === 'dec' ? 'decc' : mt.fmt)}));
    });
    gb.appendChild(eixo);
    return gb;
  }
  // ------------------------------------------------------------------ controles
  function segmentado(rotulo, opcoes, atual, aoMudar) {
    var g = h('div', {class: 'seg', role: 'group', 'aria-label': rotulo});
    opcoes.forEach(function (o) {
      var b = h('button', {type: 'button', 'aria-pressed': String(o[0] === atual), texto: o[1],
        disabled: o[2] ? true : null, title: o[3] || null});
      b.addEventListener('click', function () {
        $$('button', g).forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
        aoMudar(o[0]);
      });
      g.appendChild(b);
    });
    return h('div', {class: 'controle'}, rotulo ? h('span', null, rotulo) : null, g);
  }
  function seletor(rotulo, opcoes, atual, aoMudar, id) {
    var s = h('select', {id: id});
    opcoes.forEach(function (o) {
      if (o.grupo) {
        var og = h('optgroup', {label: o.grupo});
        o.itens.forEach(function (i) { og.appendChild(h('option', {value: i[0], texto: i[1], selected: i[0] === atual ? true : null})); });
        s.appendChild(og);
      } else s.appendChild(h('option', {value: o[0], texto: o[1], selected: o[0] === atual ? true : null}));
    });
    s.addEventListener('change', function () { aoMudar(s.value); });
    return h('label', {class: 'controle', for: id}, rotulo, s);
  }
  function caixa(rotulo, marcado, aoMudar, id) {
    var c = h('input', {type: 'checkbox', id: id, checked: marcado ? true : null});
    c.addEventListener('change', function () { aoMudar(c.checked); });
    return h('label', {class: 'controle', for: id}, c, rotulo);
  }
  function vazio(texto, acao) {
    return h('div', {class: 'vazio'}, h('p', {texto: texto}), acao || null);
  }
  function opcoesTipo() {
    return segmentado('Reconstrução', [['refinada', 'Refinadas', null, 'img2img guiado pela legenda prevista; nos modelos com ramo blurry, 75% refinada + 25% borrada, como no artigo'],
      ['unclip', 'unCLIP', null, 'a saída do unCLIP, sem refinar']], E.tipo, function (v) { E.tipo = v; guarda(); renderAba(); });
  }

  // ------------------------------------------------------------------ resultados: estrutura
  var resIniciado = false;
  function iniciaResultados() {
    if (resIniciado) return;
    resIniciado = true;
    var abas = $('#abas');
    ABAS.forEach(function (a) {
      var b = h('button', {type: 'button', role: 'tab', id: 'aba-' + a[0], 'aria-controls': 'vista-' + a[0], 'data-aba': a[0], texto: a[1]});
      b.addEventListener('click', function () { trocaAba(a[0], true); });
      abas.appendChild(b);
    });
    constroiLista();
    $('#busca').addEventListener('input', function (ev) {
      E.busca = ev.target.value.trim().toLowerCase();
      $('#lista-modelos').classList.toggle('buscando', !!E.busca);
      $$('.item-modelo').forEach(function (it) { it.hidden = E.busca && it.getAttribute('data-busca').indexOf(E.busca) < 0; });
    });
    $('#sel-todos').addEventListener('click', function () {
      $$('.item-modelo').forEach(function (it) { if (it.offsetParent) E.sel.add($('input', it).value); });
      mudouSelecao();
    });
    $('#dobra-todos').addEventListener('click', function () {
      var ids = $$('.grupo-modelos').map(function (b) { return b.getAttribute('data-grupo'); });
      E.fechados = E.fechados.length < ids.length ? ids : [];
      sincronizaDobras();
      guarda();
    });
    $('#sel-nenhum').addEventListener('click', function () { E.sel = new Set(); mudouSelecao(); });
    if ('ResizeObserver' in window) {
      var larg = 0, t = null;
      new ResizeObserver(function (ents) {
        var w = Math.round(ents[0].contentRect.width);
        if (Math.abs(w - larg) < 8) return;
        larg = w;
        clearTimeout(t);
        t = setTimeout(function () { if (['resumo', 'dispersao', 'treino', 'frr'].indexOf(E.aba) >= 0) renderAba(); }, 120);
      }).observe($('#vistas'));
    }
    if (innerWidth < 1080) $('#trilho').open = false;
  }
  function constroiLista() {
    var lista = $('#lista-modelos');
    lista.textContent = '';
    D.grupos.forEach(function (g) {
      var ms = MODELOS.filter(function (m) { return (m.grupo === 'experimento' ? 'experimento' : m.familia) === g.id && temDados(m); });
      if (!ms.length) return;
      var cab = h('input', {type: 'checkbox', 'aria-label': 'Todos de ' + g.nome});
      var dobra = h('button', {class: 'dobra', type: 'button', 'aria-controls': 'grupo-' + g.id},
        h('span', {texto: g.nome}), h('span', {class: 'mudo num'}));
      var itens = h('div', {class: 'grupo-itens', id: 'grupo-' + g.id});
      var bloco = h('div', {class: 'grupo-modelos', 'data-grupo': g.id}, h('div', {class: 'grupo-cab'}, cab, dobra), itens);
      dobra.addEventListener('click', function () {
        var i = E.fechados.indexOf(g.id);
        if (i < 0) E.fechados.push(g.id); else E.fechados.splice(i, 1);
        sincronizaDobras();
        guarda();
      });
      cab.addEventListener('change', function () {
        ms.forEach(function (m) { if (cab.checked) E.sel.add(m.id); else E.sel.delete(m.id); });
        mudouSelecao();
      });
      ms.forEach(function (m) {
        var cb = h('input', {type: 'checkbox', value: m.id, id: 'sel-' + m.id});
        cb.addEventListener('change', function () { if (cb.checked) E.sel.add(m.id); else E.sel.delete(m.id); mudouSelecao(); });
        itens.appendChild(h('label', {class: 'item-modelo', for: 'sel-' + m.id, 'data-busca': (m.rotulo + ' ' + m.id).toLowerCase(), title: m.id},
          cb, h('span', {class: 'amostra', style: '--c:' + m.cor}), h('span', {texto: m.rotulo}),
          h('span', {class: 'meta'}, h('span', {class: 'ponto-status st-' + m.status}), m.status_txt)));
      });
      lista.appendChild(bloco);
    });
    sincronizaLista();
    sincronizaDobras();
  }
  function sincronizaDobras() {
    var blocos = $$('.grupo-modelos');
    blocos.forEach(function (b) {
      var fechado = E.fechados.indexOf(b.getAttribute('data-grupo')) >= 0;
      b.classList.toggle('fechado', fechado);
      $('.dobra', b).setAttribute('aria-expanded', String(!fechado));
    });
    var abertos = blocos.filter(function (b) { return !b.classList.contains('fechado'); }).length;
    $('#dobra-todos').textContent = abertos ? 'Recolher grupos' : 'Expandir grupos';
  }
  function sincronizaLista() {
    $$('.item-modelo input').forEach(function (cb) { cb.checked = E.sel.has(cb.value); });
    $$('.grupo-modelos').forEach(function (b) {
      var cbs = $$('.item-modelo input', b), n = cbs.filter(function (c) { return c.checked; }).length;
      var cab = $('.grupo-cab > input', b);
      cab.checked = n === cbs.length; cab.indeterminate = n > 0 && n < cbs.length;
      $('.dobra .mudo', b).textContent = n + ' de ' + cbs.length;      // recolhido, ainda diz quantos estao marcados
    });
    var total = MODELOS.filter(temDados).length;
    $('#contagem').textContent = E.sel.size + ' de ' + total;
  }
  function mudouSelecao() { sincronizaLista(); guarda(); renderAba(); }
  function trocaAba(aba, porClique) {
    E.aba = aba;
    guarda();
    $$('#abas button').forEach(function (b) { b.setAttribute('aria-selected', String(b.getAttribute('data-aba') === aba)); });
    $$('.vista').forEach(function (v) { v.hidden = v.getAttribute('data-aba') !== aba; });
    if (porClique) { try { history.replaceState(null, '', '#' + aba); } catch (e) { /* endereco fixo */ } }
    renderAba();
  }
  function renderAba() {
    if (E.pagina !== 'resultados') return;
    var v = $('#vista-' + E.aba);
    v.textContent = '';
    escondeDica();
    RENDER[E.aba](v);
    marcaLista();
  }
  // O que falta a um modelo para aparecer em cada aba (null: aparece). Marcar ou desmarcar um desses
  // nao muda a vista, e a lista diz por que, em vez de parecer que o clique nao funcionou.
  var FORA = {
    resumo: function (m) {
      return D.pares.some(function (p) { return (p.de === m.id || p.para === m.id) && POR_ID[p.de] && POR_ID[p.para]; }) ? null : 'sem par de 1 e 40 sessões';
    },
    tabela: function (m) { return m.valores ? null : 'sem métricas'; },
    graficos: function (m) {
      return m.valores && METRICAS.some(function (mt) { return mt.grupo === E.grupo && valor(m, mt.id) != null; }) ? null : 'sem métricas deste grupo';
    },
    dispersao: function (m) {
      var x = valor(m, E.x), y = valor(m, E.y);
      return x != null && y != null && (!(EXTRAS[E.x] && E.logx) || x > 0) ? null : 'sem um dos dois valores';
    },
    treino: function (m) { return m.curva && m.curva.series[E.curva] ? null : 'sem curva de treino'; },
    galeria: function (m) { return D.galeria && D.galeria.modelos[m.id] ? null : 'sem reconstruções'; },
    frr: function (m) { return m.familia === 'frr' && m.frr ? null : 'não é FRR'; },
    fichas: function () { return E.fichas === 'todos' ? 'a aba mostra todos' : E.fichas === 'planejados' ? 'a aba mostra os planejados' : null; }
  };
  function marcaLista() {
    var fora = FORA[E.aba];
    $$('.item-modelo').forEach(function (it) {
      var m = POR_ID[$('input', it).value], motivo = fora && m ? fora(m) : null;
      var meta = $('.meta', it);       // o CSS le o motivo do proprio .meta (attr() nao sobe para o pai)
      it.classList.toggle('fora', !!motivo);
      if (motivo) meta.setAttribute('data-fora', motivo); else meta.removeAttribute('data-fora');
    });
  }
  function cabecalho(titulo, texto) {
    return h('div', {class: 'vista-cab'}, h('h2', {texto: titulo}), texto ? h('p', {class: 'nota', html: texto}) : null);
  }
  function notasRodape(ms, ids) {
    var temL = ms.some(function (m) { return m.espaco_clip === 'ViT-L/14'; }) && ids.some(function (i) { return MET[i] && MET[i].grupo === 'retrieval'; });
    var temT = ms.some(function (m) { return m.retrieval && m.retrieval.fonte === 'treino'; });
    var partes = [];
    if (temL) partes.push('† MindEye1: retrieval no espaço CLIP dele (ViT-L/14, 257 × 768); os outros modelos usam o ViT-bigG/14. Não é diretamente comparável.');
    if (temT) partes.push('‡ retrieval da última época do treino (300 imagens fixas), porque o modelo ainda não foi avaliado.');
    return partes.length ? h('p', {class: 'nota', texto: partes.join(' ')}) : null;
  }

  // ------------------------------------------------------------------ vista: resumo
  function renderResumo(v) {
    var cont = {completo: 0, retrieval: 0, treino: 0};
    MODELOS.forEach(function (m) { cont[m.status] = (cont[m.status] || 0) + 1; });
    v.appendChild(cabecalho('Resumo', 'Cada modelo foi treinado nas mesmas exibições do sujeito 1 e testado nas mesmas 1.000 imagens, com as mesmas métricas. Escolha os modelos na lista ao lado; todas as abas mostram a mesma seleção.'));
    v.appendChild(h('div', {class: 'situacao'},
      h('span', {class: 'pilula'}, h('span', {class: 'ponto-status st-completo'}), plural(cont.completo || 0, 'modelo avaliado', 'modelos avaliados')),
      cont.retrieval ? h('span', {class: 'pilula'}, h('span', {class: 'ponto-status st-retrieval'}), plural(cont.retrieval, 'com só retrieval', 'com só retrieval')) : null,
      cont.treino ? h('span', {class: 'pilula'}, h('span', {class: 'ponto-status st-treino'}), plural(cont.treino, 'em treino ou sem avaliação', 'em treino ou sem avaliação')) : null,
      h('span', {class: 'pilula'}, h('span', {class: 'ponto-status st-planejado'}), plural(D.planejados.length, 'experimento planejado', 'experimentos planejados'))));

    // 1 -> 40 sessoes
    var opcoes = ['BwdRetrieval', 'FwdRetrieval', 'CLIP', 'InceptionV3', 'AlexNet(2)', 'PixCorr', 'SSIM'];
    var pares = D.pares.map(function (p) { return {nome: p.nome, a: POR_ID[p.de], b: POR_ID[p.para]}; })
      .filter(function (p) { return p.a && p.b && (E.sel.has(p.a.id) || E.sel.has(p.b.id)); });      // o par de cada modelo marcado
    var painel = h('section', {class: 'painel'});
    var mt = MET[E.resumoMet] || MET.BwdRetrieval;
    painel.appendChild(h('header', null, h('b', {texto: 'De 1 para 40 sessões'}),
      h('span', {texto: 'Cada modelo marcado na lista, treinado com ~1 h e com ~40 h de fMRI. ' + (mt.nome || mt.rotulo) + (mt.maior ? ' (maior é melhor)' : ' (menor é melhor)') + '.'})));
    painel.appendChild(h('div', {class: 'controles'}, segmentado('Métrica', opcoes.map(function (id) { return [id, MET[id].rotulo]; }), mt.id,
      function (val) { E.resumoMet = val; guarda(); renderAba(); })));
    var caixaSvg = h('div', {class: 'svg-caixa'});
    painel.appendChild(caixaSvg);
    v.appendChild(painel);
    if (pares.length) requestAnimationFrame(function () { inclinacao(caixaSvg, pares, mt); });
    else caixaSvg.appendChild(vazio('Nenhum modelo marcado tem versão de 1 e de 40 sessões.', presetBotao('principais', 'Ver os modelos principais')));
    var faltam = pares.filter(function (p) { return valor(p.a, mt.id, 'refinada') == null || valor(p.b, mt.id, 'refinada') == null; })
      .map(function (p) { return p.nome; });
    if (faltam.length) painel.appendChild(h('p', {class: 'nota', texto: 'Sem esta métrica: ' + faltam.join(', ') + '.'}));
    if (pares.some(function (p) { return !E.sel.has(p.a.id) || !E.sel.has(p.b.id); }))
      painel.appendChild(h('p', {class: 'nota', texto: 'Ponto vazado: a versão que não está marcada na lista, mostrada como referência.'}));
    poe(painel, notasRodape(pares.map(function (p) { return p.a; }), [mt.id]));      // null quando nao ha nota

    if (D.leitura && D.leitura.length) {
      v.appendChild(h('section', {class: 'secao'}, h('h3', {texto: 'Leitura do benchmark'}),
        h('p', {class: 'nota', texto: 'Escritas a partir das tabelas do benchmark (as do FRR são montadas dos números medidos). Valem para o benchmark inteiro: não mudam com a seleção.'}),
        h('ul', {class: 'conclusoes'}, D.leitura.map(function (t) { return h('li', {html: t}); }))));
    }
  }
  function inclinacao(caixa, pares, mt) {
    caixa.textContent = '';
    var W = Math.max(320, caixa.clientWidth), estreito = W < 560;
    var it = pares.map(function (p) { return {p: p, a: valor(p.a, mt.id, 'refinada'), b: valor(p.b, mt.id, 'refinada')}; })
      .filter(function (x) { return x.a != null && x.b != null; });
    if (!it.length) { caixa.appendChild(vazio('Nenhum par de modelos tem esta métrica.')); return; }
    var vals = [];
    it.forEach(function (x) { vals.push(x.a, x.b); });
    var dom = dominio(vals, {pct: mt.fmt === 'pct'});
    var H = 320, m = {t: 18, b: 34, l: estreito ? 46 : 64, r: estreito ? 140 : 250};
    var x0 = m.l + 66, x1 = W - m.r;
    var Y = function (v) { return m.t + (1 - (v - dom[0]) / (dom[1] - dom[0])) * (H - m.t - m.b); };
    var svg = sv('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Variação de ' + (mt.nome || mt.rotulo) + ' de 1 para 40 sessões'}, caixa);
    marcasEixo(dom[0], dom[1], 5).forEach(function (t) {
      if (t < dom[0] - 1e-9 || t > dom[1] + 1e-9) return;
      sv('line', {class: 'grade', x1: m.l, x2: W - m.r + 8, y1: Y(t), y2: Y(t)}, svg);
      svTexto({x: m.l - 6, y: Y(t) + 4, 'text-anchor': 'end'}, fmt(t, mt.fmt === 'pct' ? 'pctc' : 'decc'), svg);
    });
    if (mt.acaso != null && mt.acaso > dom[0] && mt.acaso < dom[1]) {
      sv('line', {class: 'acaso', x1: m.l, x2: W - m.r + 8, y1: Y(mt.acaso), y2: Y(mt.acaso)}, svg);
      svTexto({x: W - m.r + 8, y: Y(mt.acaso) - 4, 'text-anchor': 'end'}, 'acaso', svg);
    }
    svTexto({x: x0, y: H - 10, 'text-anchor': 'middle', class: 'rot'}, '1 sessão', svg);
    svTexto({x: x1, y: H - 10, 'text-anchor': 'middle', class: 'rot'}, '40 sessões', svg);
    // rotulos sem sobreposicao: desce os que colidem
    function espalha(lista, chave) {
      lista.sort(function (a, b) { return a[chave] - b[chave]; });
      for (var i = 1; i < lista.length; i++) if (lista[i][chave] - lista[i - 1][chave] < 15) lista[i][chave] = lista[i - 1][chave] + 15;
    }
    // dois pares com o mesmo valor em 1 sessao (o MindEye1 com e sem low-level): um rotulo so
    var dir = it.map(function (x) { return {x: x, y: Y(x.b)}; }), esq = [], vistos = {};
    it.forEach(function (x) {
      var txt = fmt(x.a, mt.fmt) + marcas(x.p.a, mt.id), chave = txt + '@' + Math.round(Y(x.a));
      if (vistos[chave]) return;
      vistos[chave] = true;
      esq.push({x: x, y: Y(x.a), txt: txt});
    });
    espalha(dir, 'y'); espalha(esq, 'y');
    it.forEach(function (x) {
      var cor = x.p.b.cor;
      var g = sv('g', {style: 'color:' + cor}, svg);
      sv('line', {x1: x0, x2: x1, y1: Y(x.a), y2: Y(x.b), stroke: 'currentColor', 'stroke-width': 2.2}, g);
      [[x0, x.a, x.p.a], [x1, x.b, x.p.b]].forEach(function (e) {      // ponta nao marcada na lista: vazada
        var marcado = E.sel.has(e[2].id);
        sv('circle', {cx: e[0], cy: Y(e[1]), r: marcado ? 4.5 : 3.6, fill: marcado ? 'currentColor' : 'var(--bg)',
          stroke: 'currentColor', 'stroke-width': marcado ? 0 : 1.8}, g);
      });
      g.addEventListener('pointermove', function (ev) {
        mostraDica(ev, '<b>' + esc(x.p.nome) + '</b><br>1 sessão: ' + fmt(x.a, mt.fmt) + '<br>40 sessões: ' + fmt(x.b, mt.fmt) +
          '<br>diferença: ' + diferenca(x.b - x.a, mt.fmt));
      });
      g.addEventListener('pointerleave', escondeDica);
    });
    esq.forEach(function (o) { svTexto({x: x0 - 9, y: o.y + 4, 'text-anchor': 'end', class: 'forte'}, o.txt, svg); });
    dir.forEach(function (o) {
      var t = svTexto({x: x1 + 10, y: o.y + 4, class: 'rot'}, '', svg);
      var v1 = sv('tspan', {class: 'forte'}, t); v1.textContent = fmt(o.x.b, mt.fmt) + marcas(o.x.p.b, mt.id) + '  ';
      var v2 = sv('tspan', null, t); v2.textContent = o.x.p.nome;
    });
  }

  // ------------------------------------------------------------------ vista: tabela
  function renderTabela(v) {
    v.appendChild(cabecalho('Tabela', 'Todas as métricas dos modelos selecionados. Clique no nome de uma coluna para ordenar. Em destaque, o melhor valor da coluna entre os selecionados (sem contar as reconstruções publicadas); a barrinha mostra a posição de cada valor entre o pior e o melhor. A última linha é a amplitude (maior − menor), que serve de régua quando a seleção difere só na semente.'));
    var grupos = D.grupos_metricas;
    var ctrl = h('div', {class: 'controles'}, opcoesTipo());
    var gseg = h('div', {class: 'seg', role: 'group', 'aria-label': 'Colunas'});
    grupos.forEach(function (g) {
      var b = h('button', {type: 'button', 'aria-pressed': String(E.colunas.indexOf(g.id) >= 0), texto: g.nome});
      b.addEventListener('click', function () {
        var i = E.colunas.indexOf(g.id);
        if (i >= 0) E.colunas.splice(i, 1); else E.colunas.push(g.id);
        guarda(); renderAba();
      });
      gseg.appendChild(b);
    });
    ctrl.appendChild(h('div', {class: 'controle'}, h('span', {texto: 'Colunas'}), gseg));
    var botaoCopia = h('button', {class: 'botao leve', type: 'button', texto: 'Copiar tabela'});
    ctrl.appendChild(botaoCopia);
    v.appendChild(ctrl);
    var ms = selecionados().filter(function (m) { return m.valores; });
    if (!ms.length) { v.appendChild(vazio('Nenhum modelo com métricas na seleção. Escolha modelos na lista.')); return; }
    var cols = METRICAS.filter(function (mt) {
      return E.colunas.indexOf(mt.grupo) >= 0 && ms.some(function (m) { return valor(m, mt.id) != null; });
    });
    if (E.ordem && (MET[E.ordem] || E.ordem === 'rotulo')) {
      var o = E.ordem, s = E.ordemDesc ? -1 : 1;
      ms = ms.slice().sort(function (a, b) {
        if (o === 'rotulo') return s * a.rotulo.localeCompare(b.rotulo, 'pt');
        var va = valor(a, o), vb = valor(b, o);
        if (va == null) return 1;
        if (vb == null) return -1;
        return s * (va - vb);
      });
    }
    var est = {};
    cols.forEach(function (mt) {
      var vs = ms.filter(function (m) { return !m.publicada; }).map(function (m) { return valor(m, mt.id); }).filter(function (x) { return x != null; });
      var todos = ms.map(function (m) { return valor(m, mt.id); }).filter(function (x) { return x != null; });
      est[mt.id] = {lo: Math.min.apply(null, todos), hi: Math.max.apply(null, todos),
        melhor: vs.length > 1 ? (mt.maior ? Math.max.apply(null, vs) : Math.min.apply(null, vs)) : null};
    });
    var tab = h('table', {class: 'tm'});
    var thead = h('thead'), fam = h('tr', {class: 'fam'}), col = h('tr', {class: 'col'});
    fam.appendChild(h('th', {class: 'mod'}));
    grupos.forEach(function (g) {
      var n = cols.filter(function (c) { return c.grupo === g.id; }).length;
      if (n) fam.appendChild(h('th', {colspan: n, texto: g.curto || g.nome}));
    });
    function cabCol(id, texto, titulo, seta) {
      var b = h('button', {type: 'button', title: titulo}, texto, seta ? h('span', {class: 'seta', texto: seta}) : null,
        E.ordem === id ? h('span', {class: 'seta', texto: E.ordemDesc ? ' ▾' : ' ▴'}) : null);
      b.addEventListener('click', function () {
        if (E.ordem === id) { if (E.ordemDesc) E.ordemDesc = false; else E.ordem = null; } else { E.ordem = id; E.ordemDesc = true; }
        guarda(); renderAba();
      });
      return b;
    }
    col.appendChild(h('th', {class: 'mod', scope: 'col'}, cabCol('rotulo', 'Modelo', 'Ordenar pelo nome')));
    cols.forEach(function (mt) { col.appendChild(h('th', {scope: 'col'}, cabCol(mt.id, mt.rotulo, mt.nome + ': ' + mt.desc, mt.maior ? '↑' : '↓'))); });
    thead.appendChild(fam); thead.appendChild(col); tab.appendChild(thead);
    var tbody = h('tbody');
    ms.forEach(function (m) {
      var tr = h('tr');
      var sub = [m.sessoes ? (m.sessoes === 1 ? '1 sessão' : m.sessoes + ' sessões') : null, m.status_txt].filter(Boolean).join(' · ');
      tr.appendChild(h('td', {class: 'mod'}, h('div', {class: 'nome'}, h('span', {class: 'amostra', style: '--c:' + m.cor}), h('span', {texto: m.rotulo})),
        h('small', {texto: sub})));
      cols.forEach(function (mt) {
        var x = valor(m, mt.id);
        if (x == null) { tr.appendChild(h('td', {class: 'falta', texto: '—'})); return; }
        var e = est[mt.id], p = e.hi > e.lo ? (x - e.lo) / (e.hi - e.lo) : 1;
        if (!mt.maior) p = 1 - p;
        var melhor = !m.publicada && e.melhor != null && Math.abs(x - e.melhor) < 1e-12;
        var ic = intervalo(m, mt.id);
        tr.appendChild(h('td', {class: 'v' + (melhor ? ' melhor' : ''), title: ic ? 'IC 95%: ' + fmt(ic[0], mt.fmt) + ' a ' + fmt(ic[1], mt.fmt) : null},
          fmt(x, mt.fmt), marcas(m, mt.id) ? h('span', {class: 'marca-nota', texto: marcas(m, mt.id)}) : null,
          h('span', {class: 'db'}, h('i', {style: '--p:' + Math.round(p * 100) + '%'}))));
      });
      tbody.appendChild(tr);
    });
    tab.appendChild(tbody);
    if (ms.length > 1) {
      var tf = h('tfoot'), tr = h('tr');
      tr.appendChild(h('td', {class: 'mod', texto: 'Amplitude'}));
      cols.forEach(function (mt) { var e = est[mt.id]; tr.appendChild(h('td', {texto: isFinite(e.hi - e.lo) ? diferenca(e.hi - e.lo, mt.fmt) : '—'})); });
      tf.appendChild(tr); tab.appendChild(tf);
    }
    v.appendChild(h('div', {class: 'tm-caixa'}, tab));
    poe(v, notasRodape(ms, cols.map(function (c) { return c.id; })));      // null quando nao ha nota
    if (ms.some(function (m) { return m.familia === 'frr'; })) v.appendChild(h('p', {class: 'nota', texto: 'O FRR só prevê o embedding: tem retrieval, mas não reconstrução nem legendas.'}));
    botaoCopia.addEventListener('click', function () {
      var linhas = [['Modelo', 'id'].concat(cols.map(function (c) { return c.rotulo; })).join('\t')];
      ms.forEach(function (m) { linhas.push([m.rotulo, m.id].concat(cols.map(function (c) { var x = valor(m, c.id); return x == null ? '' : fmt(x, c.fmt); })).join('\t')); });
      copia(linhas.join('\n'), botaoCopia);
    });
  }

  // ------------------------------------------------------------------ vista: graficos
  function renderGraficos(v) {
    v.appendChild(cabecalho('Gráficos', 'Um painel por métrica, com os modelos selecionados. Linha tracejada: o acaso (1 em 300 no retrieval, 50% nas identificações 2-way). Barras horizontais no retrieval: intervalo de 95% dos 30 sorteios, quando a fonte o guarda.'));
    var grupos = D.grupos_metricas;
    var ctrl = h('div', {class: 'controles'},
      segmentado('Grupo', grupos.map(function (g) { return [g.id, g.nome]; }), E.grupo, function (x) { E.grupo = x; guarda(); renderAba(); }),
      E.grupo !== 'retrieval' ? opcoesTipo() : null,
      segmentado('Eixo', [['ajustada', 'Ajustado aos valores'], ['zero', 'A partir de zero']], E.escala, function (x) { E.escala = x; guarda(); renderAba(); }),
      caixa('Ordenar por valor', E.ordenaGraf, function (x) { E.ordenaGraf = x; guarda(); renderAba(); }, 'graf-ordena'));
    v.appendChild(ctrl);
    var ms = selecionados().filter(function (m) { return m.valores; });
    if (!ms.length) { v.appendChild(vazio('Nenhum modelo com métricas na seleção.')); return; }
    var mets = METRICAS.filter(function (mt) { return mt.grupo === E.grupo; });
    var grade = h('div', {class: 'paineis'});
    var semNada = [];
    mets.forEach(function (mt) {
      var itens = ms.map(function (m) { return {m: m, v: valor(m, mt.id), ic: intervalo(m, mt.id), marca: marcas(m, mt.id),
        nota: mt.grupo === 'retrieval' && m.retrieval ? 'fonte: ' + D.fontes_retrieval[m.retrieval.fonte] : null}; })
        .filter(function (it) { return it.v != null; });
      if (!itens.length) { semNada.push(mt.rotulo); return; }
      if (E.ordenaGraf) itens.sort(function (a, b) { return mt.maior ? b.v - a.v : a.v - b.v; });
      var faltam = ms.filter(function (m) { return valor(m, mt.id) == null; }).map(function (m) { return m.rotulo; });
      grade.appendChild(h('section', {class: 'painel'},
        h('header', null, h('b', {texto: mt.nome}), h('span', {texto: (mt.maior ? '↑ maior é melhor' : '↓ menor é melhor') + ' · ' + mt.desc})),
        grafico(itens, mt, {barras: E.escala === 'zero'}),
        faltam.length ? h('p', {class: 'nota', texto: 'Sem esta métrica: ' + faltam.join(', ') + '.'}) : null));
    });
    if (grade.children.length) v.appendChild(grade);
    else v.appendChild(vazio('Nenhum modelo selecionado tem métricas deste grupo.'));
    poe(v, notasRodape(ms, mets.map(function (m) { return m.id; })));      // null quando nao ha nota
  }

  // ------------------------------------------------------------------ vista: dispersao
  function renderDispersao(v) {
    v.appendChild(cabecalho('Dispersão', 'Uma métrica contra outra, ou contra o tamanho do treino. Com o eixo x em exibições ou imagens de treino, a curva responde à pergunta do plano de estágio: quanto dado cada modelo precisa.'));
    var opcoes = [{grupo: 'Treino', itens: Object.keys(EXTRAS).map(function (k) { return [k, EXTRAS[k].nome]; })}];
    D.grupos_metricas.forEach(function (g) {
      opcoes.push({grupo: g.nome, itens: METRICAS.filter(function (m) { return m.grupo === g.id; }).map(function (m) { return [m.id, m.nome]; })});
    });
    var ctrl = h('div', {class: 'controles'},
      seletor('Eixo x', opcoes, E.x, function (x) { E.x = x; guarda(); renderAba(); }, 'disp-x'),
      seletor('Eixo y', opcoes, E.y, function (x) { E.y = x; guarda(); renderAba(); }, 'disp-y'),
      EXTRAS[E.x] ? caixa('x em escala log', E.logx, function (x) { E.logx = x; guarda(); renderAba(); }, 'disp-log') : null,
      caixa('Nomes nos pontos', E.rotulosDisp, function (x) { E.rotulosDisp = x; guarda(); renderAba(); }, 'disp-rot'),
      (MET[E.x] && MET[E.x].grupo !== 'retrieval') || (MET[E.y] && MET[E.y].grupo !== 'retrieval') ? opcoesTipo() : null);
    v.appendChild(ctrl);
    var dx = descritor(E.x), dy = descritor(E.y);
    var ms = selecionados();
    var pts = ms.map(function (m) { return {m: m, x: valor(m, E.x), y: valor(m, E.y)}; })
      .filter(function (p) { return p.x != null && p.y != null && (!(EXTRAS[E.x] && E.logx) || p.x > 0); });
    var caixaSvg = h('div', {class: 'svg-caixa'});
    v.appendChild(h('section', {class: 'painel'}, h('header', null, h('b', {texto: dy.nome + ' × ' + dx.nome}),
      h('span', {texto: (MET[E.y] ? (MET[E.y].maior ? '↑ maior é melhor' : '↓ menor é melhor') : '') + (MET[E.x] ? ' · x: ' + (MET[E.x].maior ? 'maior é melhor' : 'menor é melhor') : '')})), caixaSvg));
    var faltam = ms.filter(function (m) { return pts.every(function (p) { return p.m !== m; }); });
    if (faltam.length) v.appendChild(h('p', {class: 'nota', texto: 'Fora do gráfico (sem um dos dois valores): ' + faltam.map(function (m) { return m.rotulo; }).join(', ') + '.'}));
    poe(v, notasRodape(ms, [E.x, E.y]));      // null quando nao ha nota
    if (!pts.length) { caixaSvg.appendChild(vazio('Nenhum modelo selecionado tem os dois valores.')); return; }
    requestAnimationFrame(function () { dispersao(caixaSvg, pts, dx, dy, EXTRAS[E.x] && E.logx); });
  }
  function dispersao(caixa, pts, dx, dy, logx) {
    caixa.textContent = '';
    var W = Math.max(320, caixa.clientWidth), H = Math.round(Math.max(300, Math.min(540, W * 0.58)));
    var m = {t: 14, r: 24, b: 48, l: 62};
    var xs = pts.map(function (p) { return p.x; }), ys = pts.map(function (p) { return p.y; });
    var dxm, X, ticksX;
    if (logx) {
      var lo = Math.min.apply(null, xs), hi = Math.max.apply(null, xs);
      lo = Math.pow(10, Math.floor(Math.log10(lo) * 4) / 4 - 0.05); hi = Math.pow(10, Math.ceil(Math.log10(hi) * 4) / 4 + 0.05);
      dxm = [lo, hi];
      X = function (v) { return m.l + (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (W - m.l - m.r); };
      ticksX = marcasLog(lo, hi);
    } else {
      dxm = dominio(xs, {pct: dx.fmt === 'pct', zero: !!EXTRAS[dx.id]});
      X = function (v) { return m.l + (v - dxm[0]) / (dxm[1] - dxm[0]) * (W - m.l - m.r); };
      ticksX = marcasEixo(dxm[0], dxm[1], 5);
    }
    var dym = dominio(ys, {pct: dy.fmt === 'pct', zero: !!EXTRAS[dy.id]});
    var Y = function (v) { return m.t + (1 - (v - dym[0]) / (dym[1] - dym[0])) * (H - m.t - m.b); };
    var svg = sv('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': dy.nome + ' contra ' + dx.nome}, caixa);
    var fx = dx.fmt === 'pct' ? 'pctc' : dx.fmt === 'dec' ? 'decc' : dx.fmt, fy = dy.fmt === 'pct' ? 'pctc' : dy.fmt === 'dec' ? 'decc' : dy.fmt;
    ticksX.forEach(function (t) {
      if (t < dxm[0] - 1e-9 || t > dxm[1] + 1e-9) return;
      sv('line', {class: 'grade', x1: X(t), x2: X(t), y1: m.t, y2: H - m.b}, svg);
      svTexto({x: X(t), y: H - m.b + 16, 'text-anchor': 'middle'}, fmt(t, fx), svg);
    });
    marcasEixo(dym[0], dym[1], 5).forEach(function (t) {
      if (t < dym[0] - 1e-9 || t > dym[1] + 1e-9) return;
      sv('line', {class: 'grade', x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t)}, svg);
      svTexto({x: m.l - 6, y: Y(t) + 4, 'text-anchor': 'end'}, fmt(t, fy), svg);
    });
    if (dy.acaso != null && dy.acaso > dym[0] && dy.acaso < dym[1]) sv('line', {class: 'acaso', x1: m.l, x2: W - m.r, y1: Y(dy.acaso), y2: Y(dy.acaso)}, svg);
    if (dx.acaso != null && !logx && dx.acaso > dxm[0] && dx.acaso < dxm[1]) sv('line', {class: 'acaso', x1: X(dx.acaso), x2: X(dx.acaso), y1: m.t, y2: H - m.b}, svg);
    svTexto({x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle', class: 'rot'}, dx.nome + (logx ? ' (escala log)' : ''), svg);
    svTexto({x: 14, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', class: 'rot', transform: 'rotate(-90 14 ' + ((m.t + H - m.b) / 2) + ')'}, dy.nome, svg);
    var rotulos = [];
    pts.forEach(function (p) {
      var g = sv('g', {style: 'color:' + p.m.cor}, svg);
      sv('circle', {cx: X(p.x), cy: Y(p.y), r: 6, fill: 'currentColor', stroke: 'var(--surface)', 'stroke-width': 1.5}, g);
      g.addEventListener('pointermove', function (ev) {
        mostraDica(ev, '<b>' + esc(p.m.rotulo) + '</b><br>' + esc(dx.nome) + ': ' + fmt(p.x, dx.fmt) + '<br>' + esc(dy.nome) + ': ' + fmt(p.y, dy.fmt));
      });
      g.addEventListener('pointerleave', escondeDica);
      rotulos.push({p: p, x: X(p.x) + 9, y: Y(p.y) + 4});
    });
    if (E.rotulosDisp) {
      rotulos.sort(function (a, b) { return a.y - b.y; });
      var postos = [];
      rotulos.forEach(function (r) {
        var y = r.y;
        postos.forEach(function (o) { if (Math.abs(o.x - r.x) < 150 && Math.abs(o.y - y) < 13) y = o.y + 13; });
        r.y = y; postos.push(r);
        var alinha = r.x > W - 170 ? 'end' : 'start';
        svTexto({x: alinha === 'end' ? r.x - 18 : r.x, y: r.y, 'text-anchor': alinha, class: 'rot'}, r.p.m.rotulo, svg);
      });
    }
  }

  // ------------------------------------------------------------------ vista: treino
  function renderTreino(v) {
    v.appendChild(cabecalho('Treino', 'As curvas que o treino do MindEye2 grava a cada época (<code>metrics.csv</code>). O retrieval do treino usa sempre as mesmas 300 primeiras imagens de teste, por isso difere um pouco do das tabelas (30 sorteios de 300). O FRR não tem épocas; o MindEye1 e os checkpoints do release não gravaram curva.'));
    var ms = selecionados(), comCurva = ms.filter(function (m) { return m.curva; });
    var series = D.curvas.filter(function (c) { return comCurva.some(function (m) { return m.curva.series[c.id]; }); });
    if (!series.length) {
      v.appendChild(vazio('Nenhum modelo selecionado tem curva de treino.', presetBotao('sementes', 'Ver as sementes do Ridge 1024')));
      return;
    }
    if (!series.some(function (c) { return c.id === E.curva; })) E.curva = series[0].id;
    var ctrl = h('div', {class: 'controles'},
      seletor('Série', series.map(function (c) { return [c.id, c.nome]; }), E.curva, function (x) { E.curva = x; guarda(); renderAba(); }, 'curva-serie'),
      segmentado('Eixo x', [['epoca', 'Época'], ['fracao', 'Fração do treino']], E.eixoCurva, function (x) { E.eixoCurva = x; guarda(); renderAba(); }));
    v.appendChild(ctrl);
    var desc = D.curvas.filter(function (c) { return c.id === E.curva; })[0];
    var linhas = comCurva.filter(function (m) { return m.curva.series[E.curva]; });
    var caixaSvg = h('div', {class: 'svg-caixa'});
    var legenda = h('ul', {class: 'legenda-series'});
    v.appendChild(h('section', {class: 'painel'}, h('header', null, h('b', {texto: desc.nome}), h('span', {texto: desc.desc})), legenda, caixaSvg));
    var semCurva = ms.filter(function (m) { return !m.curva; });
    if (semCurva.length) v.appendChild(h('p', {class: 'nota', texto: 'Sem curva de treino: ' + semCurva.map(function (m) { return m.rotulo; }).join(', ') + '.'}));
    var apagados = new Set();
    linhas.forEach(function (m) {
      var li = h('li', null, h('span', {class: 'amostra', style: '--c:' + m.cor}), m.rotulo);
      li.addEventListener('click', function () {
        if (apagados.has(m.id)) apagados.delete(m.id); else apagados.add(m.id);
        li.classList.toggle('apagado'); desenha();
      });
      legenda.appendChild(li);
    });
    function desenha() { curvas(caixaSvg, linhas, desc, E.eixoCurva === 'fracao', apagados); }
    requestAnimationFrame(desenha);
  }
  function curvas(caixa, linhas, desc, fracao, apagados) {
    caixa.textContent = '';
    var W = Math.max(320, caixa.clientWidth), H = Math.round(Math.max(280, Math.min(480, W * 0.5)));
    var m = {t: 12, r: 18, b: 40, l: 60};
    var dados = linhas.map(function (mo) {
      var c = mo.curva, ys = c.series[desc.id], n = c.epocas.length;
      return {m: mo, xs: c.epocas.map(function (e) { return fracao ? e / c.epocas[n - 1] : e; }), ys: ys};
    });
    var todosX = [], todosY = [];
    dados.forEach(function (d) { todosX = todosX.concat(d.xs); todosY = todosY.concat(d.ys); });
    var dxm = fracao ? [0, 1] : [Math.min(0, Math.min.apply(null, todosX)), Math.max.apply(null, todosX)];
    var log = desc.log;
    var dym;
    if (log) {
      var pos = todosY.filter(function (y) { return y > 0; });
      dym = [Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, pos)))), Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, pos))))];
    } else dym = dominio(todosY, {pct: desc.fmt === 'pct'});
    var X = function (x) { return m.l + (x - dxm[0]) / (dxm[1] - dxm[0] || 1) * (W - m.l - m.r); };
    var Y = log ? function (y) { return m.t + (1 - (Math.log10(Math.max(y, dym[0])) - Math.log10(dym[0])) / (Math.log10(dym[1]) - Math.log10(dym[0]))) * (H - m.t - m.b); }
      : function (y) { return m.t + (1 - (y - dym[0]) / (dym[1] - dym[0])) * (H - m.t - m.b); };
    var svg = sv('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': desc.nome + ' por época'}, caixa);
    (fracao ? [0, .25, .5, .75, 1] : marcasEixo(dxm[0], dxm[1], 6)).forEach(function (t) {
      if (t < dxm[0] - 1e-9 || t > dxm[1] + 1e-9) return;
      sv('line', {class: 'grade', x1: X(t), x2: X(t), y1: m.t, y2: H - m.b}, svg);
      svTexto({x: X(t), y: H - m.b + 16, 'text-anchor': 'middle'}, fracao ? Math.round(t * 100) + '%' : String(t), svg);
    });
    var fy = desc.fmt === 'pct' ? 'pctc' : 'decc';
    (log ? marcasLog(dym[0], dym[1]) : marcasEixo(dym[0], dym[1], 5)).forEach(function (t) {
      if (t < dym[0] - 1e-12 || t > dym[1] * 1.0001) return;
      sv('line', {class: 'grade', x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t)}, svg);
      svTexto({x: m.l - 6, y: Y(t) + 4, 'text-anchor': 'end'}, log ? t.toExponential(0).replace('e', '·10^') : fmt(t, fy), svg);
    });
    svTexto({x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', class: 'rot'}, fracao ? 'Fração do treino' : 'Época', svg);
    dados.forEach(function (d) {
      if (apagados.has(d.m.id)) return;
      var pts = d.xs.map(function (x, i) { return X(x).toFixed(1) + ',' + Y(d.ys[i]).toFixed(1); }).join(' ');
      sv('polyline', {points: pts, fill: 'none', stroke: d.m.cor, 'stroke-width': 2, 'stroke-linejoin': 'round'}, svg);
      var u = d.xs.length - 1;
      sv('circle', {cx: X(d.xs[u]), cy: Y(d.ys[u]), r: 3.5, fill: d.m.cor}, svg);
    });
    var guia = sv('line', {class: 'eixo', y1: m.t, y2: H - m.b, visibility: 'hidden'}, svg);
    var alvo = sv('rect', {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent'}, svg);
    alvo.addEventListener('pointermove', function (ev) {
      var r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
      var xv = dxm[0] + (px - m.l) / (W - m.l - m.r) * (dxm[1] - dxm[0]);
      guia.setAttribute('x1', px); guia.setAttribute('x2', px); guia.setAttribute('visibility', 'visible');
      var html = [];
      dados.forEach(function (d) {
        if (apagados.has(d.m.id)) return;
        var melhor = 0;
        d.xs.forEach(function (x, i) { if (Math.abs(x - xv) < Math.abs(d.xs[melhor] - xv)) melhor = i; });
        html.push('<span style="color:' + d.m.cor + '">●</span> ' + esc(d.m.rotulo) + ': <b>' + fmt(d.ys[melhor], desc.fmt === 'pct' ? 'pct' : 'dec') + '</b> (época ' + d.m.curva.epocas[melhor] + ')');
      });
      mostraDica(ev, html.join('<br>'));
    });
    alvo.addEventListener('pointerleave', function () { guia.setAttribute('visibility', 'hidden'); escondeDica(); });
  }
  function presetBotao(id, texto) {
    var p = D.presets.filter(function (x) { return x.id === id; })[0];
    if (!p) return null;
    var b = h('button', {class: 'botao', type: 'button', texto: texto});
    b.addEventListener('click', function () { E.sel = new Set(p.modelos); mudouSelecao(); });
    return b;
  }

  // ------------------------------------------------------------------ vista: galeria
  var TAMANHOS = {p: 96, m: 136, g: 176};
  function tiraDe(m, tipo) {
    var G = D.galeria, t = G && G.modelos[m.id];
    if (!t) return null;
    return t[tipo] || (m.familia === 'mindeye1' ? t.unclip : null);
  }
  function mini(url, j, n, rotulo) {
    if (!url) return h('div', {class: 'mini sem', texto: rotulo || 'sem esta reconstrução'});
    return h('div', {class: 'mini', role: 'img', 'aria-label': rotulo || '',
      style: "background-image:url('" + url + "');background-size:" + (n * 100) + '% 100%;background-position:' + (n > 1 ? (j / (n - 1) * 100).toFixed(4) : 0) + '% 0'});
  }
  function renderGaleria(v) {
    var G = D.galeria;
    v.appendChild(cabecalho('Galeria', G ? (G.n + ' imagens de teste sorteadas (semente 0), as mesmas para todos os modelos. Embaixo da imagem vista, uma legenda humana do COCO; embaixo de cada reconstrução, a legenda que o modelo previu do cérebro. Clique numa linha para ampliar.') : null));
    if (!G) { v.appendChild(vazio('A galeria ainda não foi gerada: rode python src/make_site.py numa máquina com os tensores de avaliação.')); return; }
    var ctrl = h('div', {class: 'controles'},
      segmentado('Reconstrução', [['refinada', 'Refinadas'], ['unclip', 'unCLIP'], ['borrada', 'Borradas']], E.galTipo, function (x) { E.galTipo = x; guarda(); renderAba(); }),
      segmentado('Tamanho', [['p', 'P'], ['m', 'M'], ['g', 'G']], E.galTam, function (x) { E.galTam = x; guarda(); renderAba(); }),
      caixa('Legendas', E.galLeg, function (x) { E.galLeg = x; guarda(); renderAba(); }, 'gal-leg'));
    v.appendChild(ctrl);
    var ms = selecionados().filter(function (m) { return G.modelos[m.id]; });
    var sem = selecionados().filter(function (m) { return !G.modelos[m.id]; });
    if (!ms.length) { v.appendChild(vazio('Nenhum modelo selecionado tem reconstruções (o FRR só prevê o embedding).', presetBotao('principais', 'Ver os modelos principais'))); return; }
    var cel = TAMANHOS[E.galTam] || 136;
    var gal = h('div', {class: 'gal' + (E.galLeg ? '' : ' sem-legendas'), style: '--ncol:' + ms.length + ';--cel:' + cel + 'px;--col-vista:' + Math.max(150, cel + 20) + 'px'});
    gal.appendChild(h('div', {class: 'gal-cab vista', texto: 'Imagem vista'}));
    ms.forEach(function (m) { gal.appendChild(h('div', {class: 'gal-cab'}, h('span', {class: 'amostra', style: '--c:' + m.cor}), h('span', {texto: m.rotulo}))); });
    for (var j = 0; j < G.n; j++) {
      (function (j) {
        var id = 'teste #' + G.idx[j] + (G.nsd ? ' · NSD ' + G.nsd[j] : '');
        var fv = h('figure', {class: 'vista'}, mini(G.vista, j, G.n, 'imagem vista, ' + id),
          h('figcaption', null, h('span', {class: 'id', texto: id}), G.coco ? h('span', {class: 'txt', texto: G.coco[j]}) : null));
        fv.addEventListener('click', function () { abreLupa(j, ms); });
        gal.appendChild(fv);
        ms.forEach(function (m) {
          var leg = G.legendas && G.legendas[m.id] ? G.legendas[m.id][j] : null;
          var f = h('figure', null, mini(tiraDe(m, E.galTipo), j, G.n, m.rotulo + ', ' + id),
            leg ? h('figcaption', null, h('span', {class: 'txt', texto: leg})) : null);
          f.addEventListener('click', function () { abreLupa(j, ms); });
          gal.appendChild(f);
        });
      })(j);
    }
    v.appendChild(h('div', {class: 'gal-caixa'}, gal));
    var notas = [];
    if (ms.some(function (m) { return m.familia === 'mindeye1'; })) notas.push('O MindEye1 não tem refinamento: a reconstrução final dele (Versatile Diffusion, com img2img a partir da borrada quando há low-level) aparece em Refinadas e em unCLIP.');
    if (E.galTipo === 'borrada') notas.push('Só os modelos com ramo blurry têm a reconstrução borrada.');
    if (sem.length) notas.push('Sem reconstruções na galeria: ' + sem.map(function (m) { return m.rotulo; }).join(', ') + '.');
    if (notas.length) v.appendChild(h('p', {class: 'nota', texto: notas.join(' ')}));
  }
  var lupa = $('#lupa');
  function abreLupa(j, ms) {
    var G = D.galeria;
    if (!lupa || !lupa.showModal) return;
    var corpo = $('.lupa-corpo', lupa);
    corpo.textContent = '';
    var fechar = h('button', {class: 'botao leve', type: 'button', texto: 'Fechar'});
    fechar.addEventListener('click', function () { lupa.close(); });
    corpo.appendChild(h('div', {class: 'lupa-cab'}, h('b', {texto: 'Teste #' + G.idx[j] + (G.nsd ? ' · NSD ' + G.nsd[j] : '')}), fechar));
    var grade = h('div', {class: 'lupa-grade'});
    grade.appendChild(h('figure', null, mini(G.vista, j, G.n, 'imagem vista'), h('figcaption', null, h('b', {texto: 'Imagem vista'}), G.coco ? G.coco[j] : '')));
    ms.forEach(function (m) {
      var leg = G.legendas && G.legendas[m.id] ? G.legendas[m.id][j] : '';
      grade.appendChild(h('figure', null, mini(tiraDe(m, E.galTipo), j, G.n, m.rotulo), h('figcaption', null, h('b', {texto: m.rotulo}), leg)));
    });
    corpo.appendChild(grade);
    lupa.showModal();
  }
  if (lupa) lupa.addEventListener('click', function (ev) { if (ev.target === lupa) lupa.close(); });

  // ------------------------------------------------------------------ vista: FRR
  function renderFrr(v) {
    v.appendChild(cabecalho('FRR', 'Regressão ridge fracionária dos 15.724 voxels direto para o embedding CLIP achatado (256 × 1.664 = 425.984 números), com uma fração escolhida por validação cruzada para cada dimensão do alvo (Doerig et al., 2025). Sem rede e sem reconstrução: o que dá para comparar com os outros modelos é o retrieval.'));
    var fs = selecionados().filter(function (m) { return m.familia === 'frr' && m.frr; });
    if (!fs.length) { v.appendChild(vazio('Nenhum FRR na seleção.', presetBotao('frr', 'Ver as seis variantes do FRR'))); return; }
    var cols = [
      ['Img→cér', function (f, m) { return m.retrieval.fwd.media; }, 'pct'], ['Cér→img', function (f, m) { return m.retrieval.bwd.media; }, 'pct'],
      ['Img→cér centrado', function (f, m) { return m.retrieval_centrado ? m.retrieval_centrado.fwd.media : null; }, 'pct'],
      ['Cér→img centrado', function (f, m) { return m.retrieval_centrado ? m.retrieval_centrado.bwd.media : null; }, 'pct'],
      ['Cosseno', function (f) { return f.embedding && f.embedding.cosseno_medio; }, 'dec'], ['Pearson', function (f) { return f.embedding && f.embedding.pearson_medio; }, 'dec'],
      ['Cosseno centrado', function (f) { return f.embedding && f.embedding.cosseno_centrado_medio; }, 'dec'],
      ['R² da CV', function (f) { return f.cv && f.cv.cv_r2_escolhida; }, 'dec'],
      ['Exibições', function (f) { return f.config && f.config.exibicoes_treino; }, 'int'],
      ['Coeficientes', function (f) { return f.parametros_lineares; }, 'mi'],
      ['Ajuste', function (f) { return f.custo && f.custo.segundos_ajuste; }, 'dur'],
      ['Pico GPU', function (f) { return f.custo && f.custo.pico_gpu_gib; }, 'gib'], ['Pico RAM', function (f) { return f.custo && f.custo.pico_ram_gib; }, 'gib']];
    var tab = h('table', {class: 'tm'});
    var tr = h('tr', {class: 'col'}, h('th', {class: 'mod', texto: 'Variante'}));
    cols.forEach(function (c) { tr.appendChild(h('th', {texto: c[0]})); });
    tab.appendChild(h('thead', null, tr));
    var tb = h('tbody');
    fs.forEach(function (m) {
      var linha = h('tr', null, h('td', {class: 'mod'}, h('div', {class: 'nome'}, h('span', {class: 'amostra', style: '--c:' + m.cor}), h('span', {texto: m.rotulo}))));
      cols.forEach(function (c) {
        var x = c[1](m.frr, m);
        var t = x == null ? '—' : c[2] === 'mi' ? mi(x) : c[2] === 'dur' ? duracao(x) : c[2] === 'gib' ? virg(x, 1) + ' GiB' : fmt(x, c[2]);
        linha.appendChild(h('td', {texto: t}));
      });
      tb.appendChild(linha);
    });
    tab.appendChild(tb);
    v.appendChild(h('div', {class: 'tm-caixa'}, tab));
    v.appendChild(h('p', {class: 'nota', texto: D.textos.frr_hubness}));
    // histogramas das fracoes escolhidas
    var grade = h('div', {class: 'paineis'});
    fs.forEach(function (m) {
      var cv = m.frr.cv;
      if (!cv) return;
      var tot = cv.histograma_fracoes.reduce(function (a, b) { return a + b; }, 0) || 1;
      var itens = cv.fracs.map(function (fr, i) { return {m: m, v: cv.histograma_fracoes[i] / tot, fr: fr}; });
      var g = h('div', {class: 'gb'});
      itens.forEach(function (it) {
        g.appendChild(h('div', {class: 'gb-rot mono', texto: String(it.fr).replace('.', ',')}));
        var tri = h('div', {class: 'gb-trilho', style: '--c:' + m.cor});
        tri.appendChild(h('span', {class: 'gb-barra', style: 'width:' + (it.v * 100).toFixed(2) + '%'}));
        tri.appendChild(h('span', {class: 'gb-val' + (it.v > .8 ? ' dentro' : ''), style: 'left:' + (it.v * 100).toFixed(2) + '%', texto: virg(it.v * 100, 1) + '%'}));
        g.appendChild(tri);
      });
      grade.appendChild(h('section', {class: 'painel'}, h('header', null, h('b', {texto: m.rotulo}),
        h('span', {texto: 'Fração escolhida em cada dimensão do alvo (% das 425.984 dimensões).'})), g));
    });
    v.appendChild(h('h3', {texto: 'Frações escolhidas'}));
    v.appendChild(grade);
    // R2 da CV por fracao
    var caixaSvg = h('div', {class: 'svg-caixa'});
    v.appendChild(h('section', {class: 'painel'}, h('header', null, h('b', {texto: 'R² da validação cruzada por fração'}),
      h('span', {texto: 'Com uma fração só para todas as dimensões; escala log em x. Valores abaixo de −2× o pico ficam cortados.'})), caixaSvg));
    requestAnimationFrame(function () { r2(caixaSvg, fs); });
  }
  function r2(caixa, fs) {
    caixa.textContent = '';
    var W = Math.max(320, caixa.clientWidth), H = 300, m = {t: 12, r: 18, b: 40, l: 60};
    var series = fs.filter(function (f) { return f.frr.cv; }).map(function (f) { return {m: f, xs: f.frr.cv.fracs, ys: f.frr.cv.cv_r2_por_fracao}; });
    var topo = Math.max.apply(null, series.map(function (s) { return Math.max.apply(null, s.ys); }));
    var piso = Math.max(Math.min.apply(null, series.map(function (s) { return Math.min.apply(null, s.ys); })), -2 * Math.max(topo, 1e-6));
    var dom = dominio([piso, topo, 0], {});
    var xlo = Math.min.apply(null, series.map(function (s) { return s.xs[0]; })), xhi = 1;
    var X = function (x) { return m.l + (Math.log10(x) - Math.log10(xlo)) / (Math.log10(xhi) - Math.log10(xlo)) * (W - m.l - m.r); };
    var Y = function (y) { return m.t + (1 - (Math.max(y, dom[0]) - dom[0]) / (dom[1] - dom[0])) * (H - m.t - m.b); };
    var svg = sv('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'R² da validação cruzada por fração'}, caixa);
    marcasLog(xlo, xhi).forEach(function (t) {
      sv('line', {class: 'grade', x1: X(t), x2: X(t), y1: m.t, y2: H - m.b}, svg);
      svTexto({x: X(t), y: H - m.b + 16, 'text-anchor': 'middle'}, String(t).replace('.', ','), svg);
    });
    marcasEixo(dom[0], dom[1], 5).forEach(function (t) {
      if (t < dom[0] - 1e-9 || t > dom[1] + 1e-9) return;
      sv('line', {class: t === 0 ? 'eixo' : 'grade', x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t)}, svg);
      svTexto({x: m.l - 6, y: Y(t) + 4, 'text-anchor': 'end'}, fmt(t, 'decc'), svg);
    });
    svTexto({x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', class: 'rot'}, 'Fração', svg);
    series.forEach(function (s) {
      var pts = s.xs.map(function (x, i) { return X(x).toFixed(1) + ',' + Y(s.ys[i]).toFixed(1); }).join(' ');
      sv('polyline', {points: pts, fill: 'none', stroke: s.m.cor, 'stroke-width': 2}, svg);
      var ib = 0;
      s.ys.forEach(function (y, i) { if (y > s.ys[ib]) ib = i; });
      var c = sv('circle', {cx: X(s.xs[ib]), cy: Y(s.ys[ib]), r: 4, fill: s.m.cor}, svg);
      c.addEventListener('pointermove', function (ev) { mostraDica(ev, '<b>' + esc(s.m.rotulo) + '</b><br>melhor fração ' + String(s.xs[ib]).replace('.', ',') + ': R² ' + virg(s.ys[ib], 3)); });
      c.addEventListener('pointerleave', escondeDica);
    });
  }

  // ------------------------------------------------------------------ vista: fichas
  var NOME_FAM = {};
  D.familias.forEach(function (f) { NOME_FAM[f.id] = f.nome; });
  function pyval(v) {
    if (v === null || v === undefined) return 'None';
    if (v === true) return 'True';
    if (v === false) return 'False';
    if (typeof v === 'string') return '"' + v + '"';
    if (Array.isArray(v)) return '[' + v.map(pyval).join(', ') + ']';
    if (typeof v === 'object') return '{' + Object.keys(v).map(function (k) { return '"' + k + '": ' + pyval(v[k]); }).join(', ') + '}';
    return String(v);
  }
  function kv(pares) {
    var dl = h('dl', {class: 'kv'});
    pares.forEach(function (p) {
      if (p == null) return;
      dl.appendChild(h('dt', {texto: p[0]}));
      dl.appendChild(h('dd', {class: p[2] ? 'alterado' : null, title: p[2] ? 'diferente do padrão' : null}, p[1]));
    });
    return dl;
  }
  function blocoConfig(cfg, descritiva) {
    if (!cfg) return h('p', {class: 'nota', texto: 'Configuração não registrada: o modelo não passou pelo treina.py nem está em src/experimentos.py.'});
    var padH = D.padroes.hiper[cfg.modelo] || {}, padD = D.padroes.dataset;
    var hp = Object.keys(cfg.hiper || {}).map(function (k) { return [k, pyval(cfg.hiper[k]), !descritiva && k in padH && pyval(padH[k]) !== pyval(cfg.hiper[k])]; });
    var ds = Object.keys(cfg.dataset || {}).filter(function (k) { return k === 'sessoes' || !(k in padD) || pyval(padD[k]) !== pyval(cfg.dataset[k]); })
      .map(function (k) { return [k, pyval(cfg.dataset[k]), k !== 'sessoes' || cfg.dataset[k] !== 40]; });
    return h('div', null, h('h4', {texto: 'Hiperparâmetros (' + cfg.modelo + ')'}), kv(hp),
      h('h4', {texto: 'Dataset', style: 'margin-top:10px'}), kv(ds),
      ds.length <= 1 ? h('p', {class: 'nota', texto: 'Os demais campos do dataset estão no padrão: as exibições completas das N primeiras sessões.'}) : null);
  }
  function resumoDataset(r) {
    if (!r) return null;
    var reps = r.repeticoes ? Object.keys(r.repeticoes).map(function (k) { return milhar(r.repeticoes[k]) + ' × ' + k; }).join(', ') : '';
    var txt = plural(r.exibicoes, 'exibição', 'exibições') + ' de ' + plural(r.imagens_unicas, 'imagem', 'imagens');
    if (reps) txt += ' (imagens × exibições: ' + reps + ')';
    if (r.imagens_por_classe) txt += '; por classe: ' + Object.keys(r.imagens_por_classe).map(function (k) { return k + ' ' + milhar(r.imagens_por_classe[k]); }).join(', ');
    return txt;
  }
  function ficha(m) {
    var ret = m.retrieval;
    var p = m.params, treinado = null;
    if (m.familia === 'frr' && m.frr) treinado = mi(m.frr.parametros_lineares) + ' coeficientes lineares, forma fechada';
    else if (p) {
      var tot = Object.keys(p).reduce(function (a, k) { return a + p[k]; }, 0);
      treinado = m.familia === 'mindeye2' && p.ridge ? 'só a ridge: ' + mi(p.ridge) + ' de ' + mi(tot) + ' (' + virg(p.ridge / tot * 100, 2) + '%)'
        : (m.familia === 'mindeye1' ? 'tudo, do zero: ' : 'tudo: ') + mi(tot);
    }
    var cmd = m.grupo === 'benchmark' && D.benchmark_ids.indexOf(m.id) >= 0 ?
      'import treina\nfrom experimentos import BENCHMARK\ntreina.roda("novo_nome", BENCHMARK["' + m.id + '"])' :
      (m.config && m.config_fonte && m.config_fonte.indexOf('experimentos') >= 0 && m.id.indexOf('__') < 0 ? 'python src/treina.py ' + m.id : null);
    var receita = m.receita ? [m.receita.treino, m.receita.partida ? 'a partir de ' + m.receita.partida : null, m.receita.prior].filter(Boolean).join(' · ') : null;
    return h('article', {class: 'ficha', style: '--c:' + m.cor},
      h('div', {class: 'ficha-cab'},
        h('div', {class: 'linha'}, h('b', {texto: m.rotulo}), h('span', {class: 'pilula'}, h('span', {class: 'ponto-status st-' + m.status}), m.status_txt)),
        h('code', {texto: m.id})),
      kv([['Modelo', NOME_FAM[m.familia] || m.familia],
          ['Sessões', m.sessoes != null ? String(m.sessoes) : '—'],
          ['Treino', resumoDataset(m.resumo_dataset)],
          receita ? ['Receita', receita] : null,
          treinado ? ['Parâmetros', treinado] : null,
          ['Tempo de treino', m.tempo_s != null ? duracao(m.tempo_s) : '—'],
          ret ? ['Retrieval', 'Img→cér ' + fmt(ret.fwd.media, 'pct') + ' · Cér→img ' + fmt(ret.bwd.media, 'pct') + ' (' + D.fontes_retrieval[ret.fonte] + ')'] : null,
          ['Espaço CLIP', m.espaco_clip],
          m.commit ? ['Commit', h('code', {texto: String(m.commit).slice(0, 12)})] : null]),
      m.notas && m.notas.length ? h('ul', null, m.notas.map(function (n) { return h('li', {texto: n}); })) : null,
      h('details', null, h('summary', {texto: 'Configuração' + (m.config_fonte ? ' (de ' + m.config_fonte + ')' : '')}), blocoConfig(m.config, m.config_descritiva),
        m.config_erro ? h('p', {class: 'nota', texto: 'O treina.py recusaria esta configuração: ' + m.config_erro}) : null),
      m.fontes && m.fontes.length ? h('details', null, h('summary', {texto: 'Arquivos de resultado'}), h('ul', null, m.fontes.map(function (f) { return h('li', null, h('code', {texto: f})); }))) : null,
      cmd ? h('details', null, h('summary', {texto: 'Treinar de novo'}), h('pre', null, h('code', {texto: cmd}))) : null);
  }
  function fichaPlanejada(p) {
    return h('article', {class: 'ficha'},
      h('div', {class: 'ficha-cab'}, h('div', {class: 'linha'}, h('b', {texto: p.rotulo}), h('span', {class: 'pilula'}, h('span', {class: 'ponto-status st-planejado'}), 'planejado')),
        h('code', {texto: p.id})),
      kv([['Modelo', NOME_FAM[p.familia] || p.familia]]),
      blocoConfig(p.config, false),
      p.config_erro ? h('p', {class: 'nota', texto: 'O treina.py recusaria esta configuração: ' + p.config_erro}) : null,
      h('pre', null, h('code', {texto: 'python src/treina.py ' + p.id + ' --so_dataset   # confere o dataset\npython src/treina.py ' + p.id})));
  }
  function renderFichas(v) {
    v.appendChild(cabecalho('Fichas', 'A configuração completa de cada modelo, de onde veio cada número e como treinar de novo. Em destaque, os valores diferentes do padrão.'));
    v.appendChild(h('div', {class: 'controles'}, segmentado('Mostrar', [['selecionados', 'Selecionados'], ['todos', 'Todos'], ['planejados', 'Planejados (' + D.planejados.length + ')']],
      E.fichas, function (x) { E.fichas = x; guarda(); renderAba(); })));
    var grade = h('div', {class: 'fichas'});
    if (E.fichas === 'planejados') {
      if (!D.planejados.length) { v.appendChild(vazio('Nenhum experimento planejado: todos os de src/experimentos.py já têm resultado.')); return; }
      D.planejados.forEach(function (p) { grade.appendChild(fichaPlanejada(p)); });
    } else {
      var ms = E.fichas === 'todos' ? MODELOS : selecionados();
      if (!ms.length) { v.appendChild(vazio('Nenhum modelo selecionado.')); return; }
      ms.forEach(function (m) { grade.appendChild(ficha(m)); });
    }
    v.appendChild(grade);
    $$('pre', grade).forEach(function (pre) {
      var b = h('button', {class: 'copiar', type: 'button', texto: 'Copiar'});
      b.addEventListener('click', function () { copia(pre.innerText, b); });
      var cx = h('div', {class: 'cod'});
      pre.parentNode.insertBefore(cx, pre); cx.appendChild(pre); cx.appendChild(b);
    });
  }

  var RENDER = {resumo: renderResumo, tabela: renderTabela, graficos: renderGraficos, dispersao: renderDispersao,
                treino: renderTreino, galeria: renderGaleria, frr: renderFrr, fichas: renderFichas};

  // ------------------------------------------------------------------ simulador de dataset (Como usar)
  var simIniciado = false;
  function iniciaSimulador() {
    if (simIniciado) return;
    simIniciado = true;
    var caixaSim = $('#simulador');
    if (!caixaSim) return;
    if (!D.semantica || !window.Simulador) {
      caixaSim.textContent = '';
      caixaSim.appendChild(vazio('O simulador precisa dos dados semânticos, que ainda não foram gerados: rode python src/make_site.py numa máquina com os dados do MindEye2 e a tabela de frações da tela.'));
      return;
    }
    var dec = Simulador.decodifica(D.semantica);
    var nomes = D.semantica.nomes;
    var chipsSup = $('#sim-sup'), chipsPri = $('#sim-pri');
    D.semantica.sup.forEach(function (s) {
      chipsSup.appendChild(h('label', {title: 'sup_' + s}, h('input', {type: 'checkbox', value: s, checked: s === 'animal' ? true : null}), nomes[s] || s));
    });
    Simulador.PRIORIDADE.forEach(function (c) { chipsPri.appendChild(h('label', null, h('input', {type: 'checkbox', value: c, checked: true}), c)); });
    var form = $('#sim-form');
    function atualiza() {
      var regra = $('input[name="sim-regra"]:checked', form).value;
      $('#sim-exclusiva').hidden = regra !== 'exclusiva';
      $('#sim-prioridade').hidden = regra !== 'prioridade';
      $('#sim-sessoes-v').textContent = $('#sim-sessoes').value;
      var cfg = {sessoes: +$('#sim-sessoes').value};
      if (regra === 'exclusiva') {
        var sups = $$('input:checked', chipsSup).map(function (c) { return c.value; });
        if (sups.length) {
          cfg.classes = {};
          sups.forEach(function (s) { cfg.classes[(nomes[s] || s)] = ['sup_' + s]; });
          cfg.minimo = +$('#sim-minimo').value;
          cfg.maximo_outros = +$('#sim-maximo').value;
        }
      } else {
        cfg.regra = 'prioridade';
        var pri = $$('input:checked', chipsPri).map(function (c) { return c.value; });
        if (pri.length < 3) cfg.classes = pri;
      }
      var rep = $('#sim-rep').value;
      if (rep) cfg.repeticoes = +rep;
      var n = $('#sim-nimg').value;
      if (n) cfg.n_imagens = +n;
      if ($('#sim-bal').checked) cfg.balanceado = true;
      var ag = $('#sim-ag').value;
      if (ag) cfg.agregacao = ag;
      mostraSimulacao(cfg, Simulador.simula(dec, cfg));
    }
    form.addEventListener('input', atualiza);
    form.addEventListener('change', atualiza);
    form.addEventListener('submit', function (ev) { ev.preventDefault(); });
    atualiza();
  }
  function mostraSimulacao(cfg, r) {
    var saida = $('#sim-res');
    saida.textContent = '';
    var codigo = '"dataset": ' + pyval(cfg);
    var pre = h('pre', null, h('code', {texto: codigo}));
    var b = h('button', {class: 'copiar', type: 'button', texto: 'Copiar'});
    b.addEventListener('click', function () { copia(codigo, b); });
    if (r.erro) {
      saida.appendChild(h('p', {class: 'sim-erro', texto: 'O treina.py recusaria: ' + r.erro + '.'}));
      saida.appendChild(h('div', {class: 'cod'}, pre, b));
      return;
    }
    var aprox = r.exato ? '' : '≈ ';
    var reps = r.repeticoes ? Object.keys(r.repeticoes).map(function (k) { return milhar(r.repeticoes[k]) + ' com ' + k; }).join(' · ') : 'decidido no sorteio';
    saida.appendChild(h('dl', {class: 'sim-numeros'},
      h('div', null, h('dt', {texto: 'Imagens únicas'}), h('dd', {texto: milhar(r.imagens_unicas)})),
      h('div', null, h('dt', {texto: 'Exibições'}), h('dd', {texto: aprox + milhar(r.exibicoes)})),
      h('div', null, h('dt', {texto: 'Do pool de'}), h('dd', {texto: milhar(r.pool.exibicoes)}))));
    saida.appendChild(h('p', {class: 'nota', texto: 'Exibições por imagem: ' + reps + '. Pool: as ' + milhar(r.pool.exibicoes) + ' exibições de ' + milhar(r.pool.imagens) + ' imagens das ' + r.sessoes + ' primeiras sessões. ' +
      (cfg.n_imagens ? 'Elegíveis antes do sorteio: ' + milhar(r.elegiveis) + ' imagens; quais entram, o treina.py sorteia com a semente.' : '')}));
    var cls = r.imagens_por_classe || r.elegiveis_por_classe;
    if (cls && Object.keys(cls).length) {
      var mx = Math.max.apply(null, Object.keys(cls).map(function (k) { return cls[k]; })) || 1;
      var lista = h('div', {class: 'sim-classes'}, h('h4', {texto: r.imagens_por_classe ? 'Imagens por classe' : 'Elegíveis por classe (antes do sorteio)'}));
      Object.keys(cls).forEach(function (k) {
        lista.appendChild(h('div', {class: 'sim-classe'}, h('span', {texto: k}),
          h('span', {class: 'barra'}, h('i', {style: '--p:' + (cls[k] / mx * 100).toFixed(1) + '%'})), h('span', {class: 'num', texto: milhar(cls[k])})));
      });
      saida.appendChild(lista);
    }
    var am = Simulador.amostras(r, cfg);
    var t = h('table', {class: 'tab'}, h('thead', null, h('tr', null, h('th', {texto: 'Modelo'}), h('th', {texto: 'Amostras por época'}), h('th', {texto: 'Lotes por época'}))));
    var tb = h('tbody');
    [['FRR', am.frr, null], ['MindEye2 (batch 16; 8 no 4096 + blurry)', am.mindeye2, [16, 8]], ['MindEye1 (batch 16)', am.mindeye1, [16]]].forEach(function (l) {
      var a = l[1];
      tb.appendChild(h('tr', null, h('td', {texto: l[0]}),
        h('td', {class: 'num', texto: a.erro ? a.erro : aprox + milhar(a.amostras) + (l[0] === 'FRR' ? ' linhas (sem épocas)' : '')}),
        h('td', {class: 'num', texto: a.erro || !l[2] ? '—' : l[2].map(function (b) { return aprox + milhar(a.lotes[b]); }).join(' / ')})));
    });
    t.appendChild(tb);
    saida.appendChild(h('div', {class: 'rolagem'}, t));
    saida.appendChild(h('h4', {texto: 'No dicionário do experimento'}));
    saida.appendChild(h('div', {class: 'cod'}, pre, b));
  }

  // ------------------------------------------------------------------ partida
  rota();
  primeira = false;
})();
