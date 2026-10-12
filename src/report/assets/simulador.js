/* Simulador do dataset controlado: refaz, no navegador, a conta do dataset_controlado.monta
   (src/mindeye_ridge) com os dados que report/semantica_site.py grava (sessoes de cada exibicao e
   fracao da tela de cada supercategoria, por imagem de treino). Conta imagens, exibicoes e amostras
   por epoca; o sorteio em si (n_imagens) so o treina.py faz. A pagina e os testes (node) usam este
   mesmo arquivo. */
(function (raiz) {
  'use strict';
  var SEM = 255, SEM_FRACAO = 0xFFFFFF;
  var OUTROS_VALIDOS = ['vehicle', 'furniture', 'outdoor'];   // semantica.rotulo_prioridade
  var PRIORIDADE = ['pessoa', 'animal', 'outro'];

  function binario(b64) {
    if (typeof atob === 'function') return atob(b64);
    return Buffer.from(b64, 'base64').toString('binary');
  }

  // a fracao da tela exatamente como semantica.calcula_fracoes a calcula (100 * a media da mascara)
  function fracao(pixels, tela) { return 100 * (pixels / tela); }

  function decodifica(d) {
    var bin = binario(d.dados), n = d.n, k = d.sup.length, passo = 3 + 3 * k, tela = d.tela || 180625;
    var sess = new Uint8Array(n * 3), frac = new Float64Array(n * k), fora = new Uint8Array(n);
    for (var i = 0; i < n; i++) {
      var o = i * passo;
      for (var j = 0; j < 3; j++) sess[i * 3 + j] = bin.charCodeAt(o + j);
      for (j = 0; j < k; j++) {
        var b = o + 3 + 3 * j;
        var v = bin.charCodeAt(b) | (bin.charCodeAt(b + 1) << 8) | (bin.charCodeAt(b + 2) << 16);
        if (v === SEM_FRACAO) { fora[i] = 1; v = 0; }
        frac[i * k + j] = fracao(v, tela);                    // % da tela
      }
    }
    var col = {};
    d.sup.forEach(function (s, j) { col[s] = j; });
    return {n: n, k: k, sup: d.sup, col: col, sess: sess, frac: frac, fora: fora};
  }

  // regra exclusiva (semantica.rotulo_exclusivo): a classe ocupa >= minimo e cada outra classe
  // listada fica abaixo de maximo_outros; como no Python, se mais de uma passar, fica a ultima
  function rotuloExclusivo(dec, i, classes, minimo, maximo) {
    if (dec.fora[i]) return null;
    var f = classes.map(function (c) {
      var s = 0;
      c.cols.forEach(function (j) { s += dec.frac[i * dec.k + j]; });
      return Math.min(100, s);
    });
    var rot = null;
    for (var a = 0; a < classes.length; a++) {
      if (f[a] < minimo) continue;
      var ok = true;
      for (var b = 0; b < classes.length; b++) if (b !== a && f[b] >= maximo) { ok = false; break; }
      if (ok) rot = classes[a].nome;
    }
    return rot;
  }

  // regra de prioridade (semantica.rotulo_prioridade): pessoa >= 5%, senao animal >= 2%, senao
  // "outro" se a supercategoria dominante entre as restantes for veiculo, movel ou exterior
  function rotuloPrioridade(dec, i) {
    if (dec.fora[i]) return null;
    var F = function (s) { return dec.frac[i * dec.k + dec.col[s]]; };
    if (F('person') >= 5) return 'pessoa';
    if (F('animal') >= 2) return 'animal';
    var melhor = 0, dom = null;
    dec.sup.forEach(function (s) {
      if (s === 'person' || s === 'animal') return;
      if (F(s) > melhor) { melhor = F(s); dom = s; }
    });
    return dom && OUTROS_VALIDOS.indexOf(dom) >= 0 ? 'outro' : null;
  }

  function classesExclusiva(dec, classes) {
    // {nome: [colunas sup_*]} -> [{nome, cols}]; so supercategorias tem dados no simulador
    return Object.keys(classes).map(function (nome) {
      return {nome: nome, cols: classes[nome].map(function (c) {
        var s = String(c).replace(/^sup_/, '');
        if (!(s in dec.col)) throw new Error('o simulador só tem supercategorias (sup_*): ' + c);
        return dec.col[s];
      })};
    });
  }

  /* cfg: os campos do "dataset" de um experimento. Devolve o resumo (como dataset_controlado.resumo)
     mais o que a pagina mostra, ou {erro}. */
  function simula(dec, cfg) {
    cfg = cfg || {};
    var S = cfg.sessoes == null ? 40 : Number(cfg.sessoes);
    var regra = cfg.regra || 'exclusiva';
    var rep = cfg.repeticoes == null ? null : Number(cfg.repeticoes);
    var nImg = cfg.n_imagens == null ? null : Number(cfg.n_imagens);
    var classes = cfg.classes == null ? null : cfg.classes;
    if (!(S >= 1 && S <= 40)) return {erro: 'sessoes tem de estar entre 1 e 40'};
    if (rep != null && !(rep >= 1 && rep <= 3)) return {erro: 'repeticoes tem de ser 1, 2, 3 ou nenhuma'};
    if (nImg != null && !(nImg >= 1)) return {erro: 'n_imagens tem de ser positivo'};
    if (cfg.balanceado && !classes && regra === 'exclusiva') return {erro: 'balanceado exige classes'};
    var semantico = classes != null || regra === 'prioridade';
    var lista = null, nomes;
    if (regra === 'prioridade') {
      nomes = classes == null ? PRIORIDADE.slice() : classes.slice();
    } else if (classes) {
      lista = classesExclusiva(dec, classes);
      nomes = lista.map(function (c) { return c.nome; });
    } else {
      nomes = [];
    }
    var minimo = Number(cfg.minimo == null ? 10 : cfg.minimo);
    var maximo = Number(cfg.maximo_outros == null ? 2 : cfg.maximo_outros);

    var nPool = new Uint8Array(dec.n), exibPool = 0, imgPool = 0;
    for (var i = 0; i < dec.n; i++) {
      var c = 0;
      for (var j = 0; j < 3; j++) { var s = dec.sess[i * 3 + j]; if (s !== SEM && s < S) c++; }
      nPool[i] = c; exibPool += c; if (c) imgPool++;
    }
    var cand = [], rots = [];
    for (i = 0; i < dec.n; i++) {
      if (!nPool[i]) continue;
      var r = null;
      if (semantico) {
        r = regra === 'prioridade' ? rotuloPrioridade(dec, i) : rotuloExclusivo(dec, i, lista, minimo, maximo);
        if (regra === 'prioridade' && classes != null && classes.indexOf(r) < 0) r = null;
        if (r == null) continue;
      }
      if (rep != null && nPool[i] < rep) continue;
      cand.push(i); rots.push(r);
    }
    var elegPorClasse = {};
    nomes.forEach(function (n) { elegPorClasse[n] = 0; });
    rots.forEach(function (r) { if (r != null) elegPorClasse[r] = (elegPorClasse[r] || 0) + 1; });

    var imagens, porClasse = null, exato = true, media = 0;
    if (nImg != null) {
      if (cfg.balanceado) {
        var faltam = nomes.filter(function (n) { return (elegPorClasse[n] || 0) < nImg; });
        if (faltam.length) {
          return {erro: 'n_imagens = ' + nImg + ' por classe, mas há só ' +
            faltam.map(function (n) { return n + ': ' + (elegPorClasse[n] || 0); }).join(', ')};
        }
        imagens = nImg * nomes.length;
        porClasse = {};
        nomes.forEach(function (n) { porClasse[n] = nImg; });
      } else {
        if (nImg > cand.length) return {erro: 'n_imagens = ' + nImg + ', mas só ' + cand.length + ' imagens são elegíveis'};
        imagens = nImg;
      }
    } else {
      imagens = cand.length;
      if (semantico) porClasse = elegPorClasse;
    }
    if (!imagens) return {erro: 'nenhuma imagem elegível com esta configuração'};

    var exib, hist = {};
    if (rep != null) {
      exib = imagens * rep; hist[rep] = imagens;
    } else if (nImg == null) {
      exib = 0;
      cand.forEach(function (i) { exib += nPool[i]; hist[nPool[i]] = (hist[nPool[i]] || 0) + 1; });
    } else {
      // o sorteio decide quais imagens entram: o numero de exibicoes so e conhecido em media
      cand.forEach(function (i) { media += nPool[i]; });
      media /= cand.length;
      exib = Math.round(imagens * media); hist = null; exato = false;
    }
    return {exibicoes: exib, imagens_unicas: imagens, repeticoes: hist, imagens_por_classe: porClasse,
            exato: exato, elegiveis: cand.length, elegiveis_por_classe: semantico ? elegPorClasse : null,
            pool: {exibicoes: exibPool, imagens: imgPool}, sessoes: S};
  }

  /* Amostras por epoca e lotes por epoca de cada modelo, dado o resumo e a agregacao. */
  function amostras(res, cfg) {
    var ag = cfg.agregacao || null;
    var porImagem = ag === 'media' || ag === 'sorteio' || ag === 'combinacao';
    var out = {};
    // FRR: uma linha por exibicao, ou por imagem com a media; sem epocas
    out.frr = ag === 'sorteio' || ag === 'combinacao' ? {erro: 'o FRR aceita só exibicoes e media'}
      : {amostras: porImagem ? res.imagens_unicas : res.exibicoes};
    // MindEye2: a epoca e 750 x sessoes, na proporcao das exibicoes mantidas (amostras_por_epoca_mindeye2)
    var n2 = porImagem ? res.imagens_unicas : res.exibicoes;
    var a2 = Math.floor(750 * res.sessoes * n2 / res.pool.exibicoes);
    out.mindeye2 = {amostras: a2, lotes: {16: Math.floor(a2 / 16), 8: Math.floor(a2 / 8)}};
    // MindEye1: uma amostra por imagem (repeticoes empilhadas), ou por exibicao com "exibicoes"
    var n1 = ag === 'exibicoes' ? res.exibicoes : res.imagens_unicas;
    out.mindeye1 = {amostras: n1, lotes: {16: Math.floor(n1 / 16)}};
    return out;
  }

  var api = {decodifica: decodifica, simula: simula, amostras: amostras, PRIORIDADE: PRIORIDADE};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else raiz.Simulador = api;
})(typeof window !== 'undefined' ? window : this);
