"""A pagina HTML: tabelas, graficos do FRR, curvas, ruido e galeria."""

import datetime
import json
import math
import pathlib

from .config import CEREBRO, GRUPOS, LEGENDAS, METRICAS, TIPOS
from .fmt import bi, commit, data_uri, duracao, esc, fmt, frac_txt, mi, milhar, parte, tempo_fmt, virg
from .rows import colunas_frr, linhas_de, melhores, tem_valor, valor_frr
from .texts import FRR_NOTA_HUBNESS, leitura


ASSETS = pathlib.Path(__file__).resolve().parent / "assets"
ESTILO = (ASSETS / "style.css").read_text()
SCRIPT = (ASSETS / "script.js").read_text()


def celula(v, formato, melhor=False):
    if v is None:
        return '<td class="falta">—</td>'
    return f'<td><span data-melhor>{fmt(v, formato)}</span></td>' if melhor else f'<td>{fmt(v, formato)}</td>'


def tabela_metricas(dados, colunas, familias):
    cab_fam = "".join(f'<th colspan="{n}">{esc(f)}</th>' if f else f'<th colspan="{n}"></th>'
                      for f, n in familias)
    cab = "".join(f'<th>{esc(r)}<span class="seta">{"↑" if maior else "↓"}</span></th>'
                  for _, r, maior, _ in colunas)
    corpo = []
    for g, titulo in GRUPOS.items():
        linhas = [l for l in linhas_de(dados, g) if tem_valor(l, colunas)]
        corpo.append(f'<tr class="grupo"><th colspan="{len(colunas) + 1}">{esc(titulo)}</th></tr>')
        best = {(c, t): melhores(linhas, c, maior, t)
                for c, _, maior, _ in colunas for t in ("enh", "base")}
        for i, l in enumerate(linhas):
            sub = f"<small>{esc(l['sub'])}</small>" if l["sub"] else ""
            tds = [f'<td class="modelo">{esc(l["rotulo"])}{sub}</td>']
            for c, _, maior, formato in colunas:
                partes = []
                for t in ("enh", "base"):
                    v = l[t].get(c) if l[t] else None
                    cls = f"v-{t}"
                    if v is None:
                        partes.append(f'<span class="{cls} falta">—</span>')
                    elif i in best[(c, t)]:
                        partes.append(f'<span class="{cls}" data-melhor>{fmt(v, formato)}</span>')
                    else:
                        partes.append(f'<span class="{cls}">{fmt(v, formato)}</span>')
                tds.append("<td>" + "".join(partes) + "</td>")
            corpo.append(f'<tr class="{"pub" if l["pub"] else ""}">' + "".join(tds) + "</tr>")
    return (f'<div class="rolagem"><table><thead><tr class="familia"><th></th>{cab_fam}</tr>'
            f'<tr><th>Modelo</th>{cab}</tr></thead><tbody>{"".join(corpo)}</tbody></table></div>')


def tabela_legendas(dados):
    cab = "".join(f"<th>{esc(r)}<span class=\"seta\">↑</span></th>" for _, r in LEGENDAS)
    corpo = []
    teto = next((m["legendas"] for m in dados["modelos"] if m["legendas"]), None)
    for g, titulo in GRUPOS.items():
        corpo.append(f'<tr class="grupo"><th colspan="{len(LEGENDAS) + 1}">{esc(titulo)}</th></tr>')
        ms = [m for m in dados["modelos"] if m["grupo"] == g and not m.get("frr")]
        vals = {k: [m["legendas"][f"{k}_brain_ref"] for m in ms if m["legendas"]] for k, _ in LEGENDAS}
        for m in ms:
            tds = [f'<td class="modelo">{esc(m["rotulo"])}</td>']
            for k, _ in LEGENDAS:
                v = m["legendas"][f"{k}_brain_ref"] if m["legendas"] else None
                tds.append(celula(v, "dec", v is not None and len(vals[k]) > 1 and v == max(vals[k])))
            corpo.append("<tr>" + "".join(tds) + "</tr>")
    if teto:
        tds = ['<td class="modelo">GIT na imagem vista (teto)</td>'] + \
              [f'<td>{fmt(teto[f"{k}_img_ref"], "dec")}</td>' for k, _ in LEGENDAS]
        corpo.append('<tr class="teto">' + "".join(tds) + "</tr>")
    return (f'<div class="rolagem"><table><thead><tr><th>Modelo</th>{cab}</tr></thead>'
            f'<tbody>{"".join(corpo)}</tbody></table></div>')


def tabela_modelos(dados):
    corpo = []
    for m in dados["modelos"]:
        p = m["params"]
        if m.get("frr"):
            f = m["frr_dados"]
            sub = f"{bi(f['parametros_lineares'])} de coeficientes lineares, forma fechada" if f else ""
            tempo = esc(m["tempo"]) if m["tempo"] else '<span class="falta">—</span>'
            corpo.append(
                "<tr>"
                f'<td class="modelo">{esc(m["rotulo"])}<small class="mono">{esc(m["id"])}</small></td>'
                f'<td class="num-l">{m["sessoes"]} {"sessão" if m["sessoes"] == 1 else "sessões"}</td>'
                '<td class="num-l">regressão linear<small>sem rede, sem ramo blurry</small></td>'
                f'<td class="cfg">só os coeficientes<small>{sub}</small></td>'
                f'<td class="cfg">{esc(m["treino"])}<small>FRR de Rokem &amp; Kay, como em Doerig et al.</small></td>'
                f'<td class="num-l">{tempo}<small>CV + ajuste final, GPU</small></td>'
                "</tr>")
            continue
        if m.get("me1"):
            tempo = esc(m["tempo"]) if m["tempo"] else '<span class="falta">—</span>'
            corpo.append(
                "<tr>"
                f'<td class="modelo">{esc(m["rotulo"])}<small class="mono">{esc(m["id"])}</small></td>'
                f'<td class="num-l">{m["sessoes"]} {"sessão" if m["sessoes"] == 1 else "sessões"}</td>'
                f'<td class="num-l">hidden {m["hidden"]}<small>com ramo blurry · CLIP ViT-L/14</small></td>'
                f'<td class="cfg">tudo, do zero<small>{mi(sum(p.values())) if p else ""}'
                f'{" · " if p else ""}{esc(m["prior"])}</small></td>'
                f'<td class="cfg">{esc(m["treino"])}<small>sem pré-treino; imagem pelo Versatile '
                'Diffusion</small></td>'
                f'<td class="num-l">{tempo}<small>alto + baixo nível</small></td>'
                "</tr>")
            continue
        if m.get("paper"):
            treinado = "tudo" + (f"<small>{mi(sum(p.values()))}</small>" if p else "")
        elif p:
            total = sum(p.values())
            treinado = (f"só a ridge<small>{mi(p['ridge'])} de {mi(total)} "
                        f"({parte(p['ridge'], total)})</small>")
        else:
            treinado = "só a ridge"
        tempo = esc(m["tempo"]) if m["tempo"] else '<span class="falta">—</span>'
        corpo.append(
            "<tr>"
            f'<td class="modelo">{esc(m["rotulo"])}<small class="mono">{esc(m["id"])}</small></td>'
            f'<td class="num-l">{m["sessoes"]} {"sessão" if m["sessoes"] == 1 else "sessões"}</td>'
            f'<td class="num-l">hidden {m["hidden"]}<small>{"com" if m["blurry"] else "sem"} ramo blurry'
            f'</small></td>'
            f'<td class="cfg">{treinado}<small>{esc(m["prior"])}</small></td>'
            f'<td class="cfg">{esc(m["treino"])}<small>a partir de <span class="mono">'
            f'{esc(m["partida"])}</span></small></td>'
            f'<td class="num-l">{tempo}</td>'
            "</tr>")
    return ('<div class="rolagem"><table class="modelos"><thead><tr><th>Modelo</th>'
            '<th style="text-align:left">Dados</th><th style="text-align:left">Arquitetura</th>'
            '<th style="text-align:left">Treinado</th><th style="text-align:left">Receita</th>'
            '<th style="text-align:left">Tempo (A4500)</th></tr></thead>'
            f'<tbody>{"".join(corpo)}</tbody></table></div>')


def tabela_frr(dados):
    cols = colunas_frr()
    cab = "".join(f'<th>{esc(r)}<span class="seta">↑</span></th>' for r, _, _ in cols)
    corpo = []
    for g, titulo in GRUPOS.items():
        vs = [v for v in dados["frr_variantes"] if v["grupo"] == g and v["dados"]]
        if not vs:
            continue
        corpo.append(f'<tr class="grupo"><th colspan="{len(cols) + 1}">{esc(titulo)}</th></tr>')
        for v in vs:
            tds = "".join(celula(valor_frr(get, v["dados"]), fm) for _, get, fm in cols)
            corpo.append(f'<tr><td class="modelo">{esc(v["rotulo"])}</td>{tds}</tr>')
    return (f'<div class="rolagem"><table><thead><tr><th>Variante</th>{cab}</tr></thead>'
            f'<tbody>{"".join(corpo)}</tbody></table></div>')


def tabela_custo_frr(dados):
    linhas = []
    for m in dados["modelos"]:
        f = m.get("frr_dados")
        if not f:
            continue
        c, cv, cfg = f["custo"], f["cv"], f["config"]
        linhas.append(
            f'<tr><td class="modelo">{esc(m["rotulo"])}</td>'
            f'<td>{milhar(cfg["exibicoes_treino"])}</td><td>{milhar(cfg["imagens_treino"])}</td>'
            f'<td>{bi(f["parametros_lineares"])}</td>'
            f'<td>{duracao(cv["segundos_cv"])}</td><td>{duracao(c["segundos_ajuste"])}</td>'
            f'<td>{virg(c["pico_gpu_gib"], 1)} GiB</td><td>{virg(c["pico_ram_gib"], 1)} GiB</td></tr>')
    if not linhas:
        return ""
    return ('<div class="rolagem"><table><thead><tr><th>Modelo</th><th>Exibições</th><th>Imagens</th>'
            '<th>Coeficientes</th><th>Validação cruzada</th><th>Ajuste + previsão</th>'
            '<th>Pico na GPU</th><th>Pico de RAM</th></tr></thead>'
            f'<tbody>{"".join(linhas)}</tbody></table></div>')


def graficos_frr(dados):
    figs = []
    for m in dados["modelos"]:
        f = m.get("frr_dados")
        if not f:
            continue
        cv = f["cv"]
        fr, h = cv["fracs"], cv["histograma_fracoes"]
        tot = sum(h) or 1
        barras = "".join(
            f'<div class="barra" style="--h:{100 * v / tot:.2f}%" '
            f'title="fração {frac_txt(x)}: {virg(100 * v / tot, 1)}% das dimensões"></div>'
            for x, v in zip(fr, h))
        eixo = "".join(f"<span>{frac_txt(x)}</span>" for x in (fr[0], fr[len(fr) // 2], fr[-1]))
        pico = max(range(len(h)), key=h.__getitem__)
        figs.append(
            f'<figure class="curva"><figcaption><b>{esc(m["rotulo"])}</b>'
            f'<span>fração escolhida em cada dimensão do alvo, uma barra por fração testada · a mais '
            f'comum: {frac_txt(fr[pico])} ({100 * h[pico] / tot:.0f}%)</span></figcaption><div class="frr-graf">'
            f'<div class="barras" style="--n:{len(fr)}" role="img" '
            f'aria-label="Histograma das frações escolhidas">{barras}</div>'
            f'<div class="eixo">{eixo}</div></div></figure>')
    for m in dados["modelos"]:
        f = m.get("frr_dados")
        if not f:
            continue
        cv = f["cv"]
        fr, r2 = cv["fracs"], cv["cv_r2_por_fracao"]
        W, H, pad = 300, 150, 8
        ib = max(range(len(r2)), key=r2.__getitem__)
        topo = max(r2[ib], 0.0)
        piso = max(min(r2), -2 * topo) if topo > 0 else min(r2)     # o pico e o que importa: corta o resto
        span = (topo - piso) or 1.0
        x = lambda fr_: pad + (math.log10(fr_) - math.log10(fr[0])) * (W - 2 * pad) / ((math.log10(fr[-1]) - math.log10(fr[0])) or 1)
        y = lambda v: pad + (topo - max(v, piso)) * (H - 2 * pad - 12) / span
        pts, cortou = [], False
        for i, (fr_, v) in enumerate(zip(fr, r2)):
            if v < piso and i > 0:
                t = (r2[i - 1] - piso) / (r2[i - 1] - v)
                pts.append(f"{x(fr[i - 1]) + t * (x(fr_) - x(fr[i - 1])):.1f},{y(piso):.1f}")
                cortou = True
                break
            pts.append(f"{x(fr_):.1f},{y(v):.1f}")
        aviso = (f'<text x="{W - pad}" y="{H - 20}" text-anchor="end">'
                 f'cai a {virg(r2[-1], 2).replace("-", "−")} em {frac_txt(fr[-1])}</text>'
                 if cortou else "")
        figs.append(
            f'<figure class="curva"><figcaption><b>{esc(m["rotulo"])}</b>'
            f'<span>R² da validação cruzada com uma fração só para todas as dimensões, em escala '
            f'logarítmica · melhor: {frac_txt(fr[ib])} (R² {virg(r2[ib], 3)})</span></figcaption><div class="frr-graf">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="R² da validação cruzada por fração">'
            f'<line class="zero" x1="{pad}" x2="{W - pad}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>'
            f'<text x="{pad}" y="{y(0) - 4:.1f}">R² = 0</text>'
            f'<polyline class="linha" points="{" ".join(pts)}"/>'
            f'<circle class="pico" cx="{x(fr[ib]):.1f}" cy="{y(r2[ib]):.1f}" r="3.5"/>{aviso}'
            f'<text x="{pad}" y="{H - 2}">{frac_txt(fr[0])}</text>'
            f'<text x="{W - pad}" y="{H - 2}" text-anchor="end">{frac_txt(fr[-1])}</text></svg></div></figure>')
    return f'<div class="curvas">{"".join(figs)}</div>' if figs else ""


def galeria(dados):
    g = dados["galeria"]
    cols = g["colunas"]
    partes = ['<div class="cab vista">Imagem vista</div>'] + \
             [f'<div class="cab">{esc(c["rotulo"])}</div>' for c in cols]
    for j, i in enumerate(g["idx"]):
        rot = f'teste #{i}' + (f' · NSD {g["nsd"][j]}' if g["nsd"] else "")
        partes.append(
            f'<figure class="vista-cel"><img src="{data_uri(g["vista"][j])}" '
            f'alt="Imagem vista, {esc(rot)}" loading="lazy" width="224" height="224">'
            f'<figcaption><span class="id">{esc(rot)}</span>'
            f'<span class="txt">{esc(g["coco"][j])}</span></figcaption></figure>')
        for c in cols:
            imgs = []
            for tipo, nome in TIPOS:
                if tipo in c["img"]:
                    imgs.append(f'<img class="k-{tipo}" src="{data_uri(c["img"][tipo][j])}" '
                                f'alt="{esc(nome)}: {esc(c["rotulo"])}, {esc(rot)}" loading="lazy" '
                                f'width="224" height="224">')
                elif tipo == "blur" and c["img"] and c.get("me1"):
                    imgs.append('<div class="k-blur sem">a borrada só entra no img2img</div>')
                elif tipo == "blur" and c["img"]:
                    imgs.append('<div class="k-blur sem">sem ramo blurry</div>')
                else:
                    imgs.append(f'<div class="k-{tipo} sem">pendente</div>')
            leg = f'<figcaption><span class="txt">{esc(c["legendas"][j])}</span></figcaption>' \
                if c["legendas"] else ""
            partes.append(f'<figure>{"".join(imgs)}{leg}</figure>')
    return (f'<div class="galeria-caixa"><div class="galeria" style="--cols:{len(cols) + 1}">'
            + "".join(partes) + "</div></div>")


def curvas_html(dados):
    figs = []
    for i, c in enumerate(dados["curvas"]):
        d = c["dados"]
        if d:
            corpo = f'<div class="grafico" data-i="{i}"></div>'
            linhas = "".join(f'<tr><td>{e + 1}</td><td>{fmt(f, "pct")}</td><td>{fmt(b, "pct")}</td></tr>'
                             for e, f, b in zip(d["epocas"], d["fwd"], d["bwd"]))
            corpo += ('<details class="numeros"><summary>Números por época</summary>'
                      '<div class="rolagem"><table><thead><tr><th>Época</th><th>Img→cér</th>'
                      f'<th>Cér→img</th></tr></thead><tbody>{linhas}</tbody></table></div></details>')
            sub = c["sub"]
            if d.get("segundos"):
                sub += f" · {tempo_fmt(d['segundos'])} de treino"
        else:
            corpo = '<div class="vazia">treino ainda não rodou</div>'
            sub = c["sub"]
        figs.append(f'<figure class="curva"><figcaption><b>{esc(c["titulo"])}</b>'
                    f'<span>{esc(sub)}</span></figcaption>{corpo}</figure>')
    js = json.dumps([c["dados"] for c in dados["curvas"]], separators=(",", ":"))
    return "".join(figs), js


def ruido_html(dados):
    r = dados["ruido"]
    pp = lambda d: None if d is None else f"{d * 100:.1f} p.p.".replace(".", ",", 1)

    linhas = []
    for rot, f, b in r["treino"]:
        linhas.append(f'<tr><td class="modelo">{esc(rot)}</td>{celula(f, "pct")}{celula(b, "pct")}</tr>')
    sementes = [(f, b) for _, f, b in r["treino"][1:] if f is not None]
    if len(sementes) >= 2:
        fs, bs = [v[0] for v in sementes], [v[1] for v in sementes]
        linhas.append('<tr class="teto"><td class="modelo">maior diferença entre sementes</td>'
                      f'<td>{pp(max(fs) - min(fs))}</td><td>{pp(max(bs) - min(bs))}</td></tr>')
    treino = ('<div class="rolagem"><table><thead><tr><th>Semente do treino</th>'
              '<th>Img→cér<span class="seta">↑</span></th><th>Cér→img<span class="seta">↑</span></th>'
              f'</tr></thead><tbody>{"".join(linhas)}</tbody></table></div>')

    a42, a7 = r["amostragem"][42], r["amostragem"][7]
    cols = METRICAS[:8] + [("Brain Corr. nsd_general", "Corr. nsdgeneral", True, "dec")]
    linhas = []
    for c, rot, maior, formato in cols:
        tds = [f'<td class="modelo">{esc(rot)}<span class="seta">{"↑" if maior else "↓"}</span></td>']
        for t in ("base", "enh"):
            v42 = a42[t].get(c) if a42[t] else None
            v7 = a7[t].get(c) if a7[t] else None
            d = abs(v42 - v7) if v42 is not None and v7 is not None else None
            dtxt = (pp(d) if formato == "pct" else fmt(d, "dec")) if d is not None else None
            tds += [celula(v42, formato), celula(v7, formato),
                    f'<td>{dtxt}</td>' if dtxt else '<td class="falta">—</td>']
        linhas.append("<tr>" + "".join(tds) + "</tr>")
    amostragem = ('<div class="rolagem"><table><thead><tr class="familia"><th></th>'
                  '<th colspan="3">unCLIP</th><th colspan="3">Refinadas</th></tr>'
                  '<tr><th>Métrica</th><th>semente 42</th><th>semente 7</th><th>diferença</th>'
                  '<th>semente 42</th><th>semente 7</th><th>diferença</th></tr></thead>'
                  f'<tbody>{"".join(linhas)}</tbody></table></div>')
    linhas = [f'<tr><td class="modelo">{esc(rot)}</td>{celula(f, "pct")}{celula(b, "pct")}</tr>'
              for rot, f, b in r["controle"]]
    controle = ('<div class="rolagem"><table><thead><tr><th>Ridge 1024</th>'
                '<th>Img→cér<span class="seta">↑</span></th><th>Cér→img<span class="seta">↑</span></th>'
                f'</tr></thead><tbody>{"".join(linhas)}</tbody></table></div>')
    return treino, amostragem, controle


def situacao(dados):
    itens = []
    completos = 0
    for m in dados["modelos"]:
        ok = bool(m["frr_dados"]) if m.get("frr") else bool(m["tabelas"]["enh"] and m["tabelas"]["base"])
        completos += ok
        itens.append(f'<li class="{"" if ok else "pendente"}"><span class="ponto"></span>'
                     f'{esc(m["rotulo"])}{"" if ok else " · pendente"}</li>')
    return completos, '<ul class="situacao">' + "".join(itens) + "</ul>"


def pagina(dados, args):
    completos, sit = situacao(dados)
    n_modelos = len(dados["modelos"])
    curvas, curvas_js = curvas_html(dados)
    ruido_treino, ruido_amostragem, ruido_controle = ruido_html(dados)
    fam_rec = [("Baixo nível", 4), ("Alto nível", 4), ("Retrieval top-1 · 300", 2)]
    tab_rec = tabela_metricas(dados, METRICAS, fam_rec)
    col_cer = [(k, r, True, "dec") for k, r in CEREBRO]
    tab_cer = tabela_metricas(dados, col_cer, [("Correlação com a resposta medida", len(CEREBRO))])
    agora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    textos = leitura(dados)
    lista = ("<ul class=\"leitura\">" + "".join(f"<li>{t}</li>" for t in textos) + "</ul>") \
        if textos else ""
    if completos == n_modelos and textos:
        topo_resumo = (f'<section class="secao" id="resumo"><header><h2>Resumo</h2></header>'
                       f'{lista}</section>')
    else:
        estado = (f"{completos} de {n_modelos} modelos avaliados até agora; o restante ainda roda "
                  "no <code>run_benchmark.sh</code>.")
        topo_resumo = (f'<section class="secao" id="situacao"><header><h2>Situação</h2>'
                       f'<p class="nota">{estado}</p></header>{sit}</section>')

    titulo = "<title>Benchmark Ridge-Only</title>"
    fontes = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
              '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
              '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@'
              '62..125,500..800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@'
              '0,400;0,500;0,600;0,700;1,400&display=swap">')
    corpo = f"""
<main class="pagina">
  <header class="topo">
    <p class="olho">MindEye2 · NSD sujeito 1 · fMRI → imagem</p>
    <h1>Benchmark ridge-only</h1>
    <p class="lede">Quatro variações do fine-tune que treina só a camada ridge (a única parte do
      MindEye2 que é própria de cada pessoa), comparadas entre si, com os modelos publicados no
      artigo, que retreinam a rede inteira, com um baseline linear (FRR) que não usa rede nenhuma e
      com o MindEye1, treinado do zero.
      Mesmo conjunto de teste, mesmo pipeline de reconstrução e as mesmas métricas para todos.</p>
    <dl class="fatos">
      <div><dt>Teste</dt><dd>1.000 imagens · 3 repetições cada</dd></div>
      <div><dt>Entrada</dt><dd>15.724 voxels (nsdgeneral)</dd></div>
      <div><dt>Acaso no retrieval</dt><dd>1 em 300 = 0,33%</dd></div>
      <div><dt>GPU</dt><dd>1× RTX A4500 · 20 GB</dd></div>
    </dl>
    <nav class="indice" aria-label="Seções">
      <a href="#resumo">Resumo</a><a href="#modelos">Modelos</a><a href="#metricas">Métricas</a><a href="#cerebro">Correlação cerebral</a>
      <a href="#legendas">Legendas</a><a href="#frr">FRR</a><a href="#ruido">Ruído</a><a href="#treino">Treino</a>
      <a href="#galeria">Galeria</a>
      <a href="#metodo">Método</a>
    </nav>
  </header>

  {topo_resumo}

  <section class="secao" id="modelos">
    <header><h2>Modelos</h2>
      <p class="nota">Os ridge-only congelam backbone e prior e treinam só a ridge. Os dois do
      artigo partem do mesmo pré-treino multi-sujeito e treinam tudo. O FRR é uma regressão linear
      de forma fechada, sem treino por gradiente. O MindEye1 treina tudo do zero, nas mesmas
      sessões e com o mesmo teste, com a receita do artigo dele em batch 16 e Adam de 8 bits para
      caber na placa.</p></header>
    {tabela_modelos(dados)}
  </section>

  <section class="secao" id="metricas" data-tipo="enh">
    <header><h2>Reconstrução e retrieval</h2>
      <p class="nota">Destaque: melhor valor dentro de cada grupo, sem contar as linhas de
      reconstruções publicadas (que repetem um modelo já listado). ↑ maior é melhor, ↓ menor é
      melhor. O retrieval não depende da reconstrução e é igual nas duas abas. <b>Img→cér</b>: para
      cada imagem, achar o seu cérebro entre 300 previsões; <b>Cér→img</b>: para cada previsão, achar
      a sua imagem (no código, fwd e bwd). O FRR só tem retrieval. Cada modelo é medido no seu
      espaço CLIP: o MindEye1 no ViT-L/14 (257 × 768), os outros no ViT-bigG/14. O MindEye1 não tem
      refinamento e usa a imagem borrada no img2img: a mesma reconstrução aparece nas duas abas,
      sem a mistura 75/25.</p></header>
    <div class="controles">
      <div class="seletor" role="group" aria-label="Reconstruções avaliadas" data-alvo="metricas" data-chave="tipo">
        <button type="button" aria-pressed="true" data-valor="enh">Refinadas</button>
        <button type="button" aria-pressed="false" data-valor="base">unCLIP (sem refinar)</button>
      </div>
    </div>
    {tab_rec}
  </section>

  <section class="secao" id="cerebro" data-tipo="enh">
    <header><h2>Correlação cerebral</h2>
      <p class="nota">O GNet prevê a resposta do córtex visual a cada reconstrução; o número é a
      correlação de Pearson entre essa previsão e a resposta medida à imagem real, por região.</p></header>
    <div class="controles">
      <div class="seletor" role="group" aria-label="Reconstruções avaliadas" data-alvo="cerebro" data-chave="tipo">
        <button type="button" aria-pressed="true" data-valor="enh">Refinadas</button>
        <button type="button" aria-pressed="false" data-valor="base">unCLIP (sem refinar)</button>
      </div>
    </div>
    {tab_cer}
  </section>

  <section class="secao" id="legendas">
    <header><h2>Legendas previstas</h2>
      <p class="nota">Legenda gerada pelo GIT a partir do embedding que o modelo prevê do cérebro,
      comparada com as legendas humanas do COCO. A última linha é o GIT olhando a imagem real: o
      teto do que dá para atingir.</p></header>
    {tabela_legendas(dados)}
  </section>

  <section class="secao" id="frr">
    <header><h2>FRR: o baseline linear</h2>
      <p class="nota">Regressão ridge fracionária (Rokem &amp; Kay, 2020) dos 15.724 voxels direto para
      o embedding CLIP achatado (256 × 1.664 = 425.984 números), seguindo Doerig et al. (2025): 20
      frações de 0,05 a 1, validação cruzada de 5 dobras agrupada por imagem e uma fração escolhida
      para cada dimensão do alvo. É o modelo linear por sujeito do plano de estágio: sem rede, sem
      prior e sem reconstrução, então só o retrieval é comparável com os outros modelos. Cosseno,
      Pearson e cosseno centrado medem o quanto do embedding verdadeiro a regressão recupera, imagem
      a imagem.</p></header>
    {tabela_frr(dados)}
    <p class="nota">{FRR_NOTA_HUBNESS}</p>
    {graficos_frr(dados)}
    {tabela_custo_frr(dados)}
  </section>

  <section class="secao" id="ruido">
    <header><h2>Quanto é ruído</h2>
      <p class="nota">Com a mesma semente e o mesmo código, o treino é determinístico: o 4096 +
      blurry retreinado aqui repetiu a corrida anterior época por época. Repetir uma corrida não
      mede ruído, então esta seção troca só a semente, na configuração 1024 + prior. Diferenças
      entre modelos menores que estas estão dentro do ruído.</p></header>
    <h3>Semente do treino · retrieval top-1 entre 300</h3>
    {ruido_treino}
    <p class="nota">O retreino com a semente 42 não repete o checkpoint do release bit a bit: desde
    aquele treino, a única mudança no caminho do 1024 foi guardar as imagens da época em fp16 (o
    conserto de memória do treino de 40 sessões), o que arredonda de outro jeito os alvos CLIP já
    na primeira iteração. O resultado final fica no mesmo lugar.</p>
    <h3>Semente da reconstrução · o mesmo modelo 1024 + prior amostrado duas vezes</h3>
    {ruido_amostragem}
    <h3>Com e sem prior · o mesmo script e a mesma semente</h3>
    <p class="nota">Os dois checkpoints de 1024 do release vieram de scripts e versões de código
    diferentes. O retreino muda só <code>--use_prior</code>, para isolar o efeito da loss do prior
    no treino da ridge.</p>
    {ruido_controle}
  </section>

  <section class="secao" id="treino">
    <header><h2>Retrieval durante o treino</h2>
      <p class="nota">Top-1 entre 300 no teste, medido ao fim de cada época pelo próprio treino
      sempre nas mesmas 300 primeiras imagens de teste. O número das tabelas é a média de 30
      sorteios de 300 entre as 1.000, por isso difere um pouco.
      Só as corridas que gravaram <code>metrics.csv</code>: os checkpoints do release vieram de
      corridas anteriores, sem esse registro; as sementes do 1024 + prior mostram a mesma
      configuração.</p></header>
    <ul class="legenda-series">
      <li><span class="chave" style="border-color:var(--s1)"></span>Img→cér: dada a imagem, achar o seu cérebro</li>
      <li><span class="chave" style="border-color:var(--s2)"></span>Cér→img: dado o cérebro, achar a sua imagem</li>
    </ul>
    <div class="curvas">{curvas}</div>
  </section>

  <section class="secao" id="galeria" data-tipo="enh" data-legendas="sim">
    <header><h2>Galeria</h2>
      <p class="nota">{len(dados["galeria"]["idx"])} imagens de teste sorteadas (semente {args.seed}),
      as mesmas para todos os modelos. Embaixo da imagem vista, a legenda humana do COCO; embaixo
      de cada reconstrução, a legenda que o modelo previu. Nas métricas, os modelos com ramo blurry
      avaliam 75% refinada + 25% blurry, como no artigo; aqui a refinada aparece pura.</p></header>
    <div class="controles">
      <div class="seletor" role="group" aria-label="Tipo de reconstrução" data-alvo="galeria" data-chave="tipo">
        <button type="button" aria-pressed="true" data-valor="enh">Refinadas</button>
        <button type="button" aria-pressed="false" data-valor="base">unCLIP</button>
        <button type="button" aria-pressed="false" data-valor="blur">Blurry</button>
      </div>
      <div class="seletor" role="group" aria-label="Legendas" data-alvo="galeria" data-chave="legendas">
        <button type="button" aria-pressed="true" data-valor="sim">Com legendas</button>
        <button type="button" aria-pressed="false" data-valor="nao">Sem legendas</button>
      </div>
    </div>
    {galeria(dados)}
  </section>

  <section class="secao" id="metodo">
    <header><h2>Método</h2></header>
    <div class="colunas">
      <div>
        <h3>Pipeline</h3>
        <p>Para cada uma das 1.000 imagens de teste, o modelo roda nas 3 repetições do fMRI e as
        saídas são médias. O prior de difusão (20 passos) gera o embedding CLIP, o unCLIP do SDXL
        (38 passos) gera a imagem <em>unCLIP</em>, e um img2img com SDXL guiado pela legenda prevista
        gera a <em>refinada</em>. Os modelos do artigo passaram pelo mesmo código, com a mesma
        semente; as linhas de reconstruções publicadas avaliam os tensores divulgados pelos autores
        e servem de conferência do pipeline.</p>
      </div>
      <div>
        <h3>Métricas</h3>
        <ul>
          <li><b>PixCorr, SSIM</b>: semelhança pixel a pixel com a imagem vista.</li>
          <li><b>Alex, Incep, CLIP</b>: identificação 2-way — com que frequência a reconstrução fica
            mais perto da sua imagem do que de outra, no espaço de features da rede.</li>
          <li><b>Eff, SwAV</b>: distância de correlação entre features (menor é melhor).</li>
          <li><b>Retrieval</b>: top-1 entre 300 candidatos, média de 30 sorteios.
            <b>Img→cér</b> (fwd no código): para cada imagem, achar o seu cérebro entre as 300
            previsões. <b>Cér→img</b> (bwd): para cada previsão, achar a sua imagem. O artigo define
            Image Retrieval como cérebro→imagem, o oposto do que o código calcula em fwd; as
            colunas seguem o código.</li>
          <li><b>Cosseno, Pearson</b> (só FRR): entre o embedding CLIP previsto e o verdadeiro,
            achatados, imagem a imagem.</li>
        </ul>
      </div>
      <div>
        <h3>FRR</h3>
        <p>Os voxels (a média das 3 repetições, no teste) vão por uma regressão linear direto para o
        embedding CLIP ViT-bigG/14 achatado. Em vez do α da ridge, a FRR fixa a fração da norma da
        solução sem regularização (0 = tudo encolhido, 1 = OLS) e acha o α de cada fração e de cada
        dimensão do alvo na decomposição espectral dos voxels. A fração de cada dimensão sai da
        validação cruzada, que agrupa por imagem. O alvo de treino é o do treino do MindEye2 (imagem
        em fp16); o de teste é o das avaliações deste benchmark (imagem em fp32, como no
        <code>final_evaluations.py</code>): o embedder dá saídas um pouco diferentes nos dois casos
        (cosseno 0,996). Implementação própria em GPU, conferida contra o pacote
        <code>fracridge</code> de referência e contra uma ridge resolvida pela equação normal.</p>
      </div>
      <div>
        <h3>Ressalvas</h3>
        <ul>
          <li>Nos ridge-only, backbone e prior da reconstrução vêm do checkpoint de partida,
            porque ficaram congelados no treino. Para o 1024 sem prior é o único prior que existe
            (e é bit a bit o que o 1024 + prior usou: a diferença entre os dois está só em como a
            ridge foi treinada). Para os treinados com congelados em fp16, evita reconstruir com
            pesos arredondados.</li>
          <li>O 4096 + blurry difere do artigo 1 sessão só no que é treinado e no batch (8 contra
            24). O 1024 difere também em hidden_dim, ramo blurry e ponto de partida.</li>
          <li>O ridge de 40 sessões é 1024 sem ramo blurry e treinou 20 épocas; o artigo de 40
            sessões é 4096 com blurry e treinou 150. A diferença de baixo nível entre os dois
            mistura arquitetura e treino.</li>
          <li>A tabela antiga do artigo no repositório avaliava as reconstruções publicadas sem a
            mistura de 25% blurry e sem as legendas previstas, que não estavam disponíveis. Com as
            duas, as mesmas reconstruções publicadas sobem de PixCorr 0,202 para 0,235 e de SSIM
            0,391 para 0,428.</li>
          <li>Os modelos 4096 reconstroem com o backbone em fp16: o mesmo arredondamento que o
            autocast já faz, para caber em 20 GB.</li>
          <li>Uma corrida por modelo, semente 42. A régua de ruído está na seção
            <a href="#ruido">Quanto é ruído</a>. A diferença de cerca de 1 ponto entre as duas
            corridas do 4096 + blurry descrita no docs/EXPERIMENTO.md não era ruído: com o mesmo código,
            a corrida se repete exatamente, e entre aquelas duas o código mudou (o conserto do
            ColorJitter em fp16 entrou pouco antes da segunda).</li>
          <li>As métricas do 1024 + prior foram recalculadas sobre os tensores do release. As das
            reconstruções unCLIP saíram idênticas às da tabela antiga do repositório; as das
            refinadas diferem um pouco (PixCorr 0,181 contra 0,182; Incep 83,6% contra 84,3%),
            sinal de que a tabela antiga veio de uma geração anterior das refinadas.</li>
        </ul>
      </div>
    </div>
    <pre>./scripts/run_benchmark.sh         # treina, reconstrói, avalia e remonta esta página
python src/make_benchmark.py       # só remonta a página</pre>
  </section>

  <footer>Gerado em {agora} a partir do commit <span class="mono">{commit()}</span> por
    <span class="mono">src/make_benchmark.py</span>.</footer>
</main>
<script type="application/json" id="dados-curvas">{curvas_js}</script>
<script>{SCRIPT}</script>
"""
    return titulo, fontes, corpo
