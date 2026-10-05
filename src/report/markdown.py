"""O BENCHMARK.md e as grades de imagem que ele mostra."""

import datetime
import io

from .config import CEREBRO, GRUPOS, METRICAS, ROT_BWD, ROT_FWD
from .fmt import bi, commit, duracao, fmt, mi, milhar, parte, virg
from .rows import colunas_frr, linhas_de, melhores, tem_valor, valor_frr
from .texts import FRR_NOTA_HUBNESS, leitura


def markdown(dados, args):
    L = ["# Benchmark ridge-only", "",
         "MindEye2, sujeito 1 do NSD. Quatro variações do fine-tune só da camada ridge, comparadas "
         "com os modelos publicados no artigo (fine-tune completo), com um baseline linear (FRR) "
         "que não usa rede nenhuma e com o MindEye1 treinado do zero. Mesmo teste (1.000 imagens), mesmo pipeline de reconstrução e "
         "mesmas métricas para todos.", "",
         "A versão completa, com a galeria de reconstruções, curvas de treino e seletor "
         "refinada/unCLIP, está em [benchmark/index.html](benchmark/index.html) — um arquivo "
         "só, abre offline.", "",
         f"Gerado por `src/make_benchmark.py` em {datetime.datetime.now():%d/%m/%Y %H:%M} "
         f"(commit `{commit()}`).", "", "## Modelos", "",
         "| Modelo | Sessões | hidden_dim | Blurry | Parâmetros treinados | Treino | Tempo (A4500) |",
         "|---|---:|---:|---|---|---|---|"]
    for m in dados["modelos"]:
        p = m["params"]
        if m.get("frr"):
            f = m["frr_dados"]
            params = f"{bi(f['parametros_lineares'])} de coeficientes (forma fechada)" if f else "—"
            L.append(f"| {m['rotulo']} | {m['sessoes']} | — | não | {params} | {m['treino']} | {m['tempo'] or '—'} |")
            continue
        if m.get("me1"):
            params = f"todos, do zero ({mi(sum(p.values()))})" if p else "todos, do zero"
            L.append(f"| {m['rotulo']} | {m['sessoes']} | {m['hidden']} | sim | {params} | {m['treino']} "
                     f"| {m['tempo'] or '—'} |")
            continue
        if p:
            total = sum(p.values())
            params = f"todos ({mi(total)})" if m.get("paper") else \
                f"{mi(p['ridge'])} de {mi(total)} ({parte(p['ridge'], total)})"
        else:
            params = "—"
        L.append(f"| {m['rotulo']} | {m['sessoes']} | {m['hidden']} | {'sim' if m['blurry'] else 'não'} "
                 f"| {params} | {m['treino']} | {m['tempo'] or '—'} |")

    def tabela(tipo, colunas, titulo):
        L.extend(["", f"## {titulo}", "",
                  "| Modelo | " + " | ".join(f"{r} {'↑' if maior else '↓'}" for _, r, maior, _ in colunas) + " |",
                  "|---|" + "---:|" * len(colunas)])
        for g, gt in GRUPOS.items():
            L.append(f"| **{gt}** |" + " |" * len(colunas))
            linhas = [l for l in linhas_de(dados, g) if tem_valor(l, colunas)]
            best = {c: melhores(linhas, c, maior, tipo) for c, _, maior, _ in colunas}
            for i, l in enumerate(linhas):
                nome = ("↳ " if l["pub"] else "") + l["rotulo"] + (f" ({l['sub']})" if l["sub"] else "")
                vals = []
                for c, _, _, formato in colunas:
                    v = l[tipo].get(c) if l[tipo] else None
                    s = fmt(v, formato) if v is not None else "—"
                    vals.append(f"**{s}**" if i in best[c] else s)
                L.append(f"| {nome} | " + " | ".join(vals) + " |")

    if leitura(dados):
        import re
        L.extend(["", "## Resumo", ""])
        for t in leitura(dados):
            t = re.sub(r"</?b>", "**", t)
            t = re.sub(r'<a href="#[^"]*">([^<]*)</a>', r"\1 (na página)", t)
            L.append(f"- {t}")

    tabela("enh", METRICAS, "Reconstrução e retrieval — refinadas")
    tabela("base", METRICAS[:8], "Reconstrução — unCLIP (sem refinar)")
    tabela("enh", [(k, r, True, "dec") for k, r in CEREBRO], "Correlação cerebral (GNet) — refinadas")
    L += ["", "Negrito: melhor do grupo, sem contar as reconstruções publicadas. Modelos com ramo "
          "blurry avaliam 75% refinada + 25% blurry, como no artigo; o MindEye1 não tem refinamento e "
          "já usa a borrada no img2img, então a mesma reconstrução entra nas duas tabelas, sem mistura. "
          "Cada modelo faz o retrieval no seu espaço CLIP: o MindEye1 no ViT-L/14, os outros no ViT-bigG/14.", "",
          f"Retrieval top-1 entre 300, média de 30 sorteios. **{ROT_FWD}** (fwd no código): para cada "
          f"imagem, achar o seu cérebro entre as 300 previsões. **{ROT_BWD}** (bwd): para cada previsão, "
          "achar a sua imagem. É o que o código calcula, o oposto do comentário do `final_evaluations.py` "
          "e da definição de Image Retrieval no artigo (cérebro → imagem). O FRR só tem retrieval.", ""]

    cols = colunas_frr()
    L += ["## FRR: o baseline linear", "",
          "Regressão ridge fracionária (Rokem & Kay, 2020) dos 15.724 voxels direto para o embedding CLIP "
          "achatado (256 × 1.664 = 425.984 números), seguindo Doerig et al. (2025): validação cruzada de "
          "5 dobras agrupada por imagem e uma fração escolhida para cada dimensão do alvo. Sem rede, sem "
          "prior e sem reconstrução, então só o retrieval é comparável com os outros modelos; cosseno, "
          "Pearson e cosseno centrado medem o quanto do embedding verdadeiro a regressão recupera.", "",
          "| Variante | " + " | ".join(f"{r} ↑" for r, _, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
    for g, gt in GRUPOS.items():
        vs = [v for v in dados["frr_variantes"] if v["grupo"] == g and v["dados"]]
        if vs:
            L.append(f"| **{gt}** |" + " |" * len(cols))
        for v in vs:
            vals = [fmt(valor_frr(get, v["dados"]), fm) or "—" for _, get, fm in cols]
            L.append(f"| {v['rotulo']} | " + " | ".join(vals) + " |")
    L += ["", FRR_NOTA_HUBNESS, ""]
    custo = [m for m in dados["modelos"] if m.get("frr") and m.get("frr_dados")]
    if custo:
        L += ["| Modelo | Exibições | Imagens | Coeficientes | Validação cruzada | Ajuste + previsão | Pico na GPU | Pico de RAM |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m in custo:
            f = m["frr_dados"]
            L.append(f"| {m['rotulo']} | {milhar(f['config']['exibicoes_treino'])} | {milhar(f['config']['imagens_treino'])} "
                     f"| {bi(f['parametros_lineares'])} | {duracao(f['cv']['segundos_cv'])} | {duracao(f['custo']['segundos_ajuste'])} "
                     f"| {virg(f['custo']['pico_gpu_gib'], 1)} GiB | {virg(f['custo']['pico_ram_gib'], 1)} GiB |")
        L.append("")

    r = dados["ruido"]
    L += ["## Ruído", "",
          "O treino é determinístico com a mesma semente, então a régua troca só a semente, na "
          "configuração 1024 + prior.", "",
          "| Semente do treino | Img→cér ↑ | Cér→img ↑ |", "|---|---:|---:|"]
    for rot, f, bb in r["treino"]:
        L.append(f"| {rot} | {fmt(f, 'pct') or '—'} | {fmt(bb, 'pct') or '—'} |")
    L += ["", "| Com e sem prior (1024) | Img→cér ↑ | Cér→img ↑ |", "|---|---:|---:|"]
    for rot, f, bb in r["controle"]:
        L.append(f"| {rot} | {fmt(f, 'pct') or '—'} | {fmt(bb, 'pct') or '—'} |")
    a42, a7 = r["amostragem"][42], r["amostragem"][7]
    L += ["", "| Semente da reconstrução (refinadas) | 42 | 7 |", "|---|---:|---:|"]
    for c, rot, _, formato in METRICAS[:8]:
        v42 = a42["enh"].get(c) if a42["enh"] else None
        v7 = a7["enh"].get(c) if a7["enh"] else None
        L.append(f"| {rot} | {fmt(v42, formato) or '—'} | {fmt(v7, formato) or '—'} |")
    L += ["",
          "## Galeria", "",
          "Refinadas, os mesmos estímulos para todos os modelos (a página tem mais, e as unCLIP e blurry):",
          "", "![Galeria de reconstruções refinadas](benchmark/galeria_refinadas.jpg)", "",
          "![Galeria de reconstruções unCLIP](benchmark/galeria_unclip.jpg)", ""]
    return "\n".join(L)


def grade_jpg(dados, tipo, saida, n=8):
    """Grade estatica para o BENCHMARK.md: imagem vista + uma coluna por modelo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    g = dados["galeria"]
    cols = g["colunas"]
    n = min(n, len(g["idx"]))
    fig, ax = plt.subplots(n, len(cols) + 1, figsize=(1.55 * (len(cols) + 1), 1.55 * n + .5))
    for a in ax.ravel():
        a.axis("off")
    for j in range(n):
        ax[j][0].imshow(Image.open(io.BytesIO(g["vista"][j])))
        for k, c in enumerate(cols):
            if tipo in c["img"]:
                ax[j][k + 1].imshow(Image.open(io.BytesIO(c["img"][tipo][j])))
    for k, nome in enumerate(["Imagem vista"] + [c["rotulo"] for c in cols]):
        ax[0][k].set_title(nome.replace(" · ", "\n").replace(" + ", "\n+ ", 1) if len(nome) > 16 else nome,
                           fontsize=7.5, pad=4)
    fig.tight_layout(pad=.3)
    fig.savefig(saida, dpi=110, pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)
