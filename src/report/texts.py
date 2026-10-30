"""As conclusoes do benchmark: as escritas a mao e a do FRR, montada dos numeros medidos."""

from .config import REPO_URL
from .fmt import fmt, virg

# R² de validacao do pre-treino do FRR multi-sujeito nos sujeitos 2 a 8 (docs/FRR_MULTISUJEITO.md); nao esta
# nos registros dos modelos, entao fica aqui. Revisar se o pre-treino for refeito.
FRR_MS_R2_PRETREINO = "0,036"


FRR_NOTA_HUBNESS = (
    "Centrado: o mesmo retrieval depois de tirar a média de treino da previsão e do alvo. A regressão "
    "encolhe as previsões em direção à média e, no cosseno bruto, a imagem mais parecida com a média "
    "vence para quase toda previsão (hubness): por isso o Cér→img bruto é tão baixo. A comparação com "
    "os outros modelos usa o cosseno bruto, como o artigo; o centrado só separa esse efeito do sinal. "
    "Cosseno e Pearson brutos já valem ~0,5 só porque o embedding verdadeiro e a média de treino têm "
    "cosseno ~0,5 entre si; o cosseno centrado tira a média dos dois e mede o que a previsão acrescenta "
    "a ela. R² da CV: variância do embedding explicada nas dobras de validação, com a fração de cada dimensão; "
    "é otimista, porque a fração foi escolhida nessas mesmas dobras. As métricas limpas são as do teste "
    "(retrieval, cosseno, Pearson).")


# Conclusoes, escritas a mao a partir das tabelas; aparecem quando todos os
# modelos estao avaliados. Revisar se os numeros mudarem.
LEITURA = [
    "<b>1 sessão, mesma arquitetura do artigo.</b> O ridge 4096 + blurry fica à frente do "
    "fine-tune completo em 9 das 10 métricas de reconstrução e retrieval (perde só no SSIM: "
    "0,414 contra 0,421), treinando 2,89% dos parâmetros numa GPU de 20 GB. Nas identificações "
    "2-way a vantagem é de 1,5 a 2,2 pontos, acima dos até 0,6 ponto que mudam só por reamostrar "
    "a reconstrução. No retrieval Cér→img a distância é grande: 92,0% contra 77,6%.",
    "<b>1024 contra 4096 + blurry.</b> A dimensão maior e o ramo blurry pesam sobretudo no baixo "
    "nível (PixCorr 0,180 → 0,253; Alex(2) 85,4% → 89,6%). No alto nível o 1024 já empata com o "
    "artigo: Incep e CLIP a 1,3 e 0,7 ponto, Eff e SwAV a 0,001 e 0,003.",
    "<b>O prior importa no treino da ridge.</b> Com o mesmo script e a mesma semente, tirar a loss "
    "do prior derruba o retrieval (Img→cér / Cér→img) de 92,8% / 88,1% para 78,3% / 72,1%, e todas as métricas de "
    "reconstrução caem junto. O retrieval nem passa pelo prior: o que muda é o sinal de treino, "
    "já que a loss do prior supervisiona a saída inteira do backbone (256 × 1.664), e não só a "
    "projeção contrastiva.",
    "<b>40 sessões.</b> O ridge 1024 chega a 99,9% / 99,4% de retrieval e fica de 1 a 3 pontos do "
    "artigo nas identificações 2-way (CLIP 92,1% contra 93,6%). A distância maior é no baixo "
    "nível (PixCorr 0,277 contra 0,373), onde o artigo tem o ramo blurry e este não.",
    "<b>MindEye1, treinado do zero.</b> Sem pré-treino, com 1 sessão ele quase não recupera "
    "(retrieval 30,6% / 23,2%, contra 92,8% / 88,1% do ridge 1024 + prior) e fica atrás do ridge "
    "4096 + blurry nas identificações 2-way (CLIP 80,8% contra 83,8%, na tabela unCLIP), embora o "
    "PixCorr (0,252) seja maior. Com 40 sessões o retrieval sobe 64 pontos (94,5% / 87,6%) e o "
    "PixCorr chega a 0,382, acima do artigo reconstruído aqui (0,373 refinada, 0,324 unCLIP). O "
    "retrieval do MindEye1 é no espaço ViT-L/14 dele, não no bigG dos outros, então não é "
    "diretamente comparável.",
    "<b>Ruído.</b> Trocar a semente do treino move o retrieval em menos de 0,5 ponto; reamostrar a "
    "reconstrução move as identificações 2-way em até 0,6 ponto e o PixCorr em até 0,006. O "
    "efeito da semente do treino nas métricas de imagem não foi medido.",
    "<b>O pipeline reproduz o artigo.</b> Reconstruídos aqui, os dois modelos publicados ficam a "
    "até 1 ponto dos tensores divulgados pelos autores nas identificações 2-way, e a até 0,007 em "
    "PixCorr e SSIM.",
]


def leitura_frr(dados):
    """Conclusoes do FRR, montadas dos numeros medidos para nao ficarem desatualizadas.

    Devolve uma lista de textos (vazia se faltar resultado do FRR ou do ridge de referencia).
    """
    por_id = {m["id"]: m for m in dados["modelos"]}
    variantes = {x["id"]: x["dados"] for x in dados.get("frr_variantes", []) if x["dados"]}
    try:
        f1, f40 = por_id["subj01_frr_1sess"]["frr_dados"], por_id["subj01_frr_40sess"]["frr_dados"]
        r1 = por_id["subj01_ridgeonly_1sess_prior"]["tabelas"]["enh"]
        r40 = por_id["subj01_ridgeonly_40sess_prior"]["tabelas"]["enh"]
        v = {d: {"f": x["retrieval"]["fwd"]["media"], "b": x["retrieval"]["bwd"]["media"],
                 "fc": x["retrieval_centrado"]["fwd"]["media"], "bc": x["retrieval_centrado"]["bwd"]["media"]}
             for d, x in (("1", f1), ("40", f40))}
        rf1, rf40 = r1["FwdRetrieval"], r40["FwdRetrieval"]
    except (KeyError, TypeError):
        return []
    pc = lambda x: fmt(x, "pct")
    ganho_frr = (v["40"]["f"] - v["1"]["f"]) * 100
    ganho_ridge = (rf40 - rf1) * 100
    textos = []

    # 1. contra os modelos com rede, e o valor de escolher uma fracao por dimensao
    t = (f"<b>FRR, o baseline linear.</b> Uma regressão linear dos voxels para o embedding CLIP, sem rede "
         f"nenhuma, acha o cérebro certo de cada imagem (Img→cér) em {pc(v['1']['f'])} dos casos com 1 sessão "
         f"e {pc(v['40']['f'])} com 40, contra {pc(rf1)} e {pc(rf40)} do ridge 1024 + prior: com poucos dados "
         f"falta muito, e o ganho de 1 para 40 sessões é de {virg(ganho_frr, 0)} pontos no FRR e "
         f"{virg(ganho_ridge, 0)} no ridge.")
    g1, g40 = variantes.get("subj01_frr_1sess_global"), variantes.get("subj01_frr_40sess_global")
    if g1:
        t += (f" Escolher uma fração para cada dimensão do alvo é o que sustenta o resultado com pouco dado: com "
              f"uma fração só para todas, o de 1 sessão cai de {pc(v['1']['f'])} para "
              f"{pc(g1['retrieval']['fwd']['media'])}")
        if g40 and abs(g40["retrieval"]["fwd"]["media"] - v["40"]["f"]) < 0.005:
            t += "; com 40 sessões as duas coincidem."
        else:
            t += "."
    textos.append(t)

    # 2. a media: hubness e o que a previsao acrescenta
    t = (f"<b>FRR e a média.</b> No sentido Cér→img o cosseno bruto dá só {pc(v['1']['b'])} e {pc(v['40']['b'])}, "
         f"artefato de previsões encolhidas em direção à média (hubness); tirando a média de treino sobe para "
         f"{pc(v['1']['bc'])} e {pc(v['40']['bc'])}.")
    try:
        t += (f" O embedding previsto também acrescenta pouco a ela: o cosseno com o verdadeiro vale ~0,5, quase "
              f"todo pela média, e centrado fica em {virg(f1['embedding']['cosseno_centrado_medio'], 2)} e "
              f"{virg(f40['embedding']['cosseno_centrado_medio'], 2)}.")
    except KeyError:
        pass
    t += (f" O R² de validação cruzada é de {virg(f1['cv']['cv_r2_escolhida'] * 100, 1)}% e "
          f"{virg(f40['cv']['cv_r2_escolhida'] * 100, 1)}% (otimista, a fração foi escolhida nas mesmas dobras), "
          f"mas o retrieval só precisa do ranking.")
    textos.append(t)

    # 3. sensibilidade da grade: a estendida acha o otimo do erro quadratico, que nao e o do retrieval
    ext = variantes.get("subj01_frr_40sess_ext")
    if ext:
        f_ext, r2_ext = ext["retrieval"]["fwd"]["media"], ext["cv"]["cv_r2_escolhida"]
        piora = f_ext < v["40"]["f"]
        verbo = "piora" if piora else ("melhora" if f_ext > v["40"]["f"] else "não muda")
        textos.append(
            f"<b>FRR e a grade de frações.</b> Com 40 sessões, estender a grade até 0,001 (o ótimo do erro "
            f"quadrático cai abaixo da de Doerig) leva o R² de validação cruzada de "
            f"{virg(f40['cv']['cv_r2_escolhida'] * 100, 1)}% para {virg(r2_ext * 100, 1)}% e {verbo} o Img→cér "
            f"de {pc(v['40']['f'])} para {pc(f_ext)}" + (": o erro quadrático não é o critério do retrieval."
                                                       if piora else "."))

    # 4. multi-sujeito: A (o FRR de sempre), B (posto reduzido, o controle) e C (leitura pre-treinada nos
    # sujeitos 2 a 8). So afirma o que os numeros sustentam: se mudarem, o texto some em vez de ficar errado.
    try:
        ms = {s: {l: {"f": por_id[i]["frr_dados"]["retrieval"]["fwd"], "u": por_id[i]["tabelas"]["base"]}
                  for l, i in (("A", f"subj01_frr_{s}sess"), ("B", f"subj01_frr_postoreduzido_{s}sess"),
                               ("C", f"subj01_frr_multisujeito_{s}sess"))}
              for s in ("1", "20", "40")}
        g = lambda s, l, k: ms[s][l]["u"][k]
        r = lambda s, l: ms[s][l]["f"]["media"]
        meia = max([(x["f"]["ic95"][1] - x["f"]["ic95"][0]) / 2 for x in ms["1"].values() if x["f"].get("ic95")],
                   default=0.01)
        gargalo = all(g(s, "B", "CLIP") > g(s, "A", "CLIP") for s in ms) and g("1", "B", "PixCorr") > g("1", "A", "PixCorr")
        outros = (r("1", "C") - r("1", "B") > 2 * meia and g("1", "C", "CLIP") <= g("1", "B", "CLIP")
                  and all(abs(g(s, "C", "CLIP") - g(s, "A", "CLIP")) < 0.01 for s in ("20", "40")))
    except (KeyError, TypeError, IndexError):
        return textos
    if gargalo and outros:
        pix = lambda s, l: virg(g(s, l, "PixCorr"), 3)
        textos.append(
            f"<b>FRR multi-sujeito.</b> Três FRR sobre os mesmos dados do subj01: o de sempre (A); um que passa os "
            f"voxels por um espaço de 1.024 dimensões antes do CLIP, com a leitura aprendida só no subj01 (B, o "
            f"controle de posto reduzido); e o mesmo espaço com a leitura pré-treinada nos sujeitos 2 a 8 (C, "
            f"multi-sujeito). Os números de imagem são os da tabela unCLIP. "
            f"<b>O gargalo é o que ajuda a imagem</b> (B contra A): com 1 sessão, PixCorr {pix('1', 'B')} contra "
            f"{pix('1', 'A')} e CLIP {pc(g('1', 'B', 'CLIP'))} contra {pc(g('1', 'A', 'CLIP'))}; com 20 e 40 sessões o "
            f"CLIP é {pc(g('20', 'B', 'CLIP'))} contra {pc(g('20', 'A', 'CLIP'))} e "
            f"{pc(g('40', 'B', 'CLIP'))} contra {pc(g('40', 'A', 'CLIP'))}. "
            f"<b>Os outros sujeitos só ajudam o retrieval, e só com pouco dado</b> (C contra B): com 1 sessão o Img→cér "
            f"sobe de {pc(r('1', 'B'))} para {pc(r('1', 'C'))}, bem além do intervalo de 95% (±{virg(meia * 100, 1)} "
            f"ponto), mas a imagem não melhora (PixCorr {pix('1', 'C')} contra {pix('1', 'B')}, CLIP "
            f"{pc(g('1', 'C', 'CLIP'))} contra {pc(g('1', 'B', 'CLIP'))}); com 20 e 40 sessões C fica no nível de A "
            f"(CLIP {pc(g('20', 'C', 'CLIP'))} e {pc(g('40', 'C', 'CLIP'))} contra {pc(g('20', 'A', 'CLIP'))} e "
            f"{pc(g('40', 'A', 'CLIP'))}). "
            f"<b>Duas ressalvas.</b> O pré-treino linear parou na 2ª iteração e explicou pouco da validação (R² de "
            f"{FRR_MS_R2_PRETREINO} nos sujeitos 2 a 8): a conclusão vale para esse pré-treino, não para dados de outros "
            f"sujeitos em geral. E é uma semente só, sem intervalo nas métricas de imagem: diferenças de poucos "
            f"milésimos, como A contra C com 40 sessões, não valem como ordem. Detalhes em "
            f'<a href="{REPO_URL}/blob/main/docs/FRR_MULTISUJEITO.md">docs/FRR_MULTISUJEITO.md</a>.')
    return textos


def leitura(dados):
    """As conclusoes escritas a mao mais a do FRR, quando ha numeros."""
    return LEITURA + leitura_frr(dados)
