"""As conclusoes do benchmark: as escritas a mao e a do FRR, montada dos numeros medidos."""

from .fmt import fmt, virg


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
    "nível (PixCorr 0,181 → 0,253; Alex(2) 85,4% → 89,6%). No alto nível o 1024 já empata com o "
    "artigo: Incep e CLIP a menos de 1 ponto, Eff e SwAV a 0,003.",
    "<b>O prior importa no treino da ridge.</b> Com o mesmo script e a mesma semente, tirar a loss "
    "do prior derruba o retrieval (Img→cér / Cér→img) de 92,9% / 88,1% para 78,4% / 72,6%, e todas as métricas de "
    "reconstrução caem junto. O retrieval nem passa pelo prior: o que muda é o sinal de treino, "
    "já que a loss do prior supervisiona a saída inteira do backbone (256 × 1.664), e não só a "
    "projeção contrastiva.",
    "<b>40 sessões.</b> O ridge 1024 chega a 99,9% / 99,5% de retrieval e fica de 1 a 3 pontos do "
    "artigo nas identificações 2-way (CLIP 92,3% contra 93,6%). A distância maior é no baixo "
    "nível (PixCorr 0,277 contra 0,373), onde o artigo tem o ramo blurry e este não.",
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
    return textos


def leitura(dados):
    """As conclusoes escritas a mao mais a do FRR, quando ha numeros."""
    return LEITURA + leitura_frr(dados)
