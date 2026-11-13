"""O custo de cada etapa por modelo: a aba Tempos do site.

O treino vem do registro do modelo (tempo_s, gravado quando ele treinou). Reconstrucao, refino e avaliacao
vem da propria corrida do modelo, se ela gravou (train_logs/<m>/tempos.jsonl, cronometra em scripts/common.sh,
que chega aqui como tempos_medidos no registro). Sem isso, valem as medidas de
results/tables/tempos_etapas.json (src/etapas/mede_tempos.py), uma por perfil: o custo dessas etapas depende
do tipo de modelo, nao do numero de sessoes nem dos pesos.
"""
import json

from mindeye_ridge import paths


def carrega(caminho=None):
    """O JSON do mede_tempos.py; None se ele ainda nao rodou."""
    p = caminho or paths.TABLES / "tempos_etapas.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# as etapas que, interrompidas, continuam do que ja fizeram: o custo delas e a soma das tentativas. Nas
# outras (metricas, retrieval e a reconstrucao do MindEye1) a tentativa que falhou nao foi aproveitada
RETOMAM = {"mindeye1": {"treino_lowlevel"}}
RETOMAM_PADRAO = {"reconstrucao", "refino"}


def le_registro(caminho, familia=None):
    """O tempos.jsonl de um modelo: {etapa: segundos}; None se ele nao gravou nada.

    O valor e None se a etapa morreu sem gravar o fim (a maquina resetou, ou ainda esta rodando) e nao ha
    medida inteira dela: quanto levou nao se sabe, e somar so o que ficou gravado daria um tempo menor.
    """
    try:
        linhas = caminho.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    retomam = RETOMAM.get(familia, RETOMAM_PADRAO)
    medidas, soma, abertas, com_furo = {}, {}, set(), set()
    for linha in linhas:
        try:
            x = json.loads(linha)
            etapa, evento = x["etapa"], x["evento"]
        except (ValueError, KeyError, TypeError):     # a linha cortada de um reset
            continue
        if evento == "inicio":
            if etapa in abertas:                      # a tentativa anterior nunca fechou
                com_furo.add(etapa)
            abertas.add(etapa)
            continue
        abertas.discard(etapa)
        if x.get("rc") != 0:
            soma[etapa] = soma.get(etapa, 0) + x["s"]
        elif etapa in retomam:
            medidas[etapa] = None if etapa in com_furo else soma.get(etapa, 0) + x["s"]
            soma.pop(etapa, None)
            com_furo.discard(etapa)
        else:
            medidas[etapa] = x["s"]
            soma.pop(etapa, None)
    for etapa in abertas:
        medidas.setdefault(etapa, None)
    return medidas or None


def perfil(m):
    """Qual medicao vale para o modelo."""
    if m.get("familia") == "frr":
        return "frr"
    if m.get("familia") == "mindeye1":
        return "mindeye1_semlowlevel" if m["id"].endswith("_semlowlevel") else "mindeye1"
    hiper = (m.get("config") or {}).get("hiper") or {}
    return "mindeye2_4096_blurry" if hiper.get("hidden_dim") == 4096 or "4096" in m["id"] else "mindeye2_1024"


def _avaliacao(m, p, etapas, proprios):
    """Retrieval mais as tabelas de metricas que o modelo tem; None se faltar alguma medida.

    Cada parte vem da corrida do modelo, se ela gravou, e senao da medida do perfil.
    """
    metricas = etapas.get("metricas") or {}

    def parte(propria, do_perfil):
        return {"s": propria} if propria is not None else do_perfil

    if m.get("familia") == "mindeye1":          # o retrieval sai do final_evaluations, e nao ha refino nem legendas
        partes = [parte(proprios.get("metricas_base"), metricas.get("mindeye1/unclip"))]
    else:
        partes = [parte(proprios.get("metricas_base"), metricas.get("bigg/unclip")),
                  parte(proprios.get("metricas_enhanced"), metricas.get("bigg/refinada"))]
        if not m.get("publicada"):              # as publicadas nao tem checkpoint: so as tabelas
            ret = proprios.get("retrieval")
            if ret is None:                     # no FRR o retrieval roda dentro do treino
                ret = ((m.get("frr") or {}).get("custo") or {}).get("segundos_retrieval")
            partes.append(parte(ret, (etapas.get("retrieval") or {}).get(p)))
    if not all(partes):
        return None
    return {"s": round(sum(x["s"] for x in partes), 1), "origem": "medido"}


def etapas(m, medidas):
    """Os tempos do modelo: {treino, reconstrucao, refino, avaliacao: {s, origem} ou None, total_s, ...}."""
    p = perfil(m)
    dados = (medidas or {}).get("etapas") or {}

    proprios = {} if m.get("publicada") else (m.get("tempos_medidos") or {})

    def estimado(etapa):
        if proprios.get(etapa) is not None:
            return {"s": proprios[etapa], "origem": "medido"}
        x = None if m.get("publicada") else (dados.get(etapa) or {}).get(p)
        return {"s": x["s"], "origem": x["origem"]} if x else None

    treino = {"s": m["tempo_s"], "origem": "medido"} if m.get("tempo_s") is not None else None
    r = {"treino": treino, "reconstrucao": estimado("reconstrucao"), "refino": estimado("refino"),
         "avaliacao": _avaliacao(m, p, dados, proprios), "perfil": p,
         "pretreino_s": ((m.get("frr") or {}).get("custo") or {}).get("segundos_pretreino")}
    # sem o treino ou sem reconstrucao o total enganaria: pareceria o custo do modelo inteiro
    faltam = treino is None or r["reconstrucao"] is None or r["avaliacao"] is None
    r["total_s"] = None if faltam else round(sum(x["s"] for x in (treino, r["reconstrucao"], r["refino"], r["avaliacao"]) if x), 1)
    return r
