"""O custo de cada etapa por modelo: a aba Tempos do site.

O treino vem do registro do modelo (tempo_s, gravado quando ele treinou). Reconstrucao, refino e avaliacao
vem de results/tables/tempos_etapas.json (src/etapas/mede_tempos.py), medidos uma vez por perfil: o custo
deles depende do tipo de modelo, nao do numero de sessoes nem dos pesos.
"""
import json

from mindeye_ridge import paths


def carrega(caminho=None):
    """O JSON do mede_tempos.py; None se ele ainda nao rodou."""
    p = caminho or paths.TABLES / "tempos_etapas.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def perfil(m):
    """Qual medicao vale para o modelo."""
    if m.get("familia") == "frr":
        return "frr"
    if m.get("familia") == "mindeye1":
        return "mindeye1_semlowlevel" if m["id"].endswith("_semlowlevel") else "mindeye1"
    hiper = (m.get("config") or {}).get("hiper") or {}
    return "mindeye2_4096_blurry" if hiper.get("hidden_dim") == 4096 or "4096" in m["id"] else "mindeye2_1024"


def _avaliacao(m, p, etapas):
    """Retrieval mais as tabelas de metricas que o modelo tem; None se faltar alguma medida."""
    metricas = etapas.get("metricas") or {}
    if m.get("familia") == "mindeye1":          # o retrieval sai do final_evaluations, e nao ha refino nem legendas
        partes = [metricas.get("mindeye1/unclip")]
    else:
        partes = [metricas.get("bigg/unclip"), metricas.get("bigg/refinada")]
        if not m.get("publicada"):              # as publicadas nao tem checkpoint: so as tabelas
            partes.append((etapas.get("retrieval") or {}).get(p))
    if not all(partes):
        return None
    return {"s": round(sum(x["s"] for x in partes), 1), "origem": "medido"}


def etapas(m, medidas):
    """Os tempos do modelo: {treino, reconstrucao, refino, avaliacao: {s, origem} ou None, total_s, ...}."""
    p = perfil(m)
    dados = (medidas or {}).get("etapas") or {}

    def estimado(etapa):
        x = None if m.get("publicada") else (dados.get(etapa) or {}).get(p)
        return {"s": x["s"], "origem": x["origem"]} if x else None

    treino = {"s": m["tempo_s"], "origem": "medido"} if m.get("tempo_s") is not None else None
    r = {"treino": treino, "reconstrucao": estimado("reconstrucao"), "refino": estimado("refino"),
         "avaliacao": _avaliacao(m, p, dados), "perfil": p,
         "pretreino_s": ((m.get("frr") or {}).get("custo") or {}).get("segundos_pretreino")}
    # sem o treino ou sem reconstrucao o total enganaria: pareceria o custo do modelo inteiro
    faltam = treino is None or r["reconstrucao"] is None or r["avaliacao"] is None
    r["total_s"] = None if faltam else round(sum(x["s"] for x in (treino, r["reconstrucao"], r["refino"], r["avaliacao"]) if x), 1)
    return r
