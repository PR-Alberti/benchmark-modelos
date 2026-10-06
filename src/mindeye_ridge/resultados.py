"""O retrieval de um modelo treinado, pelo nome, de onde quer que ele esteja gravado.

Cada modelo deixa o resultado num lugar: o FRR em results/tables/<nome>_frr.json; o MindEye2 e o
MindEye1 nas metricas das reconstrucoes (results/tables/<nome>_all_recons.csv, final_evaluations),
no verify_retrieval (results/evals/<nome>/<nome>_retrieval.json) e, durante o treino do MindEye2,
no metrics.csv de cada epoca. `retrieval` le a primeira fonte que existir, nessa ordem de
preferencia, e diz qual foi: os numeros de fontes diferentes nao sao do mesmo protocolo.

    retrieval("subj01_frr_1sess")   # {"fwd": 0.55, "bwd": 0.03, "fonte": "..."}
"""
import csv
import json

from . import paths

FONTES = {
    "frr": "FRR: 30 sorteios de 300 imagens do teste",
    "verify_retrieval": "verify_retrieval.py: 30 sorteios de 300 imagens do teste",
    "final_evaluations": "final_evaluations.py: as 1.000 imagens do teste, em blocos de 300",
    "treino": "metrics.csv da ultima epoca: 300 imagens do teste, durante o treino",
}


def retrieval(nome):
    """Retrieval top-1 do modelo `nome`: {"fwd", "bwd", "fonte"}, ou None se nao houver resultado.

    fwd e imagem -> cerebro e bwd cerebro -> imagem, a convencao do codigo do MindEye2 (BENCHMARK.md).
    """
    frr = paths.TABLES / f"{nome}_frr.json"
    if frr.exists():
        r = json.loads(frr.read_text())["retrieval"]
        return {"fwd": r["fwd"]["media"], "bwd": r["bwd"]["media"], "fonte": FONTES["frr"]}
    vr = paths.EVALS / nome / f"{nome}_retrieval.json"
    if vr.exists():
        r = json.loads(vr.read_text())
        return {"fwd": r["fwd"]["media"], "bwd": r["bwd"]["media"], "fonte": FONTES["verify_retrieval"]}
    tabela = paths.TABLES / f"{nome}_all_recons.csv"
    if tabela.exists():
        with open(tabela, newline="") as f:
            m = {k: float(v) for k, v in list(csv.reader(f, delimiter="\t"))[1:]}
        if "FwdRetrieval" in m:
            return {"fwd": m["FwdRetrieval"], "bwd": m["BwdRetrieval"], "fonte": FONTES["final_evaluations"]}
    curva = paths.TRAIN_LOGS / nome / "metrics.csv"
    if curva.exists():
        with open(curva, newline="") as f:
            linhas = [r for r in csv.DictReader(f) if r.get("epoch")]
        if linhas:
            u = linhas[-1]
            return {"fwd": float(u["test/test_fwd_pct_correct"]), "bwd": float(u["test/test_bwd_pct_correct"]),
                    "fonte": FONTES["treino"] + f" ({len(linhas)} epocas)"}
    return None
