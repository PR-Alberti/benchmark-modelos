"""Leitura dos resultados em disco: tabelas, curvas de treino, JSONs do FRR, galeria."""

import csv
import json
import pathlib
from mindeye_ridge import paths

from .config import ARQ, CHAVES_LEGENDA, DATA, FRR_VARIANTES, MODELOS
from .fmt import duracao, jpeg, tempo_fmt


def le_tabela(nome):
    p = paths.TABLES / f"{nome}.csv"
    if not p.exists():
        return None
    with open(p, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        return {k: float(v) for k, v in r}


def le_legendas(nome):
    p = paths.TABLES / f"{nome}_caption_metrics.csv"
    if not p.exists():
        return None
    valores = [float(v) for v in p.read_text().split()[1:]]   # 1a linha: "Value"
    return dict(zip(CHAVES_LEGENDA, valores))


def le_frr(mid):
    """JSON do run_frr.py (a copia em tables/, que vai para o git), ou None se ainda nao rodou."""
    p = paths.TABLES / f"{mid}_frr.json"
    return json.loads(p.read_text()) if p.exists() else None


def le_curva(p):
    p = pathlib.Path(p)
    if not p.exists():
        return None
    with open(p, newline="") as f:
        linhas = [r for r in csv.DictReader(f) if r.get("epoch")]
    if not linhas:
        return None
    tempo = [float(r["time/epoch_s"]) for r in linhas if r.get("time/epoch_s")]
    return {"epocas": [int(r["epoch"]) for r in linhas],
            "fwd": [round(float(r["test/test_fwd_pct_correct"]), 4) for r in linhas],
            "bwd": [round(float(r["test/test_bwd_pct_correct"]), 4) for r in linhas],
            "segundos": sum(tempo) if len(tempo) == len(linhas) else None}


def conta_params(mid):
    """Parametros por modulo (ridge/backbone/diffusion_prior) do last.pth."""
    import torch
    p = paths.TRAIN_LOGS / mid / "last.pth"
    if not p.exists():
        return None
    try:
        ck = torch.load(str(p), map_location="cpu", mmap=True)
    except RuntimeError:              # ckpt em formato antigo nao aceita mmap
        ck = torch.load(str(p), map_location="cpu")
    grupos = {}
    for k, v in ck["model_state_dict"].items():
        g = k.split(".")[0]
        grupos[g] = grupos.get(g, 0) + v.numel()
    return grupos


def ids_nsd():
    """Indice NSD (0..72999) de cada uma das 1000 imagens de teste, na ordem das recons.

    O recon_inference.py percorre np.unique(ids) -- ordenado -- entao a linha i
    de qualquer tensor de reconstrucao e o i-esimo id em ordem crescente.
    """
    try:
        import numpy as np
        import webdataset as wds
        ds = wds.WebDataset(str(DATA / "wds/subj01/new_test/0.tar"), resampled=False,
                            nodesplitter=lambda u: u) \
            .decode("torch").rename(behav="behav.npy").to_tuple("behav")
        return [int(i) for i in np.unique([int(b[0][0, 0]) for b in ds])]
    except Exception as e:           # so enfeita a galeria; nao vale derrubar o script
        print(f"aviso: sem ids NSD ({e})")
        return None


def carrega_tensor(p):
    import torch
    try:
        return torch.load(str(p), map_location="cpu", mmap=True)
    except RuntimeError:              # formato antigo nao aceita mmap
        return torch.load(str(p), map_location="cpu")


def coleta(args):
    import numpy as np
    dados = {"modelos": [], "curvas": [], "galeria": None}

    for m in MODELOS:
        mid = m["id"]
        info = dict(m)
        if m.get("frr"):
            info.update(frr_dados=le_frr(mid), tabelas={"enh": None, "base": None}, publicada=None,
                        legendas=None, params=None)
            info["tempo"] = duracao(info["frr_dados"]["custo"]["segundos_ajuste"]) if info["frr_dados"] else None
            dados["modelos"].append(info)
            continue
        info["tabelas"] = {
            "enh": le_tabela(f"{mid}_all_enhancedrecons"),
            "base": le_tabela(f"{mid}_all_recons"),
        }
        info["publicada"] = {
            "enh": le_tabela(f"{mid}_published_all_enhancedrecons"),
            "base": le_tabela(f"{mid}_published_all_recons"),
        } if m.get("paper") else None
        info["legendas"] = le_legendas(f"{mid}_all_enhancedrecons") or le_legendas(f"{mid}_all_recons")
        info["params"] = conta_params(mid)
        curva = le_curva(paths.TRAIN_LOGS / mid / "metrics.csv")
        seg = curva["segundos"] if curva else None
        info["tempo"] = tempo_fmt(seg) or m.get("tempo_doc")
        dados["modelos"].append(info)

    # curvas de treino: so as corridas que gravaram metrics.csv. A corrida
    # anterior do 4096 (guardada em metrics/) so ganha grafico proprio se diferir.
    atual = le_curva(paths.TRAIN_LOGS / "subj01_ridgeonly_1sess_4096blurry/metrics.csv")
    antes = le_curva(paths.METRICS / "subj01_ridgeonly_1sess_4096blurry.csv")
    mesma = bool(atual and antes and all(atual[k] == antes[k] for k in ("epocas", "fwd", "bwd")))
    dados["curva_repetida"] = mesma
    curvas = [("Ridge 4096 + blurry",
               "idêntica, época por época, à corrida anterior do repositório" if mesma else "esta corrida",
               atual)]
    if antes and not mesma:
        curvas.append(("Ridge 4096 + blurry", "corrida anterior (repositório)", antes))
    curvas.append(("Ridge 1024 + prior · 40 sessões", "esta corrida",
                   le_curva(paths.TRAIN_LOGS / "subj01_ridgeonly_40sess_prior/metrics.csv")))
    for sem, sub in [(42, "retreino com o código atual"), (1, "outra semente"), (2, "outra semente")]:
        curvas.append((f"Ridge 1024 + prior · semente {sem}", sub,
                       le_curva(paths.TRAIN_LOGS / f"subj01_ridgeonly_1sess_prior_seed{sem}/metrics.csv")))
    curvas.append(("Ridge 1024 sem prior · semente 42", "retreino com o código atual",
                   le_curva(paths.TRAIN_LOGS / "subj01_ridgeonly_1sess_noprior_seed42/metrics.csv")))
    dados["curvas"] = [{"titulo": t, "sub": sub, "dados": c} for t, sub, c in curvas]

    # ruido de referencia: so a semente muda, na config 1024 + prior
    def retr(m):
        pj = paths.EVALS / m / f"{m}_retrieval.json"
        r = json.loads(pj.read_text()) if pj.exists() else None
        return (r["fwd"]["media"], r["bwd"]["media"]) if r else (None, None)

    e1 = le_tabela("subj01_ridgeonly_1sess_prior_all_enhancedrecons") or {}
    e3 = le_tabela("subj01_ridgeonly_1sess_all_enhancedrecons") or {}
    treino = [("42 · checkpoint do release", e1.get("FwdRetrieval"), e1.get("BwdRetrieval"))]
    for sem, rot in [(42, "42 · retreino com o código atual"), (1, "1"), (2, "2")]:
        treino.append((rot, *retr(f"subj01_ridgeonly_1sess_prior_seed{sem}")))
    controle = [
        ("com prior · checkpoint do release", e1.get("FwdRetrieval"), e1.get("BwdRetrieval")),
        ("sem prior · checkpoint do release", e3.get("FwdRetrieval"), e3.get("BwdRetrieval")),
        ("com prior · retreino, semente 42", *retr("subj01_ridgeonly_1sess_prior_seed42")),
        ("sem prior · retreino, semente 42", *retr("subj01_ridgeonly_1sess_noprior_seed42")),
    ]
    dados["ruido"] = {
        "treino": treino,
        "controle": controle,
        "amostragem": {sem: {t: le_tabela(f"{m}_{ARQ[t]}") for t in ("enh", "base")}
                       for sem, m in [(42, "subj01_ridgeonly_1sess_prior"),
                                      (7, "subj01_ridgeonly_1sess_prior_rseed7")]},
    }

    # galeria: os mesmos estimulos para todos os modelos
    imagens = carrega_tensor(DATA / "evals/all_images.pt")
    capt_coco = np.asarray(carrega_tensor(DATA / "evals/all_captions.pt")).astype(str)
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(imagens), size=args.n, replace=False).tolist())
    nsd = ids_nsd()

    dados["frr_variantes"] = [{"id": i, "rotulo": r, "grupo": g, "dados": le_frr(i)}
                              for i, r, g in FRR_VARIANTES]

    colunas = []
    for m in MODELOS:
        mid = m["id"]
        if m.get("frr"):
            continue                      # sem reconstrucao, nao tem o que mostrar na galeria
        col = {"id": mid, "rotulo": m["rotulo"], "img": {}, "legendas": None}
        for tipo, sufixo in ARQ.items():
            p = paths.EVALS / mid / f"{mid}_{sufixo}.pt"
            if p.exists():
                t = carrega_tensor(p)
                col["img"][tipo] = [jpeg(t[i], args.lado, args.qualidade) for i in idx]
                del t
        p = paths.EVALS / mid / f"{mid}_all_predcaptions.pt"
        if p.exists():
            leg = np.asarray(carrega_tensor(p)).astype(str)
            col["legendas"] = [leg[i] for i in idx]
        colunas.append(col)

    dados["galeria"] = {
        "idx": idx,
        "nsd": [nsd[i] for i in idx] if nsd else None,
        "vista": [jpeg(imagens[i], args.lado, args.qualidade) for i in idx],
        "coco": [capt_coco[i] for i in idx],
        "colunas": colunas,
    }
    return dados
