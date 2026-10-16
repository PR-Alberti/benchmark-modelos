"""Catalogo dos modelos do site (results/site): descobre cada modelo treinado nos lugares padrao,
junta configuracao e resultados e guarda um registro por modelo.

Um modelo entra no site quando tem resultado em algum destes lugares:
    results/tables/<id>_frr.json                               FRR (run_frr.py)
    results/tables/<id>_all_{recons,enhancedrecons}.csv        metricas das reconstrucoes (run_evals.sh)
    results/evals/<id>/<id>_retrieval.json                     so o retrieval (verify_retrieval.py)
    train_logs/<id>/metrics.csv                                curva do treino do MindEye2
A configuracao vem de train_logs/<id>/experimento.json (gravado pelo treina.py), do dicionario em
src/experimentos.py ou da lista CONHECIDOS abaixo; sem nenhum dos tres, o modelo aparece marcado como
sem configuracao registrada. Os experimentos de src/experimentos.py que ainda nao tem resultado
aparecem como planejados.

O registro de cada modelo (results/site/modelos/<id>.json, no git) guarda tudo o que foi extraido.
Tensores, curvas e checkpoints ficam fora do git: numa maquina sem eles, o registro mantem o que foi
extraido onde eles existiam, e o site nao perde informacao ao ser regenerado em outro computador.
Para tirar um modelo do site, apague o registro dele.
"""
import csv
import json
import os
import re
from pathlib import Path

from mindeye_ridge import paths

from .config import CHAVES_LEGENDA, MODELOS as MODELOS_BENCHMARK, fonte as fonte_de

PUBLICADA = "__publicada"      # sufixo das reconstrucoes divulgadas pelos autores do artigo

# tabela do final_evaluations.py -> (tipo de reconstrucao, e a publicada pelos autores?)
SUFIXOS = [("_published_all_enhancedrecons", "refinada", True), ("_published_all_recons", "unclip", True),
           ("_all_enhancedrecons", "refinada", False), ("_all_recons", "unclip", False)]

# colunas do metrics.csv do train_ridgeonly.py que vao para o site
CURVA = [("test/test_fwd_pct_correct", "teste_fwd"), ("test/test_bwd_pct_correct", "teste_bwd"),
         ("train/fwd_pct_correct", "treino_fwd"), ("train/bwd_pct_correct", "treino_bwd"),
         ("train/loss", "treino_loss"), ("test/loss", "teste_loss"),
         ("train/loss_prior", "treino_loss_prior"), ("test/loss_prior", "teste_loss_prior"),
         ("test/recon_cossim", "teste_cos_prior"), ("test/blurry_pixcorr", "teste_pixcorr_borrada"),
         ("train/lr", "lr")]

# Rotulos dos modelos do benchmark, na ordem em que aparecem; os outros ganham um rotulo montado da
# configuracao (rotulo_automatico) ou o "rotulo" do dicionario do experimento.
ROTULOS = {
    "subj01_frr_1sess": "FRR · 1 sessão",
    "subj01_frr_1sess_global": "FRR fração global · 1 sessão",
    "subj01_frr_1sess_ext": "FRR grade estendida · 1 sessão",
    "subj01_frr_40sess": "FRR · 40 sessões",
    "subj01_frr_40sess_global": "FRR fração global · 40 sessões",
    "subj01_frr_40sess_ext": "FRR grade estendida · 40 sessões",
    "subj01_ridgeonly_1sess_prior": "Ridge 1024 + prior · 1 sessão",
    "subj01_ridgeonly_1sess": "Ridge 1024 sem prior · 1 sessão",
    "subj01_ridgeonly_1sess_4096blurry": "Ridge 4096 + blurry · 1 sessão",
    "subj01_ridgeonly_1sess_prior_seed42": "Ridge 1024 + prior · 1 sessão · semente 42",
    "subj01_ridgeonly_1sess_prior_seed1": "Ridge 1024 + prior · 1 sessão · semente 1",
    "subj01_ridgeonly_1sess_prior_seed2": "Ridge 1024 + prior · 1 sessão · semente 2",
    "subj01_ridgeonly_1sess_noprior_seed42": "Ridge 1024 sem prior · 1 sessão · semente 42",
    "subj01_ridgeonly_1sess_prior_rseed7": "Ridge 1024 + prior · 1 sessão · reconstrução semente 7",
    "subj01_ridgeonly_40sess_prior": "Ridge 1024 + prior · 40 sessões",
    "subj01_ridgeonly_40sess_4096blurry": "Ridge 4096 + blurry · 40 sessões",
    "subj01_me1_1sess": "MindEye1 · 1 sessão",
    "subj01_me1_1sess_semlowlevel": "MindEye1 sem low-level · 1 sessão",
    "subj01_me1_40sess": "MindEye1 · 40 sessões",
    "subj01_me1_40sess_semlowlevel": "MindEye1 sem low-level · 40 sessões",
    "final_subj01_pretrained_1sess_24bs": "Artigo · fine-tune completo · 1 sessão",
    "final_subj01_pretrained_1sess_24bs" + PUBLICADA: "Artigo · reconstruções publicadas · 1 sessão",
    "final_subj01_pretrained_40sess_24bs": "Artigo · fine-tune completo · 40 sessões",
    "final_subj01_pretrained_40sess_24bs" + PUBLICADA: "Artigo · reconstruções publicadas · 40 sessões",
}
ORDEM = list(ROTULOS)


def _artigo(sessoes):
    return {"modelo": "mindeye2", "hiper": {"hidden_dim": 4096, "blurry": True, "prior": True, "num_epochs": 150,
                                            "batch_size": 24, "treino": "fine-tune completo"},
            "dataset": {"sessoes": sessoes}}


# Modelos fora de src/experimentos.py com configuracao conhecida: os do artigo (baixados), a
# reconstrucao com outra semente e o MindEye1 sem low-level (run_me1_benchmark.sh LOWLEVEL=0).
# "config" None = a do modelo em "base" (src/experimentos.py).
CONHECIDOS = {
    "final_subj01_pretrained_1sess_24bs": {"config": _artigo(1), "familia": "artigo", "notas": [
        "Modelo publicado no artigo do MindEye2, baixado e reconstruído aqui com o mesmo pipeline dos "
        "outros: fine-tune completo a partir do pré-treino multi-sujeito, 150 épocas em batch 24 numa "
        "máquina com 8 A100."]},
    "final_subj01_pretrained_40sess_24bs": {"config": _artigo(40), "familia": "artigo", "notas": [
        "Modelo publicado no artigo do MindEye2, baixado e reconstruído aqui com o mesmo pipeline dos "
        "outros: fine-tune completo, 150 épocas em batch 24 numa máquina com 8 A100."]},
    "final_subj01_pretrained_1sess_24bs" + PUBLICADA: {"config": _artigo(1), "familia": "artigo", "notas": [
        "Os tensores de reconstrução divulgados pelos autores, avaliados com as mesmas métricas: "
        "conferem o pipeline de reconstrução deste repositório."]},
    "final_subj01_pretrained_40sess_24bs" + PUBLICADA: {"config": _artigo(40), "familia": "artigo", "notas": [
        "Os tensores de reconstrução divulgados pelos autores, avaliados com as mesmas métricas: "
        "conferem o pipeline de reconstrução deste repositório."]},
    "subj01_ridgeonly_1sess_prior_rseed7": {"base": "subj01_ridgeonly_1sess_prior", "notas": [
        "O mesmo checkpoint do Ridge 1024 + prior de 1 sessão, reconstruído com a semente 7 em vez da 42: "
        "mede quanto as métricas de imagem mudam só por reamostrar a reconstrução."]},
    "subj01_me1_1sess_semlowlevel": {"base": "subj01_me1_1sess", "notas": [
        "O alto nível foi treinado de novo com a mesma receita do MindEye1 de 1 sessão (o retrieval saiu "
        "idêntico); a reconstrução usa só o Versatile Diffusion, sem o img2img a partir da imagem "
        "borrada do low-level (scripts/run_me1_benchmark.sh com LOWLEVEL=0)."]},
    "subj01_me1_40sess_semlowlevel": {"base": "subj01_me1_40sess", "notas": [
        "O alto nível foi treinado de novo com a mesma receita do MindEye1 de 40 sessões (o retrieval "
        "saiu idêntico); a reconstrução usa só o Versatile Diffusion, sem o img2img a partir da imagem "
        "borrada do low-level (scripts/run_me1_benchmark.sh com LOWLEVEL=0)."]},
}

NOTAS = {
    "subj01_ridgeonly_1sess_prior": [
        "Checkpoint do release checkpoints-v1, treinado com uma versão anterior do código; o retreino "
        "com o código atual é o de semente 42."],
    "subj01_ridgeonly_1sess": [
        "Checkpoint do release checkpoints-v1, treinado com uma versão anterior do código e outro script; "
        "o controle com o código atual é o sem prior de semente 42."],
    "subj01_ridgeonly_1sess_prior_seed42": [
        "Retreino do Ridge 1024 + prior com o código atual. Junto das sementes 1 e 2, mede o ruído do treino."],
    "subj01_ridgeonly_1sess_noprior_seed42": [
        "O mesmo script e a mesma semente do retreino com prior, só sem a loss do prior: isola o efeito "
        "do prior no treino da ridge."],
}

ESPACO_CLIP = {"frr": "ViT-bigG/14", "mindeye2": "ViT-bigG/14", "artigo": "ViT-bigG/14", "mindeye1": "ViT-L/14"}


# --------------------------------------------------------------------------- leitura dos arquivos

def _le_tabela(p):
    """Tabela de metricas do final_evaluations.py: {metrica: valor}, ou None."""
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)
        return {k: float(v) for k, v in r if k}


def _le_legendas(p):
    """*_caption_metrics.csv: so a coluna de valores, na ordem de CHAVES_LEGENDA."""
    if not p.exists():
        return None
    valores = [float(v) for v in p.read_text(encoding="utf-8").split()[1:]]
    return dict(zip(CHAVES_LEGENDA, valores)) if len(valores) == len(CHAVES_LEGENDA) else None


def _le_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _le_curva(p):
    """metrics.csv do train_ridgeonly.py: epocas (a partir de 1), series escolhidas e o tempo total."""
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        # um treino em andamento pode estar no meio de uma linha: so as linhas completas
        linhas = [r for r in csv.DictReader(f) if r.get("epoch") and None not in r.values()]
    if not linhas:
        return None
    series = {}
    for coluna, chave in CURVA:
        try:
            vals = [round(float(r[coluna]), 5) for r in linhas]
        except (KeyError, TypeError, ValueError):
            continue
        if any(vals):                               # colunas zeradas (ramo blurry desligado) ficam de fora
            series[chave] = vals
    tempos = [r.get("time/epoch_s") for r in linhas]
    segundos = sum(float(t) for t in tempos) if all(tempos) else None
    return {"epocas": [int(r["epoch"]) + 1 for r in linhas], "series": series, "segundos": segundos}


def _assinatura(p):
    try:
        s = Path(p).stat()
        return [s.st_size, s.st_mtime_ns]
    except OSError:
        return None


def _params(ckpt, antigo):
    """Parametros por modulo do last.pth; reaproveita o registro se o arquivo nao mudou."""
    sig = _assinatura(ckpt)
    if sig is None:
        return None, None
    if antigo and antigo.get("_params_sig") == sig and antigo.get("params"):
        return antigo["params"], sig
    try:
        import torch
        try:
            ck = torch.load(str(ckpt), map_location="cpu", mmap=True)
        except RuntimeError:                          # checkpoint antigo nao aceita mmap
            ck = torch.load(str(ckpt), map_location="cpu")
        grupos = {}
        for k, v in ck["model_state_dict"].items():
            g = k.split(".")[0]
            grupos[g] = grupos.get(g, 0) + v.numel()
        return grupos, sig
    except Exception as e:                            # so enfeita a ficha; nao derruba o site
        print(f"aviso: parametros de {ckpt} ({e})")
        return None, None


def _tempo_me1(etapas, log):
    """Duracao somada das etapas no log do run_me1_benchmark.sh, em segundos; None se faltar alguma."""
    import datetime
    if not log.exists():
        return None
    marcas = {}
    for linha in log.read_text(encoding="utf-8").splitlines():
        r = re.match(r"\[(.{19})\] (\S+): treino .*: (comecando|feito)$", linha)
        if r:
            marcas[(r[2], r[3])] = datetime.datetime.strptime(r[1], "%Y-%m-%d %H:%M:%S")
    try:
        return sum((marcas[(e, "feito")] - marcas[(e, "comecando")]).total_seconds() for e in etapas)
    except KeyError:
        return None


# --------------------------------------------------------------------------- configuracao

def _todos():
    try:
        from experimentos import BENCHMARK, EXPERIMENTOS
    except ImportError:
        return {}, {}
    return BENCHMARK, EXPERIMENTOS


def config_efetiva(cfg):
    """Hiperparametros completados com os padroes do modelo e o dataset com todos os campos.

    Devolve (config, erro). Uma configuracao que o treina.py recusaria volta como veio, com o erro.
    """
    if not cfg:
        return None, None
    try:
        import treina
        from mindeye_ridge import dataset_controlado as dc
        modelo = treina.MODELOS[cfg["modelo"]]
        hiper = modelo.completa(cfg.get("hiper"))
        dataset = dc.ConfigDataset.de_dict(cfg.get("dataset")).como_dict()
        return {**cfg, "hiper": hiper, "dataset": dataset}, None
    except Exception as e:
        return cfg, str(e)


def _milhar(n):
    return f"{n:,}".replace(",", ".")


AGREGACAO_TXT = {"exibicoes": "exibições separadas", "media": "média das repetições",
                 "sorteio": "repetição sorteada", "combinacao": "combinação das repetições"}


def rotulo_automatico(cfg):
    """Rotulo legivel de um experimento a partir da configuracao (efetiva)."""
    m, h, d = cfg["modelo"], cfg.get("hiper") or {}, cfg.get("dataset") or {}
    if m == "frr":
        nome = "FRR"
        if h.get("global_fraction"):
            nome += " fração global"
        if h.get("grid") == "extended":
            nome += " grade estendida"
    elif m == "mindeye2":
        nome = f"Ridge {h.get('hidden_dim', 1024)}" + (" + blurry" if h.get("blurry") else "")
        nome += " sem prior" if h.get("prior") is False else (" + prior" if not h.get("blurry") else "")
    else:
        nome = "MindEye1"
    partes = [nome]
    classes = d.get("classes")
    if classes:
        partes.append(", ".join(classes))
    if d.get("n_imagens"):
        img = f"{_milhar(d['n_imagens'])} img" + ("/classe" if d.get("balanceado") else "")
        partes.append(img + (f" × {d['repeticoes']}" if d.get("repeticoes") else ""))
    elif d.get("repeticoes"):
        rep = d["repeticoes"]
        partes.append(f"{rep} repetição" if rep == 1 else f"{rep} repetições")
    if d.get("agregacao"):
        partes.append(AGREGACAO_TXT.get(d["agregacao"], d["agregacao"]))
    s = d.get("sessoes", 40)
    partes.append(f"{s} sessão" if s == 1 else f"{s} sessões")
    if h.get("seed", 42) != 42:
        partes.append(f"semente {h['seed']}")
    return " · ".join(partes)


# --------------------------------------------------------------------------- descoberta

def _dirs():
    return {"tables": paths.TABLES, "evals": paths.EVALS, "train_logs": paths.TRAIN_LOGS,
            "me1": paths.REPO / "mindeye1" / "train_logs", "logs": paths.LOGS, "metrics": paths.METRICS}


def ids_com_resultado(d=None):
    """Ids de modelo com algum resultado em disco (ver o docstring do modulo)."""
    d = d or _dirs()
    ids = set()
    if d["tables"].exists():
        for p in d["tables"].glob("*_frr.json"):
            ids.add(p.name[:-len("_frr.json")])
        for p in d["tables"].glob("*.csv"):
            nome = p.stem
            if nome.endswith("_caption_metrics"):
                continue
            for suf, _, pub in SUFIXOS:
                if nome.endswith(suf):
                    ids.add(nome[:-len(suf)] + (PUBLICADA if pub else ""))
                    break
    if d["evals"].exists():
        for p in d["evals"].iterdir():
            if (p / f"{p.name}_retrieval.json").exists():
                ids.add(p.name)
    if d["train_logs"].exists():
        for p in d["train_logs"].iterdir():
            if (p / "metrics.csv").exists() or (p / "experimento.json").exists():
                ids.add(p.name)
    return ids


def _familia(mid, cfg, d):
    base = mid[:-len(PUBLICADA)] if mid.endswith(PUBLICADA) else mid
    if mid in CONHECIDOS and CONHECIDOS[mid].get("familia"):
        return CONHECIDOS[mid]["familia"]
    if base.startswith("final_subj01_pretrained"):
        return "artigo"
    if cfg and cfg.get("modelo") in ("frr", "mindeye2", "mindeye1"):
        return cfg["modelo"]
    if (d["tables"] / f"{base}_frr.json").exists():
        return "frr"
    if (d["me1"] / base).exists():
        return "mindeye1"
    return "mindeye2"


def extrai(mid, antigo=None, d=None, benchmark=None, experimentos=None, resumos=None):
    """Registro novo de um modelo, a partir do que estiver em disco agora."""
    d = d or _dirs()
    if benchmark is None or experimentos is None:
        benchmark, experimentos = _todos()
    antigo = antigo or {}
    publicada = mid.endswith(PUBLICADA)
    base = mid[:-len(PUBLICADA)] if publicada else mid
    r = {"id": mid}

    # configuracao: a do treina.py, a do dicionario, ou a conhecida
    exp_json = _le_json(d["train_logs"] / mid / "experimento.json")
    rotulo_cfg = None
    if exp_json:
        cfg = {k: exp_json[k] for k in ("modelo", "hiper", "dataset") if k in exp_json}
        fonte, rotulo_cfg = "train_logs/%s/experimento.json" % mid, exp_json.get("rotulo")
        r["commit"] = exp_json.get("commit")
        r["resumo_dataset"] = exp_json.get("resumo_dataset")
    elif mid in benchmark or mid in experimentos:
        cfg = (benchmark.get(mid) or experimentos.get(mid))
        rotulo_cfg = cfg.get("rotulo")
        fonte = "src/experimentos.py"
    elif mid in CONHECIDOS:
        k = CONHECIDOS[mid]
        cfg = k.get("config") or benchmark.get(k.get("base"))
        fonte = "conhecida" if k.get("config") else f"a de {k['base']} (src/experimentos.py)"
    else:
        cfg, fonte = None, None
    familia = _familia(mid, cfg, d)
    descritiva = familia == "artigo"
    efetiva, erro = (cfg, None) if descritiva else config_efetiva(cfg)
    r.update(familia=familia, config=efetiva, config_fonte=fonte, config_erro=erro,
             config_descritiva=descritiva, publicada=publicada,
             grupo="benchmark" if (mid in ROTULOS or mid in benchmark) else "experimento")
    if mid in ROTULOS:
        r["rotulo"] = ROTULOS[mid]
    elif rotulo_cfg:
        r["rotulo"] = rotulo_cfg
    elif efetiva and not erro:
        r["rotulo"] = rotulo_automatico(efetiva)
    else:
        r["rotulo"] = mid
    sessoes = (efetiva or {}).get("dataset", {}).get("sessoes") if efetiva else None
    if sessoes is None:
        m = re.search(r"_(\d+)sess", base)
        sessoes = int(m[1]) if m else None
    r["sessoes"] = sessoes
    r["espaco_clip"] = ESPACO_CLIP[familia]
    r["notas"] = list(NOTAS.get(mid, [])) + list(CONHECIDOS.get(mid, {}).get("notas", []))

    # resultados
    fontes = []
    frr = _le_json(d["tables"] / f"{mid}_frr.json") if not publicada else None
    valores = {"refinada": None, "unclip": None}
    for suf, tipo, pub in SUFIXOS:
        if pub != publicada:
            continue
        p = d["tables"] / f"{base}{suf}.csv"
        t = _le_tabela(p)
        if t:
            valores[tipo] = t
            fontes.append(f"results/tables/{p.name}")
    if familia == "mindeye1" and valores["unclip"] and not valores["refinada"]:
        valores["refinada"] = dict(valores["unclip"])     # sem refinamento: a reconstrucao final e a mesma
    legendas = None
    if familia in ("mindeye2", "artigo") and not publicada:
        for suf in ("_all_enhancedrecons", "_all_recons"):
            p = d["tables"] / f"{base}{suf}_caption_metrics.csv"
            legendas = _le_legendas(p)
            if legendas:
                break

    retrieval, centrado = None, None
    if frr:
        fontes.append(f"results/tables/{mid}_frr.json")
        retrieval = {**{k: frr["retrieval"][k] for k in ("fwd", "bwd")}, "fonte": "frr"}
        centrado = frr.get("retrieval_centrado")
    else:
        vr = _le_json(d["evals"] / base / f"{base}_retrieval.json")
        if vr and not publicada:
            fontes.append(f"results/evals/{base}/{base}_retrieval.json")
            retrieval = {"fwd": vr["fwd"], "bwd": vr["bwd"], "fonte": "verify_retrieval"}
        else:
            t = valores["refinada"] or valores["unclip"]
            if t and "FwdRetrieval" in t:
                retrieval = {"fwd": {"media": t["FwdRetrieval"]}, "bwd": {"media": t["BwdRetrieval"]},
                             "fonte": "final_evaluations"}

    curva = None if publicada else _le_curva(d["train_logs"] / mid / "metrics.csv")
    if curva:
        fontes.append(f"train_logs/{mid}/metrics.csv")
        if not retrieval:
            retrieval = {"fwd": {"media": curva["series"]["teste_fwd"][-1]},
                         "bwd": {"media": curva["series"]["teste_bwd"][-1]},
                         "fonte": "treino", "epoca": curva["epocas"][-1]}
    # todo modelo com retrieval o tem nas duas vistas da tabela (refinadas e unCLIP)
    if retrieval:
        ret = {"FwdRetrieval": retrieval["fwd"]["media"], "BwdRetrieval": retrieval["bwd"]["media"]}
        for tipo in ("refinada", "unclip"):
            if valores[tipo] is None:
                valores[tipo] = dict(ret)
            else:
                for k, v in ret.items():
                    valores[tipo].setdefault(k, v)

    r.update(valores=valores if any(valores.values()) else None, legendas=legendas, retrieval=retrieval,
             retrieval_centrado=centrado, curva=curva)
    if frr:
        r["frr"] = {k: frr.get(k) for k in ("config", "parametros_lineares", "custo", "cv", "hubness", "embedding")}

    # custo e tamanho
    tempo = None
    if frr:
        tempo = frr["custo"].get("segundos_ajuste")
    elif familia == "mindeye2" and curva:
        tempo = curva["segundos"]
    elif familia == "mindeye1":
        if base.startswith("repro_"):                    # alto nivel e low-level, com o prefixo
            etapas = [base, base.replace("subj01_me1_", "subj01_me1_lowlevel_", 1)]
        else:
            etapas = [base] if (base.endswith("_semlowlevel") or exp_json) else \
                [base, f"subj01_me1_lowlevel_{sessoes}sess"]
        tempo = _tempo_me1(etapas, d["logs"] / "me1_benchmark.log")
    r["tempo_s"] = tempo
    if not publicada and familia != "frr":
        pasta = d["me1"] if familia == "mindeye1" else d["train_logs"]
        if mid in CONHECIDOS and CONHECIDOS[mid].get("base") and familia == "mindeye2":
            params, sig = None, None                     # reconstrucao de outro checkpoint
        else:
            params, sig = _params(pasta / base / "last.pth", antigo)
        r["params"], r["_params_sig"] = params, sig
    ref = next((m for m in MODELOS_BENCHMARK if m["id"] == base), None)
    if ref:
        r["receita"] = {"treino": ref.get("treino"), "partida": ref.get("partida"), "prior": ref.get("prior")}

    # tamanho do dataset de treino
    if not r.get("resumo_dataset"):
        dsj = _le_json(d["train_logs"] / mid / "dataset.json")
        if dsj and dsj.get("resumo"):
            r["resumo_dataset"] = dsj["resumo"]
        elif efetiva and not erro and resumos is not None and _completo(efetiva.get("dataset")):
            r["resumo_dataset"] = resumos(efetiva["dataset"]["sessoes"])
    r["fontes"] = fontes
    r["concluido"] = (d["train_logs"] / mid / ".treino_completo").exists() or None
    r["iniciado"] = bool(exp_json) or None
    return r


def _completo(ds):
    """O dataset e o completo das N primeiras sessoes (o do benchmark)?"""
    if not ds:
        return False
    padrao = {"n_imagens": None, "repeticoes": None, "classes": None, "regra": "exclusiva"}
    return all(ds.get(k, v) == v for k, v in padrao.items())


def status(r):
    """completo | retrieval | treino | planejado, e o texto para o site."""
    v = r.get("valores") or {}
    imagem = any(t and any(k not in ("FwdRetrieval", "BwdRetrieval") for k in t) for t in v.values())
    if r["familia"] == "frr" and r.get("retrieval"):
        return "completo", "retrieval e embedding"
    if imagem:
        return "completo", "métricas completas"
    if r.get("retrieval") and r["retrieval"].get("fonte") != "treino":
        return "retrieval", "só retrieval"
    if r.get("curva"):
        if r.get("concluido"):
            return "treino", "treinado, sem avaliação"
        return "treino", f"em treino · época {r['curva']['epocas'][-1]}"
    if r.get("concluido"):
        return "treino", "treinado, sem avaliação"
    if r.get("iniciado"):
        return "treino", "treino iniciado"
    return "planejado", "sem resultados"


# de onde veio o retrieval, da pior para a melhor fonte
FONTE_RETRIEVAL = {"treino": 1, "final_evaluations": 2, "verify_retrieval": 3, "frr": 3}


def mescla(antigo, novo):
    """O novo por cima do antigo: o que o novo nao achou (None) fica como estava.

    Numa maquina sem train_logs/ ou sem os tensores de avaliacao, uma fonte pior nao substitui a melhor
    que o registro ja tinha: o retrieval da curva de treino (300 imagens fixas, ultima epoca) ou o das
    tabelas (sem intervalo) nao passam por cima do medido com intervalo, e a configuracao do dicionario
    nao passa por cima da gravada pelo treina.py (nem o rotulo, o commit e o resumo do dataset dela).
    """
    if not isinstance(antigo, dict) or not isinstance(novo, dict):
        return novo if novo is not None else antigo
    novo = dict(novo)
    ra, rn = antigo.get("retrieval") or {}, novo.get("retrieval") or {}
    if ra and rn and FONTE_RETRIEVAL.get(rn.get("fonte"), 0) < FONTE_RETRIEVAL.get(ra.get("fonte"), 0):
        novo["retrieval"], novo["valores"] = None, None
    if (str(antigo.get("config_fonte") or "").startswith("train_logs/")
            and not str(novo.get("config_fonte") or "").startswith("train_logs/")):
        for k in ("config", "config_fonte", "config_erro", "rotulo", "commit", "resumo_dataset"):
            novo.pop(k, None)
    out = dict(antigo)
    for k, v in novo.items():
        if v is None:
            continue
        if k in ("valores",) and isinstance(v, dict):
            out[k] = {t: (v.get(t) if v.get(t) is not None else (antigo.get(k) or {}).get(t))
                      for t in set(v) | set(antigo.get(k) or {})}
        elif k == "fontes":
            out[k] = sorted(set(antigo.get(k) or []) | set(v))
        else:
            out[k] = v
    return out


def _grava(p, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def resumo_completo_factory():
    """resumos(sessoes): o resumo do dataset completo das N primeiras sessoes, lido uma vez por N."""
    cache = {}

    def resumos(sessoes):
        if sessoes not in cache:
            try:
                from mindeye_ridge import dataset_controlado as dc, nsd_data
                ex = nsd_data.exibicoes_treino(paths.DATA, 1, sessoes)
                ex = {**ex, "rotulo": [None] * len(ex["imagem"])}
                cache[sessoes] = dc.resumo(ex)
            except Exception:
                cache[sessoes] = None
        return cache[sessoes]
    return resumos


def como_referencia(mid, repro, r):
    """O registro da reproducao `repro` com o id e o rotulo do modelo de referencia `mid`.

    Nao se mescla com o registro antigo de `mid`: ele e da corrida original, e o que a reproducao nao
    mediu nao deve sobrar dele. A corrida original fica no historico do git (docs/REEXECUTAR.md compara).
    """
    r = dict(r)
    r.update(id=mid, grupo="benchmark", reproducao=repro, rotulo=ROTULOS.get(mid, r["rotulo"]),
             notas=list(NOTAS.get(mid, [])) + list(CONHECIDOS.get(mid, {}).get("notas", [])))
    return r


def atualiza(pasta_registros, d=None, resumos=None):
    """Atualiza os registros em `pasta_registros` e devolve (modelos, planejados).

    modelos: os registros de todos os modelos com resultado (os de agora e os ja registrados).
    planejados: os experimentos de src/experimentos.py ainda sem nenhum resultado.
    """
    d = d or _dirs()
    pasta_registros = Path(pasta_registros)
    pasta_registros.mkdir(parents=True, exist_ok=True)
    benchmark, experimentos = _todos()
    registrados = {p.stem: p for p in pasta_registros.glob("*.json")}
    ids = ids_com_resultado(d) | set(registrados)
    modelos = []
    for mid in sorted(ids):
        antigo = _le_json(registrados[mid]) if mid in registrados else None
        repro = fonte_de(mid, d["tables"])
        if repro != mid:         # modelo de referencia com reproducao: o registro e o dela
            antigo_repro = _le_json(registrados[repro]) if repro in registrados else None
            novo = extrai(repro, antigo_repro, d, benchmark, experimentos, resumos)
            r = como_referencia(mid, repro, mescla(antigo_repro, novo) if antigo_repro else novo)
        else:
            novo = extrai(mid, antigo, d, benchmark, experimentos, resumos)
            r = mescla(antigo, novo) if antigo else novo
        r["status"], r["status_txt"] = status(r)
        r = json.loads(json.dumps(r, sort_keys=True))     # a mesma ordem venha do disco ou da extracao
        if r["status"] == "planejado" and not antigo:
            continue
        if r != antigo:
            _grava(pasta_registros / f"{mid}.json", r)
        modelos.append(r)
    feitos = {m["id"] for m in modelos}
    planejados = []
    for mid, cfg in {**benchmark, **experimentos}.items():
        if mid in feitos:
            continue
        efetiva, erro = config_efetiva(cfg)
        planejados.append({"id": mid, "familia": cfg["modelo"], "config": efetiva, "config_erro": erro,
                           "rotulo": cfg.get("rotulo") or (rotulo_automatico(efetiva) if not erro else mid),
                           "grupo": "benchmark" if mid in benchmark else "experimento"})
    ordem = {m: i for i, m in enumerate(ORDEM)}
    modelos.sort(key=lambda m: (ordem.get(m["id"], len(ordem)), m.get("sessoes") or 0, m["rotulo"]))
    return modelos, planejados
