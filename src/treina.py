#!/usr/bin/env python3
"""Treina um experimento de src/experimentos.py: modelo + hiperparametros + dataset controlado.

    python src/treina.py --lista
    python src/treina.py subj01_frr_5classes_300img_1rep --so_dataset
    python src/treina.py subj01_frr_5classes_300img_1rep --dry_run
    python src/treina.py subj01_frr_5classes_300img_1rep

O que acontece:
1. o dicionario "dataset" vira um ConfigDataset e o subconjunto e sorteado
   (mindeye_ridge.dataset_controlado), gravado em train_logs/<nome>/dataset.json;
2. os hiperparametros sao completados com os padroes do modelo (MODELOS, abaixo) e tudo vai
   para train_logs/<nome>/experimento.json, com o commit do codigo;
3. o script de treino do modelo roda com DATASET apontando para o manifesto:
   frr -> scripts/run_frr.sh, mindeye2 -> scripts/run_ridgeonly_prior.sh,
   mindeye1 -> scripts/me1_run.sh train.

Um nome de experimento nao muda de configuracao: se train_logs/<nome> ja tem outro dataset ou
outros hiperparametros, o treino para (os tres modelos retomam do que encontram em disco, e
misturar configuracoes estragaria a corrida). Mude o nome ou apague o diretorio.

Do Python (um notebook, por exemplo), o mesmo com um dicionario qualquer:

    from treina import roda
    roda("meu_exp", {"modelo": "frr", "hiper": {}, "dataset": {"sessoes": 10, "repeticoes": 1}})
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field

from mindeye_ridge import dataset_controlado as dc
from mindeye_ridge import paths


def _sim_nao(v):
    return "1" if v else "0"


def _flags(d):
    """{opcao: valor} -> argumentos de linha de comando; None fica de fora."""
    args = []
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, bool):
            args.append(f"--{k}" if v else f"--no-{k}")
        elif isinstance(v, (list, tuple)):
            args += [f"--{k}", *map(str, v)]
        else:
            args.append(f"--{k}={v}")
    return args


@dataclass
class Modelo:
    """Como cada modelo recebe os hiperparametros: variaveis do script ou opcoes do treino."""
    script: list
    padrao: dict
    traduz: callable                    # hiper completo -> (env, args extras)
    valida: callable = field(default=lambda h: None)

    def completa(self, hiper):
        hiper = dict(hiper or {})
        desconhecidos = sorted(set(hiper) - set(self.padrao))
        if desconhecidos:
            raise ValueError(f"hiperparametros desconhecidos: {desconhecidos}; validos: {sorted(self.padrao)}")
        completo = {**self.padrao, **hiper}
        self.valida(completo)
        return completo


def _frr(h):
    extras = _flags({"folds": h["folds"], "grid": h["grid"], "fracs": h["fracs"], "seed": h["seed"],
                     "chunk": h["chunk"]})
    if h["global_fraction"]:
        extras.append("--global_fraction")
    return {"EXTRA": " ".join(extras)}, []


def _mindeye2(h):
    env = {"HIDDEN_DIM": h["hidden_dim"], "NUM_EPOCHS": h["num_epochs"], "MAX_LR": h["max_lr"],
           "PRIOR": _sim_nao(h["prior"]), "BLURRY": _sim_nao(h["blurry"]), "SEED": h["seed"],
           "BATCH_SIZE": h["batch_size"], "CKPT_INTERVAL": h["ckpt_interval"],
           "FROZEN": None if h["frozen_fp16"] is None else _sim_nao(h["frozen_fp16"])}
    return {k: str(v) for k, v in env.items() if v is not None}, []


def _valida_mindeye2(h):
    if h["num_epochs"] < 2:
        raise ValueError("mindeye2: num_epochs >= 2 (o OneCycleLR usa pct_start = 2/num_epochs)")
    if h["blurry"] and h["hidden_dim"] != 4096:
        raise ValueError("mindeye2: blurry exige hidden_dim 4096 (o checkpoint final_multisubject_subj01)")


def _mindeye1(h):
    env = {"PAPER": _sim_nao(h["paper"]), "SAVE_EVERY": str(h["save_every"])}
    return env, _flags({"num_epochs": h["num_epochs"], "max_lr": h["max_lr"], "seed": h["seed"],
                        "mixup_pct": h["mixup_pct"], "batch_size": h["batch_size"]})


MODELOS = {
    # src/run_frr.py: validacao cruzada de 5 dobras sobre as 20 fracoes de Doerig et al.
    "frr": Modelo(["run_frr.sh"],
                  {"folds": 5, "grid": "doerig", "fracs": None, "global_fraction": False,
                   "seed": 42, "chunk": 8192},
                  _frr),
    # src/train_ridgeonly.py: so a ridge do subj01, a partir do pre-treino nos outros 7 sujeitos.
    # batch_size None = o padrao do script (16 no 1024, 8 no 4096 + blurry); frozen_fp16 None = idem
    "mindeye2": Modelo(["run_ridgeonly_prior.sh"],
                       {"hidden_dim": 1024, "num_epochs": 150, "batch_size": None, "max_lr": 3e-4,
                        "prior": True, "blurry": False, "frozen_fp16": None, "seed": 42,
                        "ckpt_interval": None},
                       _mindeye2, _valida_mindeye2),
    # mindeye1/src/Train_MindEye.py: treino do zero, CLIP ViT-L/14. batch_size None = 16 com AdamW de
    # 8 bits (cabe em 20 GB); paper=True usa batch 32 e AdamW normal, como no artigo
    "mindeye1": Modelo(["me1_run.sh", "train"],
                       {"num_epochs": 240, "max_lr": 3e-4, "seed": 42, "mixup_pct": 0.33,
                        "batch_size": None, "paper": False, "save_every": 1},
                       _mindeye1),
}


def _commit():
    try:
        rev = subprocess.run(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        sujo = subprocess.run(["git", "-C", str(paths.REPO), "status", "--porcelain", "--untracked-files=no"],
                              capture_output=True, text=True).stdout.strip()
        return rev + ("+alteracoes" if sujo else "")
    except OSError:
        return None


def _confere_igual(caminho, chave, novo, nome):
    with open(caminho) as f:
        antigo = json.load(f)[chave]
    if antigo != json.loads(json.dumps(novo)):
        raise SystemExit(f"{caminho} tem outra configuracao ({chave}).\n  antes: {antigo}\n  agora: {novo}\n"
                         f"Mude o nome do experimento ou apague train_logs/{nome}.")


def roda(nome, exp, so_dataset=False, dry_run=False, retoma=False):
    """Monta o dataset do experimento e treina. Devolve o resumo do dataset."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", nome):
        raise ValueError(f"nome {nome!r}: so letras, numeros, _ . -")
    sobra = set(exp) - {"modelo", "hiper", "dataset"}
    if sobra:
        raise ValueError(f"chaves desconhecidas no experimento: {sorted(sobra)} (use modelo, hiper, dataset)")
    if exp.get("modelo") not in MODELOS:
        raise ValueError(f"modelo {exp.get('modelo')!r}: use um de {sorted(MODELOS)}")
    modelo = MODELOS[exp["modelo"]]
    cfg = dc.ConfigDataset.de_dict(exp.get("dataset"))
    hiper = modelo.completa(exp.get("hiper"))

    pasta = paths.TRAIN_LOGS / nome
    manifesto, registro = pasta / "dataset.json", pasta / "experimento.json"
    if manifesto.exists():
        _confere_igual(manifesto, "config", cfg.como_dict(), nome)
        resumo = dc.carrega(manifesto)["resumo"]
        print(f"[{nome}] dataset ja montado: {manifesto}")
    else:
        sub = dc.monta(cfg, paths.DATA)
        resumo = sub["resumo"]
        if not dry_run:
            pasta.mkdir(parents=True, exist_ok=True)
            dc.salva(sub, manifesto)
    print(f"[{nome}] dataset {cfg.assinatura()}: {json.dumps(resumo, ensure_ascii=False)}")
    if so_dataset:
        return resumo

    if registro.exists():
        _confere_igual(registro, "hiper", hiper, nome)
        _confere_igual(registro, "modelo", exp["modelo"], nome)
    env, args = modelo.traduz(hiper)
    env.update({"MODEL_NAME": nome, "NUM_SESSIONS": str(cfg.sessoes), "DATASET": str(manifesto)})
    if retoma:
        env["RESUME"] = "1"
    cmd = ["bash", str(paths.REPO / "scripts" / modelo.script[0]), *modelo.script[1:], *args]
    print(f"[{nome}] {' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())} {shlex.join(cmd)}", flush=True)
    if dry_run:
        return resumo

    with open(registro, "w") as f:
        json.dump({"nome": nome, "modelo": exp["modelo"], "hiper": hiper, "dataset": cfg.como_dict(),
                   "assinatura_dataset": cfg.assinatura(), "resumo_dataset": resumo, "commit": _commit(),
                   "comando": {"env": env, "cmd": cmd}}, f, indent=1, ensure_ascii=False)
    subprocess.run(cmd, env={**os.environ, **env}, check=True)
    return resumo


def main():
    from experimentos import EXPERIMENTOS

    p = argparse.ArgumentParser(description="treina um experimento de src/experimentos.py")
    p.add_argument("nomes", nargs="*", help="experimentos a rodar, em sequencia")
    p.add_argument("--lista", action="store_true", help="lista os experimentos")
    p.add_argument("--so_dataset", action="store_true", help="monta e resume o dataset, sem treinar")
    p.add_argument("--dry_run", action="store_true", help="mostra o comando sem gravar nem treinar")
    p.add_argument("--retoma", action="store_true", help="mindeye2: continua do last.pth (RESUME=1)")
    a = p.parse_args()

    if a.lista or not a.nomes:
        for nome, exp in EXPERIMENTOS.items():
            print(f"{nome:40s} {exp['modelo']:9s} dataset={exp.get('dataset', {})}")
        return
    faltam = [n for n in a.nomes if n not in EXPERIMENTOS]
    if faltam:
        sys.exit(f"experimentos inexistentes: {faltam}; veja --lista")
    for nome in a.nomes:
        roda(nome, EXPERIMENTOS[nome], so_dataset=a.so_dataset, dry_run=a.dry_run, retoma=a.retoma)


if __name__ == "__main__":
    main()
