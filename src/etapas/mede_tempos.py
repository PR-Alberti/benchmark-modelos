"""Quanto custa cada etapa depois do treino, com testes curtos, sem refazer nenhum modelo.

    scripts/mede_tempos.sh --fonte results/evals/<modelo com as 1.000 reconstrucoes>

Grava results/tables/tempos_etapas.json, que a aba Tempos do site le. O tempo de treino nao passa por
aqui: ele ja esta no registro de cada modelo.

O que e estimado e o que e medido:
  reconstrucao, refino   --imagens imagens e regra de 3 para as 1.000 do teste. A carga dos modelos conta
                         uma vez so, e a primeira imagem fica fora do ritmo (veja estima()).
  retrieval, metricas    rodam inteiros: o retrieval sorteia 300 imagens, nao da para reduzir a 30. As
                         metricas usam as reconstrucoes de --fonte, porque o custo vem do script e nao do
                         modelo. O que muda o custo e o alvo do retrieval: o MindEye1 grava o dele, e o
                         final_evaluations.py deixa de calcular o embedding bigG (metade do tempo).

Tudo roda com nomes tempo_teste_* em train_logs/, mindeye1/train_logs/, results/evals/ e results/tables/,
apagados no fim. As etapas sao chamadas direto, sem os scripts run_*: estes regeneram o site. O que ja foi
medido fica no JSON e nao e refeito (--refaz recomeca); os logs ficam em results/logs/mede_tempos/.
Rode com a GPU livre: outro processo nela muda os tempos.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mindeye_ridge import paths

PREFIXO = "tempo_teste_"
IMAGENS_TESTE = 1000                    # o teste do benchmark
SAIDA = paths.TABLES / "tempos_etapas.json"
LOGS = paths.LOGS / "mede_tempos"
ME1_LOGS = paths.REPO / "mindeye1" / "train_logs"

_BARRA = re.compile(r"(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<")


def progresso(texto, total):
    """[(imagens prontas, segundos)] das barras tqdm com esse total; as barras internas do sampler tem outro."""
    pontos = {}
    for r in _BARRA.finditer(texto):
        n, tot = int(r[1]), int(r[2])
        if tot == total and n >= 1:
            pontos[n] = int(r[3] or 0) * 3600 + int(r[4]) * 60 + int(r[5])
    return sorted(pontos.items())


def estima(parede, pontos, total=IMAGENS_TESTE):
    """Regra de 3 de uma etapa que roda imagem a imagem.

    A carga (o que a barra nao cobre: abrir os modelos, gravar a saida) conta uma vez so. O ritmo vem das
    imagens depois da primeira, que costuma ser mais lenta; ela entra pelo tempo que levou.
    """
    if len(pontos) < 2:
        raise ValueError("barra de progresso nao encontrada no log")
    (n1, t1), (n2, t2) = pontos[0], pontos[-1]
    ritmo = (t2 - t1) / (n2 - n1)
    carga = max(parede - t2, 0.0)
    return {"origem": "estimado", "imagens_medidas": n2, "parede_s": round(parede, 1), "carga_s": round(carga, 1),
            "s_por_imagem": round(ritmo, 3), "s": round(carga + t1 + ritmo * (total - n1), 1)}


def roda(cmd, nome, env=None, cwd=None):
    """Roda um comando com o log em results/logs/mede_tempos/<nome>.log; devolve (segundos, texto do log)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{nome}.log"
    print(f"[{datetime.datetime.now():%H:%M:%S}] {nome}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=cwd or paths.SRC,
                           env={**os.environ, **(env or {})})
    parede = time.time() - t0
    if r.returncode:
        raise RuntimeError(f"{nome} falhou ({r.returncode}): veja {log}")
    return parede, log.read_text(encoding="utf-8", errors="replace")


def limpa():
    """Apaga tudo o que os testes criaram (e so isso: o prefixo protege os modelos de verdade)."""
    for pasta in (paths.TRAIN_LOGS, paths.EVALS, ME1_LOGS, ME1_LOGS / "models"):
        for p in pasta.glob(PREFIXO + "*") if pasta.exists() else []:
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
    for p in paths.TABLES.glob(PREFIXO + "*"):
        p.unlink()


def _ckpt(nome, origem):
    """O checkpoint de um modelo existente com um nome de teste (link: ninguem grava nele aqui)."""
    pasta = paths.TRAIN_LOGS / nome
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "last.pth"
    if not destino.exists():
        destino.symlink_to(Path(origem).resolve())


def _copia(fonte, nome, *coisas):
    """Copia (nao liga: as etapas gravam nessas pastas) os tensores de --fonte para um modelo de teste."""
    fonte = Path(fonte)
    pasta = paths.EVALS / nome
    pasta.mkdir(parents=True, exist_ok=True)
    for c in coisas:
        shutil.copyfile(fonte / f"{fonte.name}_{c}.pt", pasta / f"{nome}_{c}.pt")
    return pasta


def _mindeye2(a):
    """Os dois perfis do MindEye2: (hidden_dim, opcoes do recon, opcoes do retrieval, checkpoint)."""
    d = paths.DATA / "train_logs"
    frozen = f"--frozen_ckpt={d}/multisubject_subj01_1024hid_nolow_300ep/last.pth"
    return {
        "mindeye2_1024": (1024, ["--no-blurry_recon", frozen, "--no-backbone_fp16"], ["--no-blurry_recon", frozen],
                          Path(a.fonte.replace("results/evals", "train_logs")) / "last.pth"),
        "mindeye2_4096_blurry": (4096, ["--blurry_recon", "--backbone_fp16"], ["--blurry_recon"],
                                 d / "final_subj01_pretrained_1sess_24bs" / "last.pth"),
    }


def mede_reconstrucao(a, perfil):
    py, dados = sys.executable, f"--data_path={paths.DATA}"
    nome = PREFIXO + perfil
    if perfil in _mindeye2(a):
        hd, opcoes, _, ckpt = _mindeye2(a)[perfil]
        _ckpt(nome, ckpt)
        cmd = [py, "etapas/recon_inference.py", f"--model_name={nome}", dados, f"--cache_dir={paths.DATA}", "--subj=1",
               f"--hidden_dim={hd}", "--n_blocks=4", *opcoes, "--new_test", "--seed=42", f"--max_images={a.imagens}"]
    elif perfil == "frr":                 # o embedding de --fonte tem a forma do FRR, e a velocidade nao depende dele
        _copia(a.fonte, nome, "all_clipvoxels")
        cmd = [py, "etapas/recon_embeddings.py", f"--model_name={nome}", f"--cache_dir={paths.DATA}", "--seed=42",
               f"--max_images={a.imagens}"]
    else:
        return mede_reconstrucao_me1(a, perfil)
    parede, texto = roda(cmd, f"recon_{perfil}")
    return estima(parede, progresso(texto, a.imagens))


def _treino_me1(perfil):
    """Um treino curto so para ter pesos: o tempo por imagem nao depende da qualidade deles."""
    me1 = paths.REPO / "scripts" / "me1_run.sh"
    modelo, baixo = PREFIXO + "me1", PREFIXO + "me1_baixo"
    env = {"NUM_SESSIONS": "1", "MODEL_NAME": modelo, "AE_NAME": baixo}
    if not (ME1_LOGS / modelo / "last.pth").exists():
        roda([me1, "train", "--num_epochs=2"], "me1_treino_curto", env)
    if perfil == "mindeye1":
        pasta = ME1_LOGS / "models" / baixo
        if not (pasta / "epoch120.pth").exists():
            roda([me1, "lowlevel", "--num_epochs=3", "--ckpt_interval=3"], "me1_lowlevel_curto", env)
            # o Reconstructions.py so procura o epoch120.pth; sem ele, desliga o img2img sem reclamar
            (pasta / "epoch003.pth").rename(pasta / "epoch120.pth")
    return env


def mede_reconstrucao_me1(a, perfil):
    env = _treino_me1(perfil)
    extra = ["--autoencoder_name=None", "--img2img_strength=1"] if perfil == "mindeye1_semlowlevel" else []
    parede, texto = roda([paths.REPO / "scripts" / "me1_run.sh", "recon", *extra, f"--max_images={a.imagens}"],
                         f"recon_{perfil}", env)
    if ("Loaded low-level model!" in texto) != (perfil == "mindeye1"):
        raise RuntimeError(f"{perfil}: o low-level nao ficou como o perfil pede (veja o log)")
    return estima(parede, progresso(texto, a.imagens))


def mede_refino(a, perfil):
    nome = PREFIXO + perfil
    parede, texto = roda([sys.executable, "etapas/enhanced_recon_inference.py", f"--model_name={nome}",
                          f"--data_path={paths.DATA}", f"--cache_dir={paths.DATA}", "--subj=1", "--seed=42"],
                         f"refino_{perfil}")
    return estima(parede, progresso(texto, a.imagens))


def mede_retrieval(a, perfil):
    hd, _, opcoes, ckpt = _mindeye2(a)[perfil]
    nome = PREFIXO + "ret_" + perfil
    _ckpt(nome, ckpt)
    parede, _ = roda([sys.executable, "etapas/verify_retrieval.py", f"--model_name={nome}", f"--data_path={paths.DATA}",
                      f"--cache_dir={paths.DATA}", "--subj=1", f"--hidden_dim={hd}", "--n_blocks=4", *opcoes, "--new_test"],
                     f"retrieval_{perfil}")
    return {"origem": "medido", "s": round(parede, 1)}


def mede_retrieval_frr(a):
    pasta = paths.EVALS / (PREFIXO + "ret_frr")
    pasta.mkdir(parents=True, exist_ok=True)
    fonte = Path(a.fonte)
    parede, texto = roda([sys.executable, "etapas/mede_tempos.py", "--avalia_frr", fonte / f"{fonte.name}_all_clipvoxels.pt",
                          pasta], "retrieval_frr")
    return {"origem": "medido", "s": float(re.search(r"SEGUNDOS=([\d.]+)", texto)[1])}


def avalia_frr(clipvoxels, saida):
    """A avaliacao do run_frr.py (retrieval, centrado e similaridades) cronometrada, sem refazer o ajuste."""
    import torch
    from run_frr import avalia, imagens_do_teste
    from mindeye_ridge import nsd_data
    ids_teste, _ = nsd_data.exibicoes_teste(paths.DATA, 1)
    all_images = imagens_do_teste(paths.DATA, ids_teste)
    prev = torch.load(clipvoxels, map_location="cpu").flatten(1).to(torch.float16)
    t0 = time.time()
    avalia(PREFIXO + "ret_frr", prev, all_images, prev.float().mean(0), 42, "cuda", str(saida))
    print(f"SEGUNDOS={time.time() - t0:.1f}", flush=True)


def mede_metricas(a, chave):
    familia, tipo = chave.split("/")
    nome = PREFIXO + "metricas_" + familia
    tensor = "all_recons" if tipo == "unclip" else "all_enhancedrecons"
    if familia == "mindeye1":             # como o Reconstructions.py grava: sem legendas, retrieval no ViT-L
        import torch
        pasta = _copia(a.fonte, nome, tensor)
        for c in ("all_clipvoxels", "all_clipimages"):    # o tempo do retrieval nao depende dos valores
            torch.save(torch.randn(IMAGENS_TESTE, 257, 768), pasta / f"{nome}_{c}.pt")
    else:
        pasta = _copia(a.fonte, nome, tensor, "all_clipvoxels", "all_predcaptions")
    parede, _ = roda([sys.executable, "-u", "etapas/final_evaluations.py", f"--model_name={nome}",
                      f"--all_recons_path={pasta}/{nome}_{tensor}.pt", f"--data_path={paths.DATA}",
                      f"--cache_dir={paths.DATA}", "--subj=1"], f"metricas_{familia}_{tipo}")
    return {"origem": "medido", "s": round(parede, 1)}


def plano(a):
    """(etapa, perfil, funcao), na ordem: o refino le as reconstrucoes do teste do mesmo perfil."""
    recon = ["mindeye2_1024", "mindeye2_4096_blurry", "frr", "mindeye1", "mindeye1_semlowlevel"]
    return ([("reconstrucao", p, lambda p=p: mede_reconstrucao(a, p)) for p in recon]
            + [("refino", p, lambda p=p: mede_refino(a, p)) for p in recon[:3]]
            + [("retrieval", p, lambda p=p: mede_retrieval(a, p)) for p in recon[:2]]
            + [("retrieval", "frr", lambda: mede_retrieval_frr(a))]
            + [("metricas", m, lambda m=m: mede_metricas(a, m))
               for m in ("bigg/unclip", "bigg/refinada", "mindeye1/unclip")])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonte", help="results/evals/<modelo> com as 1.000 reconstrucoes, refino e legendas de um MindEye2 "
                                   "1024 (o checkpoint vem de train_logs/<modelo>)")
    p.add_argument("--imagens", type=int, default=30, help="imagens de cada teste de reconstrucao e refino")
    p.add_argument("--refaz", action="store_true", help="ignora o que ja esta medido em results/tables/tempos_etapas.json")
    p.add_argument("--avalia_frr", nargs=2, metavar=("CLIPVOXELS", "SAIDA"), help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.avalia_frr:
        return avalia_frr(*a.avalia_frr)
    if not a.fonte:
        p.error("falta --fonte")
    a.fonte = str(Path(a.fonte).resolve())

    medidas = {} if a.refaz or not SAIDA.exists() else json.loads(SAIDA.read_text(encoding="utf-8"))["etapas"]
    limpa()
    # o refino le as reconstrucoes do teste: sem elas, refaz tambem a reconstrucao daquele perfil
    for perfil in list(medidas.get("reconstrucao", {})):
        if perfil in ("mindeye2_1024", "mindeye2_4096_blurry", "frr") and perfil not in medidas.get("refino", {}):
            del medidas["reconstrucao"][perfil]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    for etapa, perfil, mede in plano(a):
        if perfil in medidas.get(etapa, {}):
            continue
        medidas.setdefault(etapa, {})[perfil] = mede()
        SAIDA.write_text(json.dumps({
            "data": f"{datetime.date.today()}", "gpu": gpu.stdout.strip(), "imagens_total": IMAGENS_TESTE,
            "imagens_teste": a.imagens, "fonte": Path(a.fonte).name, "etapas": medidas},
            indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"    {json.dumps(medidas[etapa][perfil])}", flush=True)
    limpa()


if __name__ == "__main__":
    main()
