#!/usr/bin/env python3
"""Baixa os pesos que os tres modelos buscam sozinhos na primeira execucao.

O download_data.py traz os dados e os checkpoints do MindEye2. Alem deles, o treino, a
reconstrucao e as metricas baixam pesos de terceiros na hora em que precisam: o CLIP bigG
(alvo do MindEye2 e do FRR), o Versatile Diffusion (MindEye1), as redes das metricas etc.
Este script baixa todos de uma vez, para a maquina nova ficar pronta antes do primeiro treino
e nenhum job parar horas depois por falta de rede.

    setup/prefetch_weights.py            baixa o que falta
    setup/prefetch_weights.py --check    so lista o que falta

Os caches sao os mesmos que os scripts usam (scripts/common.sh): HF_HOME=$MINDEYE_DATA/.cache
para o HuggingFace, ~/.cache/torch para o torch.hub, ~/.cache/clip para o CLIP da OpenAI e
~/nltk_data para o nltk. Rode com o python do ambiente (~/envs/fmri/bin/python).
"""
import argparse
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.environ.get("MINDEYE_DATA", os.path.expanduser("~/mindeyev2"))
os.environ["HF_HOME"] = os.path.join(DATA, ".cache")
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

# (repo do HuggingFace, quem usa)
HF_MODELOS = [
    ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "MindEye2 e FRR: alvo CLIP (9,5 GB)"),
    ("openai/clip-vit-large-patch14", "MindEye1 (alvo) e metricas de legenda (1,6 GB)"),
    ("openai/clip-vit-base-patch32", "metricas de legenda"),
    ("microsoft/git-large-coco", "MindEye2: legendas previstas (1,5 GB)"),
    ("sentence-transformers/all-MiniLM-L6-v2", "metricas de legenda"),
]
TORCH_HUB = os.path.join(os.path.expanduser(os.environ.get("TORCH_HOME", "~/.cache/torch")), "hub")
NLTK = ["punkt", "wordnet", "omw-1.4"]


def hf_tem(repo):
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def itens():
    """(nome, quem usa, ja_tem(), baixa())"""
    import importlib.util
    lista = []
    for repo, uso in HF_MODELOS:
        def baixa(repo=repo):
            from huggingface_hub import snapshot_download
            snapshot_download(repo)
        lista.append((repo, uso, lambda repo=repo: hf_tem(repo), baixa))

    def vd_tem():
        from huggingface_hub import snapshot_download
        try:
            p = snapshot_download("shi-labs/versatile-diffusion", local_files_only=True)
            return os.path.isdir(os.path.join(p, "image_unet"))
        except Exception:
            return False

    def vd_baixa():
        subprocess.run([sys.executable, os.path.join(REPO, "mindeye1", "download.py"), "--stage", "vd"],
                       check=True)
    lista.append(("shi-labs/versatile-diffusion", "MindEye1: gera as imagens (~23 GB)", vd_tem, vd_baixa))

    from torchvision.models import (AlexNet_Weights, EfficientNet_B1_Weights, Inception_V3_Weights)
    for w in (AlexNet_Weights.IMAGENET1K_V1, Inception_V3_Weights.DEFAULT, EfficientNet_B1_Weights.DEFAULT):
        arq = os.path.join(TORCH_HUB, "checkpoints", os.path.basename(w.url))
        lista.append((os.path.basename(w.url), "metricas (torchvision)", lambda a=arq: os.path.exists(a),
                      lambda w=w: w.get_state_dict(progress=True)))

    def swav_baixa():
        import torch
        torch.hub.load("facebookresearch/swav:main", "resnet50")
    swav = os.path.join(TORCH_HUB, "checkpoints", "swav_800ep_pretrain.pth.tar")
    lista.append(("swav resnet50", "metricas", lambda: os.path.exists(swav), swav_baixa))

    def clip_baixa():
        import clip
        clip.load("ViT-L/14", device="cpu")
    clip_arq = os.path.expanduser("~/.cache/clip/ViT-L-14.pt")
    lista.append(("CLIP ViT-L/14 (OpenAI)", "MindEye1 e metricas", lambda: os.path.exists(clip_arq), clip_baixa))

    def nltk_tem():
        import nltk
        for r in ("tokenizers/punkt", "corpora/wordnet", "corpora/omw-1.4"):
            try:
                nltk.data.find(r)
            except LookupError:
                try:
                    nltk.data.find(r + ".zip")
                except LookupError:
                    return False
        return True

    def nltk_baixa():
        import nltk
        for r in NLTK:
            nltk.download(r, quiet=True)
    lista.append(("nltk: " + ", ".join(NLTK), "metricas de legenda (METEOR)", nltk_tem, nltk_baixa))

    modulos = os.path.join(os.environ["HF_HOME"], "modules", "evaluate_modules", "metrics")

    def eval_baixa():
        import evaluate
        evaluate.load("meteor")
        evaluate.load("rouge")
    lista.append(("evaluate: meteor, rouge", "metricas de legenda",
                  lambda: all(os.path.isdir(os.path.join(modulos, f"evaluate-metric--{m}"))
                              for m in ("meteor", "rouge")), eval_baixa))
    return lista


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="so lista o que falta")
    args = ap.parse_args()

    faltou = 0
    for nome, uso, tem, baixa in itens():
        if tem():
            print(f"  ok     {nome}")
            continue
        if args.check:
            print(f"  FALTA  {nome}  ({uso})")
            faltou += 1
            continue
        print(f"  baixando {nome}  ({uso}) ...", flush=True)
        try:
            baixa()
            print(f"  ok     {nome}")
        except Exception as e:
            print(f"  ERRO   {nome}: {e}")
            faltou += 1
    return 1 if faltou and not args.check else 0


if __name__ == "__main__":
    sys.exit(main())
