#!/usr/bin/env python3
"""Baixa os arquivos do MindEye1 (dataset pscotti/naturalscenesdataset no HuggingFace).

Por etapa, como o download_data.py do MindEye2, para nao baixar tudo sem precisar:

    mindeye1/download.py --stage test ckpts vd   # inferencia com os modelos publicados (~31 GB)
    mindeye1/download.py --stage train           # + dados de treino (~39 GB)
    mindeye1/download.py --stage test --dry-run  # so lista o que falta

Etapas (so subj01, o sujeito do benchmark):
    test   shards de teste + metadados               4,3 GB
    train  shards de treino e validacao             38,7 GB
    ckpts  modelos publicados: alto nivel (12 GB, com otimizador) e low-level (2,5 GB)
    vd     Versatile Diffusion, so as partes que o diffusers carrega (12 GB, vai para o HF_HOME)

Os dados vao para --data_path (padrao: $ME1_DATA ou ~/mindeye1). O HF_HOME do
MindEye2 ($MINDEYE_DATA/.cache) e reaproveitado para o Versatile Diffusion.
"""
import argparse
import os
import sys

REPO = "pscotti/naturalscenesdataset"
VD_REPO = "shi-labs/versatile-diffusion"
SUBJ = "01"

STAGES = {
    "test": [f"webdataset_avg_split/test/test_subj{SUBJ}_*",
             f"webdataset_avg_split/metadata_subj{SUBJ}.json"],
    "train": [f"webdataset_avg_split/train/train_subj{SUBJ}_*",
              f"webdataset_avg_split/val/val_subj{SUBJ}_*",
              f"webdataset_avg_split/metadata_subj{SUBJ}.json"],
    # o release tem tambem replication.pth (12 GB), que a reconstrucao nao usa
    "ckpts": [f"mindeye_models/prior_257_final_subj{SUBJ}_bimixco_softclip_byol/last.pth",
              f"mindeye_models/autoencoder_subj{SUBJ}_4x_locont_no_reconst/*"],
}
# pretrained_pth/ tem os pesos no formato do repo original da Shi Labs (36 GB); o diffusers
# nao os le
VD_PATTERNS = ["*.json", "*.txt", "image_encoder/*", "image_unet/*", "text_encoder/*",
               "text_unet/*", "vae/*", "scheduler/*", "tokenizer/*", "image_feature_extractor/*"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True, choices=[*STAGES, "vd"])
    ap.add_argument("--data_path", default=os.environ.get("ME1_DATA", os.path.expanduser("~/mindeye1")))
    ap.add_argument("--dry-run", action="store_true", help="so lista o que seria baixado")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(
        os.environ.get("MINDEYE_DATA", os.path.expanduser("~/mindeyev2")), ".cache"))
    from huggingface_hub import HfApi, snapshot_download
    from fnmatch import fnmatch

    api = HfApi()
    padroes = sorted({p for s in args.stage if s != "vd" for p in STAGES[s]})
    if padroes:
        info = api.dataset_info(REPO, files_metadata=True)
        arqs = [s for s in info.siblings if any(fnmatch(s.rfilename, p) for p in padroes)]
        faltam = [s for s in arqs if not os.path.exists(os.path.join(args.data_path, s.rfilename))]
        print(f"{REPO}: {len(arqs)} arquivos, {sum(s.size for s in arqs) / 1e9:.1f} GB; "
              f"faltam {len(faltam)} ({sum(s.size for s in faltam) / 1e9:.1f} GB) em {args.data_path}")
        for s in faltam:
            print(f"  {s.size / 1e9:6.2f} GB  {s.rfilename}")
        if faltam and not args.dry_run:
            snapshot_download(REPO, repo_type="dataset", local_dir=args.data_path, allow_patterns=padroes)

    if "vd" in args.stage:
        if args.dry_run:
            print(f"{VD_REPO}: seria baixado para {os.environ['HF_HOME']} (~12 GB)")
        else:
            p = snapshot_download(VD_REPO, allow_patterns=VD_PATTERNS)
            print(f"Versatile Diffusion em {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
