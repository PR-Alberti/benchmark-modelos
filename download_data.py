#!/usr/bin/env python3
"""Baixa os arquivos do MindEye2 (dataset pscotti/mindeyev2 no HuggingFace).

Os arquivos sao agrupados por etapa do pipeline, para nao baixar 100+ GB quando
so se quer treinar. Exemplos:

    # so o necessario para treinar do zero (wds + betas + imagens COCO)
    ./download_data.py --stage train --subj 1

    # fine-tune a partir do checkpoint multi-sujeito (protocolo do paper)
    ./download_data.py --stage finetune --subj 1

    # gerar reconstrucoes de imagem depois de treinar
    ./download_data.py --stage recon

    # so listar o que falta, sem baixar nada
    ./download_data.py --stage finetune --subj 1 --dry-run

Aceita varias etapas de uma vez: --stage train recon evals
"""
import argparse
import os
import sys

REPO = "pscotti/mindeyev2"

# Numero de sessoes de fMRI por sujeito no NSD (define quantos .tar existem).
NSESSIONS = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}

# Checkpoint multi-sujeito usado como ponto de partida no fine-tune de 1 sessao.
MULTISUBJECT_CKPT = "train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth"

# Arquivos que nao dependem do sujeito, por etapa.
STAGE_FILES = {
    "train": [
        "coco_images_224_float16.hdf5",          # 22 GB - as 73k imagens do NSD
    ],
    "finetune": [
        "coco_images_224_float16.hdf5",
        MULTISUBJECT_CKPT,                        # 2.9 GB - pre-treino nos outros 7 sujeitos
    ],
    "blurry": [                                   # apenas se usar --blurry_recon
        "sd_image_var_autoenc.pth",               # 0.3 GB
        "convnext_xlarge_alpha0.75_fullckpt.pth", # 6.6 GB
        # O ramo low-level so existe no ckpt 4096; o multisubject 1024 e "nolow"
        # (zero chaves de low-level), entao com ele --blurry_recon criaria a
        # cabeca com pesos aleatorios -- e sob ridge_only ela ficaria congelada
        # em ruido. Este e o ponto de partida da comparacao pareada com o paper.
        "train_logs/final_multisubject_subj01/last.pth",  # 10.3 GB
    ],
    "recon": [                                    # recon_inference.ipynb
        "unclip6_epoch0_step110000.ckpt",         # 18 GB
        "bigG_to_L_epoch8.pth",
        "sd_image_var_autoenc.pth",
        "evals/all_images.pt",                    # 0.6 GB
    ],
    "enhanced": [                                 # enhanced_recon_inference.ipynb
        "zavychromaxl_v30.safetensors",           # 6.9 GB
    ],
    "paper": [                                    # modelo publicado do paper, subj01 1 sessao
        # ATENCAO: treinado com hidden_dim=4096 e --blurry_recon, partindo de
        # final_multisubject_subj01. Difere da base 1024/nolow em mais de uma
        # variavel -- veja o README do repo original, secao dos ckpts.
        "train_logs/final_subj01_pretrained_1sess_24bs/last.pth",   # 8.9 GB
        "evals/final_subj01_pretrained_1sess_24bs/final_subj01_pretrained_1sess_24bs_all_enhancedrecons.pt",
    ],
    "paper40": [                                  # modelo publicado do paper, subj01 40 sessoes
        # referencia para o ridge-only de 40 sessoes; mesma arquitetura do
        # "paper" (4096 + blurry), fine-tune completo com as 40 sessoes
        "train_logs/final_subj01_pretrained_40sess_24bs/last.pth",
        "evals/final_subj01_pretrained_40sess_24bs/final_subj01_pretrained_40sess_24bs_all_enhancedrecons.pt",
        "evals/final_subj01_pretrained_40sess_24bs/final_subj01_pretrained_40sess_24bs_all_recons.pt",
    ],
    "evals": [                                    # final_evaluations.ipynb
        "evals/all_images.pt",
        "evals/all_captions.pt",
        "evals/all_git_generated_captions.pt",
        "gnet_multisubject.pt",                   # 1.3 GB
        "brain_region_masks.hdf5",
    ],
}

# Etapas que precisam dos dados de fMRI do sujeito (betas + webdataset).
STAGES_NEED_SUBJ = {"train", "finetune", "recon", "evals", "paper", "paper40"}


def subject_files(subj, num_sessions=None):
    """Arquivos especificos de um sujeito: betas + tars do webdataset."""
    files = [f"betas_all_subj0{subj}_fp32_renorm.hdf5"]
    n = NSESSIONS[subj] if num_sessions is None else min(num_sessions, NSESSIONS[subj])
    files += [f"wds/subj0{subj}/train/{i}.tar" for i in range(n)]
    files += [f"wds/subj0{subj}/test/0.tar", f"wds/subj0{subj}/new_test/0.tar"]
    return files


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", nargs="+", default=["finetune"],
                   choices=sorted(STAGE_FILES), metavar="ETAPA",
                   help="etapas a preparar: " + ", ".join(sorted(STAGE_FILES)))
    p.add_argument("--subj", type=int, nargs="+", default=[1], choices=range(1, 9),
                   metavar="N", help="sujeitos do NSD (1-8)")
    p.add_argument("--num-sessions", type=int, default=None,
                   help="baixa so os N primeiros .tar de treino (padrao: todos)")
    p.add_argument("--data-path", default=os.path.expanduser("~/mindeyev2"),
                   help="destino dos arquivos (padrao: ~/mindeyev2)")
    p.add_argument("--dry-run", action="store_true",
                   help="lista o que seria baixado e o tamanho total, sem baixar")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="token do HuggingFace (opcional; da mais banda)")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub nao instalado. Rode ./setup_env.sh primeiro,\n"
                 "ou instale so ele com: pip install huggingface_hub")

    # monta a lista de arquivos, sem duplicatas, preservando a ordem
    wanted, seen = [], set()
    for stage in args.stage:
        for f in STAGE_FILES[stage]:
            if f not in seen:
                seen.add(f); wanted.append(f)
        if stage in STAGES_NEED_SUBJ:
            for s in args.subj:
                for f in subject_files(s, args.num_sessions):
                    if f not in seen:
                        seen.add(f); wanted.append(f)

    api = HfApi(token=args.token)
    sizes = {x.rfilename: (x.size or 0)
             for x in api.repo_info(REPO, repo_type="dataset", files_metadata=True).siblings}

    missing, total, have = [], 0, 0
    for f in wanted:
        local = os.path.join(args.data_path, f)
        size = sizes.get(f, 0)
        if f not in sizes:
            print(f"  ?  {f}  (nao existe no repo {REPO})")
            continue
        if os.path.exists(local) and os.path.getsize(local) == size:
            have += size
        else:
            missing.append(f); total += size

    print(f"\netapas: {', '.join(args.stage)} | sujeitos: {args.subj}")
    print(f"destino: {args.data_path}")
    print(f"ja presente: {human(have)} | a baixar: {human(total)} em {len(missing)} arquivo(s)\n")

    if not missing:
        print("Tudo ja esta no lugar. Nada a fazer.")
        return

    # nao lista os ~40 tars um a um; eles sao pequenos e repetitivos
    tars = [f for f in missing if f.endswith(".tar")]
    for f in missing:
        if not f.endswith(".tar"):
            print(f"  {human(sizes[f]):>10}  {f}")
    if tars:
        print(f"  {human(sum(sizes[f] for f in tars)):>10}  {len(tars)} tars do webdataset")

    if args.dry_run:
        print("\n(--dry-run: nada foi baixado)")
        return

    print()
    os.makedirs(args.data_path, exist_ok=True)
    for i, f in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {f}", flush=True)
        hf_hub_download(repo_id=REPO, filename=f, repo_type="dataset",
                        local_dir=args.data_path, token=args.token)

    print(f"\nPronto. Use --data_path={args.data_path} nos scripts de treino.")


if __name__ == "__main__":
    main()
