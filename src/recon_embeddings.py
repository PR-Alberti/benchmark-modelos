#!/usr/bin/env python3
"""Reconstroi as imagens do teste a partir de embeddings CLIP bigG ja previstos (o FRR).

O recon_inference.py passa os voxels pela rede do MindEye2 (ridge, backbone, prior) e so depois
pelo unCLIP. Aqui o embedding ja vem pronto, de results/evals/<modelo>/<modelo>_all_clipvoxels.pt
((1000, 256, 1664), na ordem crescente dos ids do teste, a do all_images.pt), e vai direto para o
unCLIP e, convertido para o ViT-L, para o GIT, que da a legenda usada no refinamento. Nada vem de um
checkpoint do MindEye2: as tres redes (mindeye_ridge/decodificador.py) foram treinadas so com imagens
e texto. Sem ramo low-level, nao ha reconstrucao borrada.

    python recon_embeddings.py --model_name repro_subj01_frr_1sess
    python recon_embeddings.py --model_name repro_subj01_frr_1sess --max_images 8    # teste rapido

Grava <modelo>_all_recons.pt (n, 3, 256, 256) e <modelo>_all_predcaptions.pt, os mesmos arquivos do
recon_inference.py, entao o enhanced_recon_inference.py e o final_evaluations.py seguem iguais. A
semente e a mesma do resto (42); uma corrida interrompida retoma do parcial com o estado dos
geradores, como no recon_inference.py.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from mindeye_ridge import paths
paths.add_vendored_to_path()                     # o utils importa o sgm (unclip_recon)
from mindeye_ridge import decodificador, utils

torch.backends.cuda.matmul.allow_tf32 = True     # como no recon_inference.py


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True)
    p.add_argument("--cache_dir", default=str(paths.DATA), help="onde estao o unCLIP e o bigG_to_L")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=None,
                   help="so as N primeiras imagens do teste (teste rapido); a saida fica com N imagens")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pasta = f"{paths.EVALS}/{a.model_name}"
    entrada = f"{pasta}/{a.model_name}_all_clipvoxels.pt"
    if not os.path.exists(entrada):
        sys.exit(f"{entrada} nao existe: rode o modelo (run_frr.py) antes")
    emb = torch.load(entrada, map_location="cpu")
    if emb.ndim != 3 or tuple(emb.shape[1:]) != (decodificador.CLIP_SEQ, decodificador.CLIP_DIM):
        sys.exit(f"{entrada}: forma {tuple(emb.shape)}, esperada (n, {decodificador.CLIP_SEQ}, {decodificador.CLIP_DIM})")
    n = len(emb) if a.max_images is None else min(a.max_images, len(emb))
    print(f"{a.model_name}: {n} embeddings de {entrada}", flush=True)

    utils.seed_everything(a.seed)
    processor, clip_text_model = decodificador.carrega_git(device)
    clip_convert = decodificador.carrega_conversor(a.cache_dir, device)
    diffusion_engine, vector_suffix = decodificador.carrega_unclip(a.cache_dir, device)

    # parcial a cada 100 imagens, com o estado dos geradores: a retomada segue a mesma sequencia de ruido
    parcial = f"{pasta}/{a.model_name}_recon_partial.pt"
    recons, legendas = [], []
    if os.path.exists(parcial):
        r = torch.load(parcial)
        recons, legendas = r["recons"], r["predcaptions"]
        torch.set_rng_state(r["rng_torch"]); torch.cuda.set_rng_state(r["rng_cuda"])
        np.random.set_state(r["rng_np"]); random.setstate(r["rng_py"])
        print(f"retomando de {len(recons)} imagens ja reconstruidas", flush=True)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for i in tqdm(range(len(recons), n)):
            x = emb[[i]].to(device).float()                                  # (1, 256, 1664)
            ids = clip_text_model.generate(pixel_values=clip_convert(x), max_length=20)
            legendas += processor.batch_decode(ids, skip_special_tokens=True)
            amostra = utils.unclip_recon(x, diffusion_engine, vector_suffix, num_samples=1)
            recons.append(transforms.Resize((256, 256))(amostra).float().cpu())
            if len(recons) % 100 == 0:
                torch.save({"recons": recons, "predcaptions": legendas,
                            "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                            "rng_np": np.random.get_state(), "rng_py": random.getstate()}, parcial + ".tmp")
                os.replace(parcial + ".tmp", parcial)

    todas = torch.cat(recons)
    print("recons", tuple(todas.shape), "| primeiras legendas:", legendas[:3], flush=True)
    torch.save(todas, f"{pasta}/{a.model_name}_all_recons.pt")
    torch.save(np.array(legendas), f"{pasta}/{a.model_name}_all_predcaptions.pt")
    if os.path.exists(parcial):
        os.remove(parcial)
    print(f"gravado: {pasta}/{a.model_name}_all_recons.pt e _all_predcaptions.pt", flush=True)


if __name__ == "__main__":
    main()
