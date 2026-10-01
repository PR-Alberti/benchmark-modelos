#!/usr/bin/env python3
"""Monta uma figura comparando imagem vista x reconstrucao.

Por padrao poe tres colunas -- a imagem que o participante viu, a reconstrucao
deste repositorio e a reconstrucao publicada no artigo -- para alguns exemplos
sorteados. Se as reconstrucoes do artigo nao estiverem baixadas, sai com duas.

    python src/make_comparison.py --model subj01_ridgeonly_1sess_prior
    python src/make_comparison.py --model X --n 18 --seed 7 --out results/figs/comparacao.png

As reconstrucoes e o all_images.pt sao indexados igual (as metricas do
final_evaluations.py usam essa mesma correspondencia), entao a linha i de cada
tensor e o mesmo estimulo.
"""
import argparse
import os
import pathlib

from mindeye_ridge import paths

REF_ARTIGO = "final_subj01_pretrained_1sess_24bs"


def carrega(caminho):
    import torch
    t = torch.load(caminho, map_location="cpu")
    return t.float().clamp(0, 1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="nome do modelo em results/evals/")
    p.add_argument("--kind", default="enhanced", choices=["base", "enhanced"],
                   help="reconstrucao bruta ou refinada (padrao: enhanced)")
    p.add_argument("--n", type=int, default=12, help="quantos exemplos (padrao: 12)")
    p.add_argument("--grupos", type=int, default=2,
                   help="blocos de 3 colunas lado a lado (padrao: 2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-path", default=str(paths.DATA))
    p.add_argument("--sem-artigo", action="store_true",
                   help="nao inclui a coluna do modelo publicado")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    sufixo = "all_enhancedrecons" if args.kind == "enhanced" else "all_recons"
    nosso_p = pathlib.Path(f"{paths.EVALS}/{args.model}/{args.model}_{sufixo}.pt")
    if not nosso_p.exists():
        raise SystemExit(f"nao achei {nosso_p}\n"
                         f"rode antes: scripts/run_recon.sh {args.model} 1024 noblurry")

    imagens = carrega(f"{args.data_path}/evals/all_images.pt")
    nosso = carrega(nosso_p)

    colunas = [("Imagem vista", imagens), ("Ridge-only", nosso)]
    artigo_p = pathlib.Path(
        f"{args.data_path}/evals/{REF_ARTIGO}/{REF_ARTIGO}_all_enhancedrecons.pt")
    if not args.sem_artigo and artigo_p.exists():
        colunas.append(("Artigo", carrega(artigo_p)))

    # As colunas podem estar salvas em resolucoes diferentes (as nossas em 256,
    # as publicadas em 512). Exibir cada uma na sua faz a de menor resolucao
    # parecer pior so pela suavizacao -- entao igualamos todas pela menor.
    import torch.nn.functional as F
    menor = min(t.shape[-1] for _, t in colunas)
    colunas = [(nome, t if t.shape[-1] == menor
                else F.interpolate(t, size=(menor, menor), mode="bilinear",
                                   antialias=True).clamp(0, 1))  # antialias extrapola um pouco
               for nome, t in colunas]
    print(f"exibindo todas as colunas em {menor}x{menor}")

    n_disp = min(len(t) for _, t in colunas)
    if n_disp != len(imagens):
        print(f"aviso: tensores de tamanhos diferentes, usando os {n_disp} primeiros")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_disp, size=min(args.n, n_disp), replace=False)

    g = max(1, args.grupos)
    linhas = int(np.ceil(len(idx) / g))
    ncol = len(colunas) * g
    fig, ax = plt.subplots(linhas, ncol, figsize=(1.7 * ncol, 1.7 * linhas + 0.5))
    ax = np.atleast_2d(ax)

    for a in ax.ravel():
        a.axis("off")
    for k, i in enumerate(idx):
        bloco, linha = k // linhas, k % linhas
        for c, (_, t) in enumerate(colunas):
            a = ax[linha][bloco * len(colunas) + c]
            a.imshow(t[i].permute(1, 2, 0).numpy())
            a.axis("off")
    for bloco in range(g):
        for c, (nome, _) in enumerate(colunas):
            ax[0][bloco * len(colunas) + c].set_title(nome, fontsize=9, pad=4)

    fig.suptitle(f"{args.model} — reconstrucoes {args.kind}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    saida = args.out or f"{paths.FIGS}/{args.model}_{args.kind}_comparacao.png"
    pathlib.Path(saida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=140, bbox_inches="tight")
    print(f"gravado: {saida}  ({os.path.getsize(saida)/1e6:.1f} MB)")
    print(f"  {len(idx)} exemplos, colunas: {', '.join(n for n, _ in colunas)}")
    print(f"  indices: {sorted(idx.tolist())}")


if __name__ == "__main__":
    main()
