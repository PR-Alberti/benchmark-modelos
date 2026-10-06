"""Os dados do benchmark no formato que os scripts do MindEye1 esperam.

Para a comparacao ser justa, o MindEye1 treina e e testado nas mesmas exibicoes que os outros
modelos do benchmark (MindEye2 ridge-only e FRR): as N primeiras sessoes do subj01 no treino e
as 1.000 imagens do teste novo, com os betas do MindEye2 (betas_all_subj01_fp32_renorm.hdf5) e as
imagens 224 x 224 do coco_images_224_float16.hdf5. Quem le e o mindeye_ridge.nsd_data, o mesmo
codigo do FRR.

O MindEye1 original le o webdataset_avg_split: uma amostra por imagem, com as 3 repeticoes
empilhadas, e o treino usa a repeticao train_i % 3. Aqui a amostra tambem e a imagem; quando ela
tem menos de 3 exibicoes nas sessoes usadas (com 1 sessao, 413 das 536 tem so uma), as que existem
se repetem ate completar 3, como o recon_inference.py do MindEye2 faz no teste.

Com `dataset` (manifesto do mindeye_ridge.dataset_controlado), o treino sao as exibicoes do
manifesto em vez das N primeiras sessoes; o teste nao muda. A agregacao do manifesto
(mindeye_ridge.agregacao) e aplicada pelo Lotes do treino: com "exibicoes" cada exibicao e uma
amostra; com "media", "sorteio" e "combinacao" a amostra e a imagem, e as 3 posicoes de repeticao
chegam ao treino iguais ao vetor escolhido, entao o rodizio do alto nivel (train_i % 3) e o
voxel_select do baixo nivel usam exatamente esse vetor. Sem agregacao, o original.
"""
import os
import sys

import h5py
import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
from mindeye_ridge import agregacao, dataset_controlado, nsd_data  # noqa: E402

SUBJ = 1


def _completa3(linhas):
    """Indices de 3 exibicoes: as que existem, repetidas na ordem ate completar 3."""
    linhas = np.asarray(linhas)
    return np.resize(linhas, 3)


def _imagens(data_path, ids):
    """Imagens 224 x 224 em [0, 1], fp16, na ordem de `ids`."""
    ordem = np.argsort(ids)
    with h5py.File(f"{data_path}/coco_images_224_float16.hdf5", "r") as f:
        lidas = f["images"][np.asarray(ids)[ordem]]   # o h5py exige indices crescentes
    saida = np.empty_like(lidas)
    saida[ordem] = lidas
    return torch.from_numpy(saida)


def carrega(data_path, num_sessions, dataset=None):
    """Treino (N sessoes, ou o manifesto `dataset`) e teste (1.000 imagens), cada um com voxel (n, 3, 15.724) fp32,
    imagem (n, 3, 224, 224) fp16 e o id COCO-73k (n,).

    O teste vem em ordem crescente de id, a do all_images.pt e a das reconstrucoes do benchmark.
    """
    betas = nsd_data.carrega_betas(data_path, SUBJ)

    ex = dataset_controlado.exibicoes(data_path, SUBJ, num_sessions, dataset)
    ids_treino = np.unique(ex["imagem"])
    if ex["agregacao"] == "exibicoes":
        # uma amostra por exibicao; cada imagem e guardada uma vez so e indexada por amostra
        treino = {"voxel": torch.from_numpy(betas[ex["beta"]])[:, None].expand(-1, 3, -1),
                  "imagem": _imagens(data_path, ids_treino),
                  "indice_imagem": torch.from_numpy(np.searchsorted(ids_treino, ex["imagem"])),
                  "coco": torch.from_numpy(ex["imagem"]),
                  "n_rep": torch.ones(len(ex["beta"]), dtype=torch.long)}
    else:
        grupos = [ex["beta"][ex["imagem"] == i] for i in ids_treino]
        treino = {"voxel": torch.from_numpy(betas[np.stack([_completa3(g) for g in grupos])]),
                  "imagem": _imagens(data_path, ids_treino),
                  "coco": torch.from_numpy(ids_treino),
                  "n_rep": torch.tensor([len(g) for g in grupos])}
    treino["agregacao"] = ex["agregacao"]

    return treino, carrega_teste(data_path, betas)


def carrega_teste(data_path, betas=None):
    """So o teste: as 1.000 imagens do teste novo, em ordem crescente de id."""
    if betas is None:
        betas = nsd_data.carrega_betas(data_path, SUBJ)
    ids_teste, linhas_teste = nsd_data.exibicoes_teste(data_path, SUBJ)
    return {"voxel": torch.from_numpy(betas[np.stack([_completa3(l) for l in linhas_teste])]),
            "imagem": _imagens(data_path, ids_teste),
            "coco": torch.from_numpy(ids_teste)}


class Lotes:
    """Itera em lotes (voxel, imagem, coco), como os DataLoaders do MindEye1.

    No treino (embaralha=True), cada passagem e uma permutacao nova e tem n // batch lotes
    completos, o que o webdataset do original faz com resampled + with_epoch. No teste, ordem
    fixa e so lotes completos (partial=False no original: 1.000 -> 3 lotes de 300).
    """

    def __init__(self, dados, batch, embaralha, device, seed=0):
        self.dados, self.batch, self.embaralha, self.device = dados, batch, embaralha, device
        self.rng = np.random.default_rng(seed)
        self.gerador = torch.Generator().manual_seed(seed)      # sorteios da agregacao
        self.n = len(dados["coco"])

    def __len__(self):
        return self.n // self.batch

    def __iter__(self):
        ordem = self.rng.permutation(self.n) if self.embaralha else np.arange(self.n)
        for k in range(len(self)):
            idx = torch.from_numpy(np.sort(ordem[k * self.batch:(k + 1) * self.batch]) if not self.embaralha
                                   else ordem[k * self.batch:(k + 1) * self.batch])
            voxel = self.dados["voxel"][idx]
            modo = self.dados.get("agregacao")
            if self.embaralha and modo is not None:
                voxel = agregacao.seleciona(voxel, self.dados["n_rep"][idx], modo, self.gerador)[:, None].expand(-1, 3, -1)
            img = self.dados["indice_imagem"][idx] if "indice_imagem" in self.dados else idx
            yield (voxel.to(self.device),
                   self.dados["imagem"][img].to(self.device).float(),
                   self.dados["coco"][idx].to(self.device))
