"""Exibicoes do NSD (tars do webdataset) e betas, lidos direto, sem o DataLoader do treino.

Cada tar de treino e uma sessao; o tar do teste novo (new_test) tem as 1.000
imagens compartilhadas, 3 repeticoes cada. Em cada exibicao, `behav` guarda o
indice da imagem (coluna 0, de 0 a 72.999) e a linha dela em `betas` (coluna 5).
O mesmo recorte que o train_ridgeonly.py enxerga, so que em arrays.
"""
import io
import tarfile

import h5py
import numpy as np

COL_IMAGEM, COL_BETA = 0, 5


def le_behav(tar):
    """behav de cada exibicao de um tar, na ordem das amostras: (n, 17)."""
    with tarfile.open(tar) as t:
        membros = sorted((m for m in t.getmembers() if m.name.endswith(".behav.npy")),
                         key=lambda m: m.name)
        return np.stack([np.load(io.BytesIO(t.extractfile(m).read()))[0] for m in membros])


def exibicoes_treino(data_path, subj, num_sessions):
    """Exibicoes das primeiras `num_sessions` sessoes: dict com imagem, linha em betas e sessao."""
    behav = [le_behav(f"{data_path}/wds/subj0{subj}/train/{s}.tar") for s in range(num_sessions)]
    return {
        "imagem": np.concatenate([b[:, COL_IMAGEM] for b in behav]).astype(int),
        "beta": np.concatenate([b[:, COL_BETA] for b in behav]).astype(int),
        "sessao": np.concatenate([np.full(len(b), s) for s, b in enumerate(behav)]),
    }


def exibicoes_teste(data_path, subj):
    """Teste novo, agrupado por imagem: ids em ordem crescente e, para cada um, as linhas de betas.

    A ordem crescente e a do all_images.pt e a de qualquer tensor de
    reconstrucao ou de clipvoxels (o recon_inference.py percorre np.unique).
    """
    b = le_behav(f"{data_path}/wds/subj0{subj}/new_test/0.tar")
    imagem, beta = b[:, COL_IMAGEM].astype(int), b[:, COL_BETA].astype(int)
    ids = np.unique(imagem)
    return ids, [beta[imagem == i] for i in ids]


def carrega_betas(data_path, subj):
    """Todos os betas do sujeito: (exibicoes, voxels) em fp32. Para o subj01, 30.000 x 15.724."""
    with h5py.File(f"{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5", "r") as f:
        return f["betas"][:]
