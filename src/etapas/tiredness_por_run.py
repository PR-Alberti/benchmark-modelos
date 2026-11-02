"""Metricas de decoding por run de cada sessao, trial a trial, nas sessoes depois do treino.

    scripts/run_tiredness.sh                      # treina (se preciso) e avalia
    python src/etapas/tiredness_por_run.py --model_name subj01_ridgeonly_10sess_prior

O protocolo esta em mindeye_ridge/tiredness.py. O modelo preve o embedding de cada trial sem difusao,
como o verify_retrieval.py, mas sem a media das repeticoes. O pool de candidatas sao as imagens
avaliadas; as metricas sao o retrieval cerebro -> imagem (bwd) e o cosseno com a imagem certa.

Grava em results/tables/tiredness/:
    <modelo>_por_run.csv     uma linha por run de cada sessao
    <modelo>_por_trial.csv   um trial por linha, com os excluidos (metricas vazias)
    <modelo>_contraste.json  inicio (runs 1-3) menos fim (runs 10-12), com os dados do protocolo
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mindeye_ridge import paths

paths.add_vendored_to_path()
import sgm  # noqa: E402,F401
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # noqa: E402

from mindeye_ridge import dataset_controlado, tiredness  # noqa: E402
from mindeye_ridge.models import BrainNetwork, MindEyeModule, RidgeRegression  # noqa: E402

CLIP_SEQ, CLIP_DIM = 256, 1664
# de onde vem o backbone de cada hidden_dim, como no run_sessoes_intermediarias.sh
CKPT_PARTIDA = {1024: "multisubject_subj01_1024hid_nolow_300ep", 4096: "final_multisubject_subj01"}


def carrega_modelo(nome, hidden_dim, blurry, num_voxels, data_path, device, n_blocks=4):
    """Ridge do modelo treinado e backbone do ckpt de partida; o prior nao e usado aqui."""
    model = MindEyeModule()
    model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)
    model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks, clip_size=CLIP_DIM,
                                  out_dim=CLIP_DIM * CLIP_SEQ, blurry_recon=blurry)
    proprio = torch.load(paths.TRAIN_LOGS / nome / "last.pth", map_location="cpu")["model_state_dict"]
    partida = f"{data_path}/train_logs/{CKPT_PARTIDA[hidden_dim]}/last.pth"
    base = torch.load(partida, map_location="cpu", mmap=True)["model_state_dict"]
    estado = {k: v for k, v in base.items() if k.startswith("backbone.")}
    estado.update({k: v for k, v in proprio.items() if k.startswith("ridge.")})
    model.load_state_dict(estado, strict=True)
    return model.to(device).eval().requires_grad_(False)


def embeddings_imagens(ids, data_path, device, lote=64):
    """Embedding CLIP normalizado de cada imagem de `ids` (crescentes), na CPU, em fp32."""
    embedder = FrozenOpenCLIPImageEmbedder(arch="ViT-bigG-14", version="laion2b_s39b_b160k", output_tokens=True,
                                           only_tokens=True).half().to(device)
    saida = torch.empty(len(ids), CLIP_SEQ * CLIP_DIM, dtype=torch.float32)
    with h5py.File(f"{data_path}/coco_images_224_float16.hdf5", "r") as f, torch.no_grad(), \
            torch.cuda.amp.autocast(dtype=torch.float16):
        for i in tqdm(range(0, len(ids), lote), desc="imagens"):
            x = torch.tensor(f["images"][ids[i:i + lote]], dtype=torch.float32).to(device)
            saida[i:i + lote] = nn.functional.normalize(embedder(x).float().flatten(1), dim=-1).cpu()
    del embedder
    torch.cuda.empty_cache()
    return saida


def avalia(model, betas, avaliados, ids, pool, device, lote=64):
    """Para cada trial: o cosseno com a imagem certa e quantas imagens do pool ficaram mais parecidas."""
    linhas = avaliados["beta"].to_numpy()
    certa = torch.from_numpy(np.searchsorted(ids, avaliados["imagem"].to_numpy())).to(device)
    cosseno, n_melhores = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(linhas), lote), desc="trials"):
            voxel = torch.from_numpy(betas[linhas[i:i + lote]]).to(device)[:, None]
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _, prev, _ = model.backbone(model.ridge(voxel, 0))
            sim = nn.functional.normalize(prev.float().flatten(1), dim=-1) @ pool.T
            alvo = sim[torch.arange(len(sim)), certa[i:i + lote]]
            cosseno.append(alvo.cpu())
            n_melhores.append((sim > alvo[:, None]).sum(1).cpu())
    return torch.cat(cosseno).numpy(), torch.cat(n_melhores).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True)
    p.add_argument("--data_path", default=str(paths.DATA))
    p.add_argument("--subj", type=int, default=1)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    device = torch.device("cuda")

    pasta = paths.TRAIN_LOGS / a.model_name
    with open(pasta / "experimento.json") as f:
        hiper = json.load(f)["hiper"]
    treino = dataset_controlado.carrega(pasta / "dataset.json")
    trials = tiredness.seleciona_teste(tiredness.exibicoes(a.data_path, a.subj), treino)
    avaliados = trials[~trials["no_treino"]]
    # a garantia do protocolo, conferida sobre o que vai ser medido e nao sobre o filtro
    assert not set(avaliados["imagem"]) & set(treino["imagem"].tolist()), "imagem do treino no teste"
    ids = np.unique(avaliados["imagem"].to_numpy())
    print(f"{a.model_name}: treino em {treino['config']['sessoes']} sessoes ({len(np.unique(treino['imagem']))} imagens); "
          f"teste nas sessoes {trials['sessao'].min()}-{trials['sessao'].max()}: {len(trials)} trials, "
          f"{int(trials['no_treino'].sum())} excluidos por imagem vista no treino, {len(avaliados)} avaliados, "
          f"{len(ids)} imagens no pool", flush=True)

    pool = embeddings_imagens(ids, a.data_path, device, a.lote).to(device)
    with h5py.File(f"{a.data_path}/betas_all_subj0{a.subj}_fp32_renorm.hdf5", "r") as f:
        betas = f["betas"][:]
    model = carrega_modelo(a.model_name, hiper["hidden_dim"], hiper["blurry"], betas.shape[1], a.data_path, device)
    cosseno, n_melhores = avalia(model, betas, avaliados, ids, pool, device, a.lote)

    trials["cosseno"], trials["n_melhores"], trials["bwd_top1"] = np.nan, np.nan, np.nan
    trials.loc[avaliados.index, "cosseno"] = cosseno
    trials.loc[avaliados.index, "n_melhores"] = n_melhores
    trials.loc[avaliados.index, "bwd_top1"] = tiredness.acerto_top1(n_melhores, len(ids))

    saida = paths.TABLES / "tiredness"
    os.makedirs(saida, exist_ok=True)
    trials.to_csv(saida / f"{a.model_name}_por_trial.csv", index=False)
    tiredness.por_run(trials).to_csv(saida / f"{a.model_name}_por_run.csv", index=False)
    aval = trials[~trials["no_treino"]]
    resultado = {"modelo": a.model_name, "sessoes_treino": treino["config"]["sessoes"],
                 "sessoes_teste": [int(trials["sessao"].min()), int(trials["sessao"].max())],
                 "n_trials": len(trials), "n_excluidos_treino": int(trials["no_treino"].sum()),
                 "n_avaliados": len(aval), "n_pool": len(ids), "n_candidatas": tiredness.N_CANDIDATAS,
                 "runs_inicio": tiredness.RUNS_INICIO, "runs_fim": tiredness.RUNS_FIM,
                 "contraste_inicio_menos_fim": {
                     m: {"todos": tiredness.contraste(aval, m, semente=a.seed),
                         "ineditos": tiredness.contraste(aval[aval["inedito"]], m, semente=a.seed)}
                     for m in tiredness.METRICAS}}
    with open(saida / f"{a.model_name}_contraste.json", "w") as f:
        json.dump(resultado, f, indent=1)
    print(json.dumps(resultado["contraste_inicio_menos_fim"], indent=1), flush=True)


if __name__ == "__main__":
    main()
