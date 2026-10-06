#!/usr/bin/env python3
"""FRR (fractional ridge regression) de voxels para o embedding CLIP, nas condicoes do benchmark.

Mesmo sujeito, mesmas exibicoes de treino que o ridge-only (as N primeiras sessoes), mesmo
teste (1.000 imagens, 3 repeticoes), mesma entrada (15.724 voxels do nsdgeneral) e mesmo
alvo: o embedding CLIP ViT-bigG/14 achatado (256 x 1664). Segue Doerig et al. (2025):
20 fracoes de 0,05 a 1, validacao cruzada de 5 dobras e uma fracao por dimensao do alvo.
A validacao cruzada agrupa por imagem, entao as repeticoes de uma imagem nunca se separam.
--grid extended acrescenta 10 fracoes menores (analise de sensibilidade; veja FRACS_ESTENDIDA
em frr.py) e --global_fraction usa uma fracao so para todas as dimensoes.

A saida e avaliada com o protocolo do verify_retrieval.py: 30 sorteios de 300 imagens,
top-1 nos dois sentidos, mesma semente. Nao ha reconstrucao: o modelo para no embedding,
como o plano de estagio propoe (o alvo continuo dispensa avaliar a imagem gerada).

    python run_frr.py --model_name subj01_frr_1sess --num_sessions 1
    python run_frr.py --model_name subj01_frr_40sess --num_sessions 40
    python run_frr.py --model_name subj01_frr_animais --num_sessions 40 --dataset ds.json

Com --dataset (manifesto do mindeye_ridge.dataset_controlado), o treino sao as exibicoes do
manifesto em vez das N primeiras sessoes; o teste nao muda. A agregacao do manifesto escolhe entre
cada exibicao como uma linha (o padrao) e a media das repeticoes de cada imagem; os modos sorteados a
cada epoca nao se aplicam, porque a FRR ajusta numa passada.

Grava, em results/evals/<model_name>/:
    <model_name>_all_clipvoxels.pt   embeddings previstos (1000, 256, 1664), fp16
    <model_name>_retrieval.json      mesmo formato do verify_retrieval.py
    <model_name>_frr.json            configuracao, custo, diagnosticos da CV e metricas de embedding
e uma copia do ultimo em results/tables/ (a que entra no git).
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

from mindeye_ridge import agregacao, dataset_controlado, embedding_metrics, frr, nsd_data, paths, utils
from mindeye_ridge.clip_targets import ClipTargets, embeddings_avaliacao, SEQ, DIM


def parse_args():
    parser = argparse.ArgumentParser(description="FRR: voxels -> embedding CLIP")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=str(paths.DATA))
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--num_sessions", type=int, default=1, help="treina nas N primeiras sessoes")
    parser.add_argument("--dataset", type=str, default=None,
                        help="manifesto de um dataset controlado (dataset_controlado.py): treina nessas exibicoes")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid", choices=["doerig", "extended"], default="doerig",
                        help="fracoes testadas: doerig = as 20 de 0,05 a 1 (padrao, o que o plano especifica); "
                             "extended = mais 10 menores, so como sensibilidade (veja FRACS_ESTENDIDA)")
    parser.add_argument("--fracs", type=float, nargs="+", default=None,
                        help="fracoes explicitas, crescentes; sobrepoe --grid")
    parser.add_argument("--global_fraction", action="store_true",
                        help="uma fracao para todas as dimensoes, em vez de uma por dimensao")
    parser.add_argument("--chunk", type=int, default=8192, help="dimensoes do alvo por bloco")
    parser.add_argument("--bias_step", type=float, default=frr.BIAS_STEP)
    parser.add_argument("--cv_dir", type=str, default=None,
                        help="onde guardar as dobras da validacao cruzada (padrao: train_logs/<modelo>/frr_cv); "
                             "aponte para as de outra corrida para trocar so a selecao da fracao, sem refazer a CV")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def reinicia_pico_ram():
    """Zera o pico de RAM do processo (Linux), para medir so a etapa seguinte."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass


def pico_ram_gib():
    with open("/proc/self/status") as f:
        for linha in f:
            if linha.startswith("VmHWM"):
                return int(linha.split()[1]) / 2**20
    return float("nan")


def imagens_do_teste(data_path, ids_teste):
    """all_images.pt, que as outras avaliacoes usam; a i-esima imagem tem de ser a i-esima de ids_teste."""
    todas = torch.load(f"{data_path}/evals/all_images.pt", map_location="cpu")
    amostra = np.linspace(0, len(ids_teste) - 1, 25).astype(int)
    with h5py.File(f"{data_path}/coco_images_224_float16.hdf5", "r") as f:
        vista = torch.from_numpy(f["images"][ids_teste[amostra]]).float()
    assert len(todas) == len(ids_teste) and torch.equal(vista, todas[amostra]), \
        "ordem do teste difere da do all_images.pt"
    return todas


def main():
    args = parse_args()
    t_inicio = time.time()
    utils.seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False       # a FRR inverte curvas: sem arredondar as contas
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saida = f"{paths.EVALS}/{args.model_name}"
    os.makedirs(saida, exist_ok=True)
    os.makedirs(paths.TABLES, exist_ok=True)

    # ------------------------------------------------------------ dados
    treino = dataset_controlado.exibicoes(args.data_path, args.subj, args.num_sessions, args.dataset)
    ids_teste, linhas_teste = nsd_data.exibicoes_teste(args.data_path, args.subj)
    assert not set(treino["imagem"]) & set(ids_teste), "imagem de teste no treino"
    n_exib, n_imagens = len(treino["imagem"]), len(np.unique(treino["imagem"]))
    modo = treino["agregacao"] or "exibicoes"
    if modo not in ("exibicoes", "media"):
        raise ValueError(f"agregacao {modo!r}: a FRR ajusta numa passada (sem epocas); use exibicoes ou media")
    print(f"treino: {args.num_sessions} sessao(oes), {n_exib} exibicoes, {n_imagens} imagens; "
          f"teste: {len(ids_teste)} imagens x {len(linhas_teste[0])} repeticoes", flush=True)
    all_images = imagens_do_teste(args.data_path, ids_teste)

    # o cache cobre todas as imagens de treino do sujeito, para valer com qualquer numero de sessoes
    alvos = ClipTargets(args.data_path, args.subj,
                        nsd_data.exibicoes_treino(args.data_path, args.subj, 40)["imagem"])
    t0 = time.time()
    n_novos = alvos.garante(treino["imagem"], device=device)
    seg_embedding = time.time() - t0

    betas = torch.from_numpy(nsd_data.carrega_betas(args.data_path, args.subj))
    if modo == "media":                                                   # uma linha por imagem
        ids_img, linhas_img = agregacao.grupos(treino["imagem"], treino["beta"])
        X = torch.stack([betas[l].mean(0) for l in linhas_img])
    else:
        ids_img, X = treino["imagem"], betas[treino["beta"]]
    X_teste = torch.stack([betas[r].mean(0) for r in linhas_teste])       # media das repeticoes
    n_voxels = X.shape[1]
    del betas
    groups = alvos.linhas(ids_img)

    # ------------------------------------------------------------ ajuste
    fracs = args.fracs or (frr.FRACS_ESTENDIDA if args.grid == "extended" else frr.FRACS_DOERIG)
    modelo = frr.FracRidge(fracs=fracs, n_folds=args.folds, chunk=args.chunk, bias_step=args.bias_step,
                           per_target=not args.global_fraction, seed=args.seed, device=device)
    reinicia_pico_ram()
    modelo.fit(X, alvos.emb, groups, workdir=args.cv_dir or f"{paths.TRAIN_LOGS}/{args.model_name}/frr_cv")
    t0 = time.time()
    prev = modelo.predict(X_teste, saida=torch.float16)
    # o ajuste inclui as dobras de uma corrida anterior, que o fit() le do disco com o tempo que levaram
    seg_ajuste = modelo.fit_seconds + (time.time() - t0)
    pico_gpu = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    pico_ram = pico_ram_gib()
    print(f"ajuste + previsao: {seg_ajuste:.0f} s (pico {pico_gpu:.1f} GiB na GPU, {pico_ram:.1f} GiB de RAM)", flush=True)
    torch.save(prev.reshape(len(prev), SEQ, DIM), f"{saida}/{args.model_name}_all_clipvoxels.pt")
    cv = modelo.resumo()
    np.save(f"{saida}/{args.model_name}_frr_fracao_por_dimensao.npy", modelo.best_.to(torch.int8).numpy())
    del modelo
    torch.cuda.empty_cache()

    # ------------------------------------------------------------ avaliacao
    # alvo de avaliacao: imagem em fp32, como nas avaliacoes dos outros modelos (ver clip_targets.py)
    Y_teste = embeddings_avaliacao(all_images, device)
    res = embedding_metrics.retrieval_top1(prev, Y_teste, args.seed, device)
    for chave, nome in (("fwd", "fwd (imagem->cerebro)"),
                        ("bwd", "bwd (cerebro->imagem)")):
        r = res[chave]
        print(f"RESULTADO {args.model_name} | {nome}: {r['media']:.4f}  "
              f"IC95% [{r['ic95'][0]:.4f}, {r['ic95'][1]:.4f}]", flush=True)
    with open(f"{saida}/{args.model_name}_retrieval.json", "w") as f:
        json.dump(res, f, indent=1)

    # diagnostico de hubness: o mesmo retrieval depois de tirar a media de treino (previsao e alvo)
    media_treino = alvos.media_ponderada(groups)
    res_centrado = embedding_metrics.retrieval_top1(prev, Y_teste, args.seed, device, centro=media_treino)
    print(f"centrado pela media de treino: fwd {res_centrado['fwd']['media']:.4f}  "
          f"bwd {res_centrado['bwd']['media']:.4f}", flush=True)

    cos, pearson = embedding_metrics.similaridades(prev, Y_teste, device)
    cos_c, _ = embedding_metrics.similaridades(prev, Y_teste, device, centro=media_treino)
    print(f"cosseno medio {cos.mean():.4f}   Pearson medio {pearson.mean():.4f}   "
          f"cosseno centrado {cos_c.mean():.4f}", flush=True)

    relatorio = {
        "modelo": args.model_name,
        "config": {"sujeito": args.subj, "sessoes": args.num_sessions, "exibicoes_treino": n_exib,
                   "dataset": dataset_controlado.carrega(args.dataset)["config"] if args.dataset else None,
                   "agregacao": modo, "amostras_treino": int(len(X)),
                   "imagens_treino": n_imagens, "imagens_teste": len(ids_teste), "voxels": n_voxels,
                   "dim_alvo": int(Y_teste.shape[1]), "dobras": args.folds, "semente": args.seed,
                   "grade": "personalizada" if args.fracs else ("estendida" if args.grid == "extended" else "doerig"),
                   "bias_step": args.bias_step, "bloco": args.chunk},
        "parametros_lineares": n_voxels * int(Y_teste.shape[1]),
        "custo": {"segundos_ajuste": seg_ajuste, "segundos_embedding_alvos": seg_embedding,
                  "embeddings_calculados_nesta_corrida": int(n_novos),
                  "pico_gpu_gib": pico_gpu,
                  "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                  "pico_ram_gib": pico_ram,
                  "segundos_total": time.time() - t_inicio},
        "cv": cv,
        "retrieval": res,
        "retrieval_centrado": res_centrado,
        "hubness": {"cos_previsao_media_treino": float(embedding_metrics.cosseno_com_vetor(prev, media_treino, device).mean()),
                    "cos_alvo_media_treino": float(embedding_metrics.cosseno_com_vetor(Y_teste, media_treino, device).mean())},
        "embedding": {"cosseno_medio": float(cos.mean()), "pearson_medio": float(pearson.mean()),
                      "cosseno_centrado_medio": float(cos_c.mean()),
                      "pearson_mediana": float(np.median(pearson))},
    }
    for destino in (f"{saida}/{args.model_name}_frr.json", f"{paths.TABLES}/{args.model_name}_frr.json"):
        with open(destino, "w") as f:
            json.dump(relatorio, f, indent=1)
    print(f"gravado: {saida}/{args.model_name}_frr.json e {paths.TABLES}/{args.model_name}_frr.json", flush=True)


if __name__ == "__main__":
    main()
