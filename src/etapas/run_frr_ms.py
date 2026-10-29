#!/usr/bin/env python3
"""FRR multi-sujeito: pre-treino linear de um espaco compartilhado e ajuste fino so da ridge do subj01.

    voxels do sujeito --R_s--> k numeros --Q--> embedding CLIP bigG (256 x 1664)

Dois modelos, o mesmo codigo (src/mindeye_ridge/multisujeito.py):

  C  --pretreino 2 3 4 5 6 7 8 (padrao): a leitura Q sai dos outros 7 sujeitos do NSD (todas as sessoes,
     peso igual por sujeito), por minimos quadrados alternados; o pre-treino e um so para 1 e 40
     sessoes do subj01 e fica em train_logs/_pretreino_frr_ms/<assinatura>/;
  B  --pretreino proprio: o controle de posto reduzido. A mesma arquitetura, com a leitura Q aprendida
     so com o treino do proprio subj01: separa o efeito do gargalo de k do efeito dos outros sujeitos.

Nos dois, o ajuste fino e o FRR comum (5 dobras, uma fracao por dimensao) dos voxels do subj01 para
T = (Y - ybar) Q', com Q congelada, e a previsao e ybar + t Q. Mesmo treino, teste, alvo e avaliacao do
run_frr.py; a saida tem os mesmos arquivos, entao --reconstrucao, o relatorio e o site servem igual.

    python src/etapas/run_frr_ms.py --model_name subj01_frr_multisujeito_1sess --num_sessions 1
    python src/etapas/run_frr_ms.py --model_name subj01_frr_postoreduzido_1sess --num_sessions 1 --pretreino proprio

Antes de qualquer conta, confere que nenhuma imagem de teste do subj01 (as 1.000 compartilhadas)
aparece no treino de um sujeito do pre-treino. Os alvos CLIP de cada sujeito ficam em cache em
<dados>/clip_targets/subj0N (~7 GB por sujeito, calculados na primeira vez).
"""
import argparse
import os
import time

import numpy as np
import torch

import run_frr
from mindeye_ridge import dataset_controlado, frr, multisujeito, nsd_data, paths, utils
from mindeye_ridge.clip_targets import ClipTargets, SEQ, DIM, D_ALVO

SUJEITOS_PRETREINO = [2, 3, 4, 5, 6, 7, 8]


def parse_args():
    p = argparse.ArgumentParser(description="FRR multi-sujeito: voxels -> espaco compartilhado -> embedding CLIP")
    p.add_argument("--model_name", required=True)
    p.add_argument("--data_path", default=str(paths.DATA))
    p.add_argument("--subj", type=int, default=1, help="o sujeito do ajuste fino e do teste")
    p.add_argument("--num_sessions", type=int, default=1)
    p.add_argument("--dataset", default=None, help="manifesto de um dataset controlado (o treino do subj01)")
    p.add_argument("--k", type=int, default=1024, help="dimensao do espaco compartilhado")
    p.add_argument("--pretreino", nargs="+", default=[str(s) for s in SUJEITOS_PRETREINO],
                   help="sujeitos do pre-treino (modelo C) ou 'proprio' (modelo B: so o subj01)")
    p.add_argument("--pesos", choices=["sujeito", "exibicao"], default="sujeito",
                   help="peso no pre-treino: igual por sujeito (como os lotes do MindEye2) ou por exibicao")
    p.add_argument("--iteracoes", type=int, default=10, help="maximo de iteracoes dos minimos alternados")
    p.add_argument("--tolerancia", type=float, default=1e-3, help="ganho relativo minimo na validacao para seguir")
    p.add_argument("--validacao", type=float, default=0.1, help="fracao das imagens de cada sujeito na validacao do pre-treino")
    p.add_argument("--grid", choices=["doerig", "extended"], default="doerig")
    p.add_argument("--fracs", type=float, nargs="+", default=None)
    p.add_argument("--folds", type=int, default=5, help="dobras da validacao cruzada do ajuste fino")
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretreino_dir", default=None, help="onde fica o pre-treino (padrao: o da assinatura)")
    return p.parse_args()


def confere_vazamento(nome, ids_treino, ids_teste):
    comum = np.intersect1d(np.unique(ids_treino), ids_teste)
    if len(comum):
        raise SystemExit(f"{nome}: {len(comum)} imagens do teste do subj01 no treino do pre-treino "
                         f"(por exemplo {comum[:5].tolist()}); o pre-treino nao pode ve-las")


def sujeitos_do_pretreino(data_path, lista, ids_teste, device):
    """Os sujeitos do pre-treino: todas as sessoes de treino, os alvos CLIP garantidos no cache e os
    voxels lidos so quando a base espectral precisa (e ainda nao esta em disco)."""
    sujeitos, meta = [], []
    for s in lista:
        n = nsd_data.sessoes(data_path, s)
        ex = nsd_data.exibicoes_treino(data_path, s, n)
        confere_vazamento(f"subj0{s}", ex["imagem"], ids_teste)
        alvos = ClipTargets(data_path, s, ex["imagem"])
        alvos.garante(ex["imagem"], device=device)
        print(f"subj0{s}: {n} sessoes, {len(ex['imagem'])} exibicoes, {len(alvos.ids)} imagens", flush=True)

        def carrega(s=s, beta=ex["beta"]):
            return torch.from_numpy(nsd_data.carrega_betas(data_path, s)[beta])
        sujeitos.append(multisujeito.Sujeito(f"subj0{s}", carrega, alvos.linhas(ex["imagem"]), alvos.emb))
        meta.append({"sujeito": s, "sessoes": n, "exibicoes": int(len(ex["imagem"])), "imagens": int(len(alvos.ids))})
    return sujeitos, meta


def main():
    args = parse_args()
    t_inicio = time.time()
    utils.seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False       # como no run_frr.py: a FRR inverte curvas
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saida = f"{paths.EVALS}/{args.model_name}"
    os.makedirs(saida, exist_ok=True)
    os.makedirs(paths.TABLES, exist_ok=True)
    proprio = args.pretreino == ["proprio"]
    lista = None if proprio else sorted({int(s) for s in args.pretreino})
    if lista and args.subj in lista:
        raise SystemExit(f"o subj0{args.subj} e o do ajuste fino: tire-o de --pretreino")
    fracs = args.fracs or (frr.FRACS_ESTENDIDA if args.grid == "extended" else frr.FRACS_DOERIG)

    # ------------------------------------------------------------ dados do subj01 (os do run_frr.py)
    treino = dataset_controlado.exibicoes(args.data_path, args.subj, args.num_sessions, args.dataset)
    ids_teste, linhas_teste = nsd_data.exibicoes_teste(args.data_path, args.subj)
    confere_vazamento(f"subj0{args.subj}", treino["imagem"], ids_teste)
    if (treino["agregacao"] or "exibicoes") != "exibicoes":
        raise ValueError(f"agregacao {treino['agregacao']!r}: o FRR multi-sujeito usa cada exibicao como amostra")
    n_exib, n_imagens = len(treino["imagem"]), len(np.unique(treino["imagem"]))
    print(f"subj0{args.subj}: {args.num_sessions} sessao(oes), {n_exib} exibicoes, {n_imagens} imagens; "
          f"teste: {len(ids_teste)} imagens x {len(linhas_teste[0])} repeticoes", flush=True)
    all_images = run_frr.imagens_do_teste(args.data_path, ids_teste)
    alvos = ClipTargets(args.data_path, args.subj,
                        nsd_data.exibicoes_treino(args.data_path, args.subj, 40)["imagem"])
    t0 = time.time()
    n_novos = alvos.garante(treino["imagem"], device=device)
    seg_embedding = time.time() - t0
    betas = torch.from_numpy(nsd_data.carrega_betas(args.data_path, args.subj))
    X = betas[treino["beta"]]
    X_teste = torch.stack([betas[r].mean(0) for r in linhas_teste])       # media das repeticoes
    n_voxels = X.shape[1]
    del betas
    groups = alvos.linhas(treino["imagem"])

    # ------------------------------------------------------------ pre-treino: a leitura Q
    modelo = multisujeito.FRRMultiSujeito(k=args.k, fracs=fracs, pesos=args.pesos, iteracoes=args.iteracoes,
                                          tolerancia=args.tolerancia, validacao=args.validacao, seed=args.seed,
                                          chunk=args.chunk, device=device)
    if proprio:
        sujeitos, meta = [multisujeito.Sujeito(f"subj0{args.subj}", lambda: X, groups, alvos.emb)], \
            [{"sujeito": args.subj, "exibicoes": n_exib, "imagens": n_imagens}]
        pre_dir = args.pretreino_dir or str(paths.TRAIN_LOGS / args.model_name / "pretreino")
    else:
        sujeitos, meta = sujeitos_do_pretreino(args.data_path, lista, ids_teste, device)
        pre_dir = args.pretreino_dir or multisujeito.diretorio_pretreino(str(paths.TRAIN_LOGS), modelo.config(), meta)
    print(f"pre-treino em {pre_dir}", flush=True)
    run_frr.reinicia_pico_ram()
    modelo.ajusta(sujeitos, pre_dir)
    Q = modelo.Q
    k_ef = int(Q.shape[0])

    # ------------------------------------------------------------ ajuste fino: so a ridge do subj01
    t1 = time.time()
    img, inv = np.unique(groups, return_inverse=True)
    T, ybar = multisujeito.alvo_reduzido(alvos.emb, img, np.bincount(inv), Q, args.chunk, device)
    seg_reduz = time.time() - t1
    fino = frr.FracRidge(fracs=fracs, n_folds=args.folds, chunk=args.chunk, seed=args.seed, device=device)
    fino.fit(X, T, inv, workdir=str(paths.TRAIN_LOGS / args.model_name / "frr_cv"))
    t2 = time.time()
    t = fino.predict(X_teste, saida=torch.float32)
    prev = multisujeito.expande(t, ybar, Q, args.chunk, device)
    # o fit_seconds soma o que cada dobra gastou, inclusive as retomadas de uma corrida anterior
    seg_ajuste = seg_reduz + fino.fit_seconds + (time.time() - t2)
    pico_gpu = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    pico_ram = run_frr.pico_ram_gib()
    print(f"ajuste fino + previsao: {seg_ajuste:.0f} s (pico {pico_gpu:.1f} GiB na GPU, {pico_ram:.1f} GiB de RAM)", flush=True)
    torch.save(prev.reshape(len(prev), SEQ, DIM), f"{saida}/{args.model_name}_all_clipvoxels.pt")
    cv = fino.resumo()
    del fino
    torch.cuda.empty_cache()

    # ------------------------------------------------------------ avaliacao (a do run_frr.py)
    aval = run_frr.avalia(args.model_name, prev, all_images, alvos.media_ponderada(groups), args.seed, device, saida)
    relatorio = {
        "modelo": args.model_name,
        "config": {"modelo": "frr_ms", "variante": "posto reduzido (so o proprio sujeito)" if proprio else "multi-sujeito",
                   "sujeito": args.subj, "sessoes": args.num_sessions, "exibicoes_treino": n_exib,
                   "dataset": dataset_controlado.carrega(args.dataset)["config"] if args.dataset else None,
                   "agregacao": "exibicoes", "amostras_treino": int(len(X)),
                   "imagens_treino": n_imagens, "imagens_teste": len(ids_teste), "voxels": n_voxels,
                   "dim_alvo": aval.pop("dim_alvo"), "k": args.k, "k_efetivo": k_ef,
                   "pretreino": "proprio" if proprio else lista, "pesos": args.pesos,
                   "dobras": args.folds, "semente": args.seed,
                   "grade": "personalizada" if args.fracs else ("estendida" if args.grid == "extended" else "doerig"),
                   "bloco": args.chunk},
        # a leitura (k_ef x D) e compartilhada; do subj01 so sai a ridge (voxels x k_ef)
        "parametros_lineares": n_voxels * k_ef + k_ef * D_ALVO,
        "parametros_ajustados_sujeito": n_voxels * k_ef,
        "custo": {"segundos_ajuste": seg_ajuste, "segundos_pretreino": modelo.segundos,
                  "segundos_embedding_alvos": seg_embedding, "embeddings_calculados_nesta_corrida": int(n_novos),
                  "pico_gpu_gib": pico_gpu, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                  "pico_ram_gib": pico_ram, "segundos_total": time.time() - t_inicio},
        "cv": cv,
        "pretreino": {**modelo.resumo(), "diretorio": pre_dir, "dados": meta},
        **aval,
    }
    run_frr.grava_relatorio(args.model_name, saida, relatorio)


if __name__ == "__main__":
    main()
