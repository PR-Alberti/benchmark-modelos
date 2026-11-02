"""O decoding muda ao longo da sessao? Selecao dos trials, metricas e agregacao, sem GPU.

O protocolo, explicado em docs/TIREDNESS.md (src/etapas/tiredness_por_run.py roda o modelo):
- treino nas N primeiras sessoes; teste trial a trial, sem media de repeticoes, nas sessoes seguintes;
- entra no teste so o trial cuja imagem o modelo nao viu no treino (o que ele viu vem do manifesto
  train_logs/<modelo>/dataset.json). Os trials das imagens compartilhadas dessas sessoes entram, com
  a marca `compartilhada`: elas nunca estao nos tars de treino;
- as metricas saem por run de cada sessao; o contraste e inicio (runs 1-3) menos fim (runs 10-12),
  dentro da sessao, com bootstrap por sessao. Positivo = pior no fim.

O tempo na sessao se confunde com a repeticao: no fim, muito mais imagens ja foram vistas naquela
mesma sessao, e a tarefa do NSD e reconhecer imagens ja vistas. Por isso cada metrica sai tambem so
com os trials `ineditos`, de imagens ainda nao vistas na sessao.
"""
import numpy as np
import pandas as pd
from scipy.special import gammaln

from . import nsd_data

COL_SESSAO, COL_RUN, COL_TRIAL = 2, 3, 4
N_CANDIDATAS = 300                    # como no retrieval do benchmark
RUNS_INICIO, RUNS_FIM = (1, 2, 3), (10, 11, 12)
METRICAS = ("bwd_top1", "cosseno")


def _tabela(behav, compartilhada):
    return pd.DataFrame({"sessao": behav[:, COL_SESSAO].astype(int), "run": behav[:, COL_RUN].astype(int),
                         "trial": behav[:, COL_TRIAL].astype(int), "beta": behav[:, nsd_data.COL_BETA].astype(int),
                         "imagem": behav[:, nsd_data.COL_IMAGEM].astype(int), "compartilhada": compartilhada})


def exibicoes(data_path, subj):
    """Todos os trials do sujeito, os dos tars de treino e os do teste novo, em ordem de apresentacao."""
    pasta = f"{data_path}/wds/subj0{subj}"
    blocos = [_tabela(nsd_data.le_behav(f"{pasta}/train/{s}.tar"), False)
              for s in range(nsd_data.sessoes(data_path, subj))]
    blocos.append(_tabela(nsd_data.le_behav(f"{pasta}/new_test/0.tar"), True))
    return anota_repeticoes(pd.concat(blocos, ignore_index=True))


def anota_repeticoes(ex):
    """Ordena pela linha em betas (a ordem de apresentacao) e marca o que a imagem ja tinha aparecido."""
    ex = ex.sort_values("beta", ignore_index=True)
    if not ex["beta"].is_unique:
        raise ValueError("ha trials com a mesma linha em betas")
    ex["exibicoes_antes"] = ex.groupby("imagem").cumcount()
    ex["inedito"] = ex.groupby(["imagem", "sessao"]).cumcount() == 0
    return ex


def seleciona_teste(ex, treino):
    """Os trials das sessoes depois do treino, com `no_treino` nos que tem imagem vista no treino.

    `treino` e o manifesto (dataset_controlado.carrega). As sessoes de ex comecam em 1, as do manifesto em 0.
    """
    n_treino = treino["config"]["sessoes"]
    if treino["sessao"].max() >= n_treino:
        raise ValueError(f"o manifesto declara {n_treino} sessoes e usa a de indice {treino['sessao'].max()}")
    teste = ex[ex["sessao"] > n_treino].copy()
    if teste.empty:
        raise ValueError(f"nao ha sessoes de teste depois das {n_treino} de treino")
    if teste["beta"].isin(treino["beta"]).any():
        raise ValueError("um trial do teste esta no treino")
    teste["no_treino"] = teste["imagem"].isin(treino["imagem"])
    return teste


def acerto_top1(n_melhores, n_pool, n_candidatas=N_CANDIDATAS):
    """Probabilidade de acertar, no top-1 entre `n_candidatas`, um trial com `n_melhores` imagens do
    pool mais parecidas com a previsao do que a imagem certa.

    E o valor esperado do retrieval do benchmark sobre todos os sorteios das n_candidatas - 1 distratoras:
    acerta se nenhuma das melhores for sorteada, C(N - m, k) / C(N, k). Sem o ruido dos sorteios.
    """
    if n_pool < n_candidatas:
        raise ValueError(f"pool de {n_pool} imagens: precisa de pelo menos {n_candidatas}")
    m = np.asarray(n_melhores, dtype=float)
    outras, k = n_pool - 1, n_candidatas - 1
    p = np.zeros_like(m)
    ok = m <= outras - k
    p[ok] = np.exp(gammaln(outras - m[ok] + 1) - gammaln(outras - m[ok] - k + 1)
                   - gammaln(outras + 1) + gammaln(outras - k + 1))
    return p


def grupo_do_run(run):
    return "inicio" if run in RUNS_INICIO else "fim" if run in RUNS_FIM else "meio"


def por_run(trials):
    """Uma linha por run de cada sessao; run sem trial avaliado fica com as metricas vazias."""
    linhas = []
    for (sessao, run), g in trials.groupby(["sessao", "run"]):
        aval = g[~g["no_treino"]]
        ineditos = aval[aval["inedito"]]
        linha = {"sessao": sessao, "run": run, "grupo": grupo_do_run(run), "n_trials": len(g),
                 "n_excluidos_treino": int(g["no_treino"].sum()), "n_avaliados": len(aval),
                 "n_ineditos": len(ineditos)}
        for m in METRICAS:
            linha[m] = aval[m].mean()
            linha[f"{m}_ineditos"] = ineditos[m].mean()
        linhas.append(linha)
    return pd.DataFrame(linhas)


def contraste(trials, metrica, n_boot=10000, semente=42):
    """inicio - fim dentro de cada sessao, a media das sessoes e o IC95% do bootstrap por sessao."""
    grupos = trials["run"].map(grupo_do_run)
    medias = trials[grupos != "meio"].groupby(["sessao", grupos[grupos != "meio"]])[metrica].mean().unstack()
    dif = (medias["inicio"] - medias["fim"]).dropna().to_numpy()
    if not len(dif):
        raise ValueError("nenhuma sessao tem trials no inicio e no fim")
    rng = np.random.default_rng(semente)
    boot = dif[rng.integers(len(dif), size=(n_boot, len(dif)))].mean(1)
    return {"inicio": float(medias["inicio"].mean()), "fim": float(medias["fim"].mean()),
            "diferenca": float(dif.mean()), "ic95": [float(q) for q in np.percentile(boot, [2.5, 97.5])],
            "sessoes": len(dif), "n_inicio": int((grupos == "inicio").sum()), "n_fim": int((grupos == "fim").sum())}
