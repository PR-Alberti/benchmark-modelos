"""Experimentos de treino: cada um e um dicionario com o modelo, os hiperparametros dele e o
dataset controlado. O nome do experimento e o nome do modelo treinado (train_logs/<nome>).
notebooks/treino.ipynb mostra o uso com exemplos; pela linha de comando:

    python src/treina.py --lista
    python src/treina.py <nome> --so_dataset     # monta e resume o dataset, sem treinar
    python src/treina.py <nome> --dry_run        # mostra o comando que rodaria
    python src/treina.py <nome>

Cada experimento:

    "modelo":  "frr" | "mindeye2" | "mindeye1"
    "hiper":   hiperparametros do modelo; o que faltar fica no padrao (treina.py, MODELOS)
    "dataset": campos do ConfigDataset (src/mindeye_ridge/dataset_controlado.py):
               sessoes, n_imagens, repeticoes, classes, regra, minimo, maximo_outros,
               balanceado, semente, agregacao; vazio = o dataset completo (40 sessoes)
    "rotulo":  opcional, o nome que o modelo ganha no site (results/site); sem ele, o site monta
               um nome a partir da configuracao

SEMENTE: todo experimento novo usa uma unica semente, sempre a mesma (42, o padrao de "seed" em
treina.py). Nao se treinam varias sementes por modelo: o benchmark compara configuracoes, e com a
semente fixa a diferenca entre duas corridas vem so da configuracao. As sementes 1 e 2 abaixo existem
so para medir o ruido do treino, uma vez, no Ridge 1024 + prior de 1 sessao; nao as repita nem as
estenda a outros modelos. (A "semente" do dataset e outra coisa: fixa em 0.)

"agregacao" diz como o treino usa as repeticoes de cada imagem (src/mindeye_ridge/agregacao.py):
"exibicoes" (cada trial uma amostra), "media" (media dos betas das repeticoes), "sorteio" (uma
repeticao sorteada a cada epoca) ou "combinacao" (combinacao aleatoria das repeticoes, como o baixo
nivel do MindEye1). None, o padrao, e o que o modelo faz no benchmark: exibicoes no FRR e no
MindEye2, rodizio das repeticoes no MindEye1. O FRR aceita so exibicoes e media.

As classes sao listas de colunas da tabela de fracoes da tela (sup_<supercategoria> ou
cat_<categoria> do COCO); notebooks/classificacao_semantica.ipynb mostra quantas imagens cada
escolha deixa.
"""

# =============================================================================================
# Os modelos do benchmark (BENCHMARK.md), escritos como experimentos. O dicionario de cada um
# reproduz o treino que o gerou: o mesmo script, os mesmos hiperparametros e o dataset completo das
# N primeiras sessoes (conferido: o FRR de 1 sessao refeito pelo treina.py da os mesmos numeros, e
# o MindEye2 com e sem o manifesto do dataset completo da as mesmas metricas epoca a epoca).
#
# Eles ja estao treinados, e o treina.py nao treina de novo um nome que ja tem modelo. Para refazer
# um deles, rode o dicionario com outro nome: roda("repro_frr_1sess", BENCHMARK["subj01_frr_1sess"]).
#
# subj01_ridgeonly_1sess_prior e subj01_ridgeonly_1sess vieram do release checkpoints-v1, treinados
# com uma versao anterior do codigo; os retreinos com o codigo atual sao os _seed42.
# =============================================================================================

BENCHMARK = {
    # FRR (src/etapas/run_frr.py): as 20 fracoes de Doerig et al., uma por dimensao do alvo, CV de 5 dobras
    "subj01_frr_1sess": {"modelo": "frr", "hiper": {}, "dataset": {"sessoes": 1}},
    "subj01_frr_40sess": {"modelo": "frr", "hiper": {}, "dataset": {"sessoes": 40}},
    # sensibilidade: uma fracao so para todas as dimensoes, e a grade estendida ate 0,001
    "subj01_frr_1sess_global": {"modelo": "frr", "hiper": {"global_fraction": True}, "dataset": {"sessoes": 1}},
    "subj01_frr_40sess_global": {"modelo": "frr", "hiper": {"global_fraction": True}, "dataset": {"sessoes": 40}},
    "subj01_frr_1sess_ext": {"modelo": "frr", "hiper": {"grid": "extended"}, "dataset": {"sessoes": 1}},
    "subj01_frr_40sess_ext": {"modelo": "frr", "hiper": {"grid": "extended"}, "dataset": {"sessoes": 40}},

    # MindEye2 ridge-only (src/etapas/train_ridgeonly.py): so a ridge do subj01 treina, a partir do
    # pre-treino nos outros 7 sujeitos
    "subj01_ridgeonly_1sess_prior": {"modelo": "mindeye2", "hiper": {}, "dataset": {"sessoes": 1}},
    "subj01_ridgeonly_1sess": {"modelo": "mindeye2", "hiper": {"prior": False}, "dataset": {"sessoes": 1}},
    "subj01_ridgeonly_1sess_4096blurry": {
        "modelo": "mindeye2", "hiper": {"hidden_dim": 4096, "blurry": True, "ckpt_interval": 10},
        "dataset": {"sessoes": 1}},
    "subj01_ridgeonly_40sess_prior": {
        "modelo": "mindeye2", "hiper": {"num_epochs": 20, "frozen_fp16": True, "ckpt_interval": 1},
        "dataset": {"sessoes": 40}},
    "subj01_ridgeonly_40sess_4096blurry": {
        "modelo": "mindeye2",
        "hiper": {"hidden_dim": 4096, "blurry": True, "num_epochs": 20, "frozen_fp16": True, "ckpt_interval": 1},
        "dataset": {"sessoes": 40}},
    # ruido entre sementes (mesma configuracao do 1sess_prior) e o controle sem prior.
    # EXCECAO unica: os outros experimentos usam so a semente 42 (veja SEMENTE no topo do arquivo)
    **{f"subj01_ridgeonly_1sess_prior_seed{s}": {"modelo": "mindeye2", "hiper": {"seed": s}, "dataset": {"sessoes": 1}}
       for s in (42, 1, 2)},
    "subj01_ridgeonly_1sess_noprior_seed42": {
        "modelo": "mindeye2", "hiper": {"prior": False, "seed": 42}, "dataset": {"sessoes": 1}},

    # MindEye1 (mindeye1/src/Train_MindEye.py): treinado do zero, 240 epocas, batch 16 com Adam de 8 bits
    "subj01_me1_1sess": {"modelo": "mindeye1", "hiper": {"save_every": 10}, "dataset": {"sessoes": 1}},
    "subj01_me1_40sess": {"modelo": "mindeye1", "hiper": {}, "dataset": {"sessoes": 40}},
}


# =============================================================================================
# Experimentos com dataset controlado (ainda nao rodados)
# =============================================================================================

# as cinco classes que o notebook de classificacao explorou: com minimo 10% e as outras abaixo de
# 2%, sobram de 336 (comida) a 1.484 (pessoa) imagens de treino nas 40 sessoes
CLASSES_5 = {"pessoa": ["sup_person"], "animal": ["sup_animal"], "veiculo": ["sup_vehicle"],
             "comida": ["sup_food"], "moveis": ["sup_furniture"]}

EXPERIMENTOS = {
    # subconjunto semantico balanceado: 300 imagens por classe, 1 exibicao cada
    "subj01_frr_5classes_300img_1rep": {
        "modelo": "frr",
        "hiper": {"seed": 42},
        "dataset": {"sessoes": 40, "classes": CLASSES_5, "minimo": 10, "maximo_outros": 2,
                    "n_imagens": 300, "balanceado": True, "repeticoes": 1, "semente": 0},
    },

    # MindEye2 ridge-only so com pessoas e animais, todas as exibicoes
    "subj01_me2_pessoa_animal": {
        "modelo": "mindeye2",
        "hiper": {"hidden_dim": 1024, "num_epochs": 20, "batch_size": 16, "max_lr": 3e-4,
                  "prior": True, "frozen_fp16": True, "seed": 42},
        "dataset": {"sessoes": 40, "classes": {"pessoa": ["sup_person"], "animal": ["sup_animal"]}},
    },

    # MindEye1 com 1.000 imagens sorteadas, 3 exibicoes cada
    "subj01_me1_1000img_3rep": {
        "modelo": "mindeye1",
        "hiper": {"num_epochs": 240, "seed": 42},
        "dataset": {"sessoes": 40, "n_imagens": 1000, "repeticoes": 3, "semente": 0},
    },
}

# trials separados x media das repeticoes, com as mesmas 1.000 imagens de 3 exibicoes
for _modo in ("exibicoes", "media"):
    EXPERIMENTOS[f"subj01_frr_1000img_3rep_{_modo}"] = {
        "modelo": "frr",
        "hiper": {"seed": 42},
        "dataset": {"sessoes": 40, "n_imagens": 1000, "repeticoes": 3, "semente": 0, "agregacao": _modo},
    }
# no MindEye2 a epoca acompanha o numero de amostras: 3.000 exibicoes, ou 1.000 imagens nos modos por
# imagem; o triplo de epocas nesses modos deixa os quatro com o mesmo numero de passos
for _modo in ("exibicoes", "media", "sorteio", "combinacao"):
    EXPERIMENTOS[f"subj01_me2_1000img_3rep_{_modo}"] = {
        "modelo": "mindeye2",
        "hiper": {"num_epochs": 20 if _modo == "exibicoes" else 60, "frozen_fp16": True, "seed": 42},
        "dataset": {"sessoes": 40, "n_imagens": 1000, "repeticoes": 3, "semente": 0, "agregacao": _modo},
    }

# imagens unicas x repeticoes com o mesmo numero de exibicoes (3.000): a pergunta do plano de
# estagio sobre o que vale mais para um tempo de aquisicao fixo
for _n, _r in ((3000, 1), (1500, 2), (1000, 3)):
    EXPERIMENTOS[f"subj01_frr_{_n}img_{_r}rep"] = {
        "modelo": "frr",
        "hiper": {"seed": 42},
        "dataset": {"sessoes": 40, "n_imagens": _n, "repeticoes": _r, "semente": 0},
    }

# FRR multi-sujeito (src/etapas/run_frr_ms.py): espaco compartilhado de k = 1.024 dimensoes, peso igual por
# sujeito, ajuste fino so da ridge do subj01. O modelo C pre-treina a leitura nos sujeitos 2-8 (um pre-
# treino so, para as duas quantidades de dados); o B, o controle de posto reduzido, a aprende so com o
# proprio subj01. C contra B mede o efeito dos outros sujeitos; B contra o FRR, o do gargalo de k (o FRR
# de 20 sessoes entra para essa comparacao ter par nas tres quantidades de dados).
EXPERIMENTOS["subj01_frr_20sess"] = {"modelo": "frr", "hiper": {}, "dataset": {"sessoes": 20}}
for _s in (1, 20, 40):
    EXPERIMENTOS[f"subj01_frr_multisujeito_{_s}sess"] = {
        "modelo": "frr_ms",
        "hiper": {"k": 1024, "pesos": "sujeito", "seed": 42},
        "dataset": {"sessoes": _s},
        "rotulo": f"FRR multi-sujeito · {_s} {'sessão' if _s == 1 else 'sessões'}",
    }
    EXPERIMENTOS[f"subj01_frr_postoreduzido_{_s}sess"] = {
        "modelo": "frr_ms",
        "hiper": {"k": 1024, "pretreino": "proprio", "seed": 42},
        "dataset": {"sessoes": _s},
        "rotulo": f"FRR posto reduzido (controle) · {_s} {'sessão' if _s == 1 else 'sessões'}",
    }


# =============================================================================================
# MindEye2 ridge-only com 2, 5, 10 e 20 sessoes, com e sem low-level (branch ME2-intermediate_sessions).
# As pontas, 1 e 40 sessoes, ja estao no benchmark (repro_subj01_ridgeonly_{1,40}sess_prior e
# _4096blurry) e entram na curva sem treinar de novo.
#
# "Sem low-level" e o Ridge 1024 + prior (partida: multisubject_subj01_1024hid_nolow_300ep) e "com" e o
# Ridge 4096 + blurry (partida: final_multisubject_subj01), os mesmos pares do benchmark: o checkpoint com
# o ramo blurry e 4096, entao o low-level e o hidden_dim mudam juntos (veja _valida_mindeye2).
#
# Epocas: as duas pontas usam receitas diferentes (150 epocas com 1 sessao, 20 com 40), entao as
# intermediarias interpolam as epocas na escala geometrica do numero de sessoes,
# 150 * (20/150) ** (ln N / ln 40), arredondado: 100, 60, 40 e 30 epocas (cerca de 8,4 mil, 12,7 mil,
# 16,9 mil e 25,3 mil passos no batch 16, entre os 6,4 mil de 1 sessao e os 33,7 mil de 40). Batch
# e padrao de cada configuracao (16 e 8), congelados em fp16 como nas 40 sessoes, e checkpoint a cada
# epoca. A unica semente e a 42.
# =============================================================================================
EPOCAS_POR_SESSOES = {2: 100, 5: 60, 10: 40, 20: 30}

for _n, _ep in EPOCAS_POR_SESSOES.items():
    _comum = {"num_epochs": _ep, "frozen_fp16": True, "ckpt_interval": 1}
    EXPERIMENTOS[f"subj01_ridgeonly_{_n}sess_prior"] = {
        "modelo": "mindeye2", "hiper": dict(_comum), "dataset": {"sessoes": _n},
        "rotulo": f"Ridge 1024 + prior · {_n} sessões"}
    EXPERIMENTOS[f"subj01_ridgeonly_{_n}sess_4096blurry"] = {
        "modelo": "mindeye2", "hiper": {**_comum, "hidden_dim": 4096, "blurry": True},
        "dataset": {"sessoes": _n}, "rotulo": f"Ridge 4096 + blurry · {_n} sessões"}


# =============================================================================================
# Reproducoes dos modelos do benchmark pelo treina.py (docs/REEXECUTAR.md, scripts/run_reexecutar.sh):
# o mesmo dicionario com outro nome, ja que o treina.py nao sobrescreve um modelo do benchmark. A
# corrida refeita fica com o que as originais nao registraram: configuracao, commit do codigo,
# manifesto do dataset, curva e tempo. As de semente 42 do Ridge 1024 ficam de fora (sao a mesma
# corrida do 1sess_prior e do 1sess). O MindEye1 e refeito por scripts/run_me1_benchmark.sh com
# PREFIXO=repro_ (alto nivel pelo treina.py, depois low-level, reconstrucao e metricas).
# =============================================================================================
_REPRODUZIR = {
    "subj01_frr_1sess": "FRR · 1 sessão", "subj01_frr_1sess_global": "FRR fração global · 1 sessão",
    "subj01_frr_1sess_ext": "FRR grade estendida · 1 sessão", "subj01_frr_40sess": "FRR · 40 sessões",
    "subj01_frr_40sess_global": "FRR fração global · 40 sessões",
    "subj01_frr_40sess_ext": "FRR grade estendida · 40 sessões",
    "subj01_ridgeonly_1sess_prior": "Ridge 1024 + prior · 1 sessão",
    "subj01_ridgeonly_1sess": "Ridge 1024 sem prior · 1 sessão",
    "subj01_ridgeonly_1sess_prior_seed1": "Ridge 1024 + prior · 1 sessão · semente 1",
    "subj01_ridgeonly_1sess_prior_seed2": "Ridge 1024 + prior · 1 sessão · semente 2",
    "subj01_ridgeonly_1sess_4096blurry": "Ridge 4096 + blurry · 1 sessão",
    "subj01_ridgeonly_40sess_prior": "Ridge 1024 + prior · 40 sessões",
    "subj01_ridgeonly_40sess_4096blurry": "Ridge 4096 + blurry · 40 sessões",
    "subj01_me1_1sess": "MindEye1 · 1 sessão", "subj01_me1_40sess": "MindEye1 · 40 sessões",
}
REPRODUCOES = {f"repro_{nome}": {**BENCHMARK[nome], "rotulo": f"{rotulo} · reprodução"}
               for nome, rotulo in _REPRODUZIR.items()}
# MindEye1 sem o low-level: o mesmo treino do alto nivel, so a reconstrucao muda
# (PREFIXO=repro_ LOWLEVEL=0 scripts/run_me1_benchmark.sh)
for _n in (1, 40):
    REPRODUCOES[f"repro_subj01_me1_{_n}sess_semlowlevel"] = {
        **BENCHMARK[f"subj01_me1_{_n}sess"], "rotulo": f"MindEye1 sem low-level · {_n} {'sessão' if _n == 1 else 'sessões'} · reprodução"}

TODOS = {**BENCHMARK, **EXPERIMENTOS, **REPRODUCOES}
assert len(TODOS) == len(BENCHMARK) + len(EXPERIMENTOS) + len(REPRODUCOES), "nome repetido entre os experimentos"
