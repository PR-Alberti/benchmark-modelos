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
    # FRR (src/run_frr.py): as 20 fracoes de Doerig et al., uma por dimensao do alvo, CV de 5 dobras
    "subj01_frr_1sess": {"modelo": "frr", "hiper": {}, "dataset": {"sessoes": 1}},
    "subj01_frr_40sess": {"modelo": "frr", "hiper": {}, "dataset": {"sessoes": 40}},
    # sensibilidade: uma fracao so para todas as dimensoes, e a grade estendida ate 0,001
    "subj01_frr_1sess_global": {"modelo": "frr", "hiper": {"global_fraction": True}, "dataset": {"sessoes": 1}},
    "subj01_frr_40sess_global": {"modelo": "frr", "hiper": {"global_fraction": True}, "dataset": {"sessoes": 40}},
    "subj01_frr_1sess_ext": {"modelo": "frr", "hiper": {"grid": "extended"}, "dataset": {"sessoes": 1}},
    "subj01_frr_40sess_ext": {"modelo": "frr", "hiper": {"grid": "extended"}, "dataset": {"sessoes": 40}},

    # MindEye2 ridge-only (src/train_ridgeonly.py): so a ridge do subj01 treina, a partir do
    # pre-treino nos outros 7 sujeitos
    "subj01_ridgeonly_1sess_prior": {"modelo": "mindeye2", "hiper": {}, "dataset": {"sessoes": 1}},
    "subj01_ridgeonly_1sess": {"modelo": "mindeye2", "hiper": {"prior": False}, "dataset": {"sessoes": 1}},
    "subj01_ridgeonly_1sess_4096blurry": {
        "modelo": "mindeye2", "hiper": {"hidden_dim": 4096, "blurry": True, "ckpt_interval": 10},
        "dataset": {"sessoes": 1}},
    "subj01_ridgeonly_40sess_prior": {
        "modelo": "mindeye2", "hiper": {"num_epochs": 20, "frozen_fp16": True, "ckpt_interval": 1},
        "dataset": {"sessoes": 40}},
    # ruido entre sementes (mesma configuracao do 1sess_prior) e o controle sem prior
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

TODOS = {**BENCHMARK, **EXPERIMENTOS}
assert len(TODOS) == len(BENCHMARK) + len(EXPERIMENTOS), "nome repetido entre BENCHMARK e EXPERIMENTOS"
