"""Experimentos de treino: cada um e um dicionario com o modelo, os hiperparametros dele e o
dataset controlado. O nome do experimento e o nome do modelo treinado (train_logs/<nome>).

    python src/treina.py --lista
    python src/treina.py <nome> --so_dataset     # monta e resume o dataset, sem treinar
    python src/treina.py <nome> --dry_run        # mostra o comando que rodaria
    python src/treina.py <nome>

Cada experimento:

    "modelo":  "frr" | "mindeye2" | "mindeye1"
    "hiper":   hiperparametros do modelo; o que faltar fica no padrao (treina.py, MODELOS)
    "dataset": campos do ConfigDataset (src/mindeye_ridge/dataset_controlado.py):
               sessoes, n_imagens, repeticoes, classes, regra, minimo, maximo_outros,
               balanceado, semente, agregacao

"agregacao" diz como o treino usa as repeticoes de cada imagem (src/mindeye_ridge/agregacao.py):
"exibicoes" (cada trial uma amostra), "media" (media dos betas das repeticoes), "sorteio" (uma
repeticao sorteada a cada epoca) ou "combinacao" (combinacao aleatoria das repeticoes, como o baixo
nivel do MindEye1). None, o padrao, e o que o modelo faz no benchmark: exibicoes no FRR e no
MindEye2, rodizio das repeticoes no MindEye1. O FRR aceita so exibicoes e media.

As classes sao listas de colunas da tabela de fracoes da tela (sup_<supercategoria> ou
cat_<categoria> do COCO); notebooks/classificacao_semantica.ipynb mostra quantas imagens cada
escolha deixa.
"""

# as cinco classes que o notebook explorou: com minimo 10% e as outras abaixo de 2%, sobram
# de 336 (comida) a 1.484 (pessoa) imagens de treino nas 40 sessoes
CLASSES_5 = {"pessoa": ["sup_person"], "animal": ["sup_animal"], "veiculo": ["sup_vehicle"],
             "comida": ["sup_food"], "moveis": ["sup_furniture"]}

EXPERIMENTOS = {
    # o ponto de partida, para comparar: igual ao benchmark (40 sessoes inteiras)
    "subj01_frr_40sess_ctrl": {
        "modelo": "frr",
        "hiper": {"folds": 5, "grid": "doerig", "seed": 42},
        "dataset": {"sessoes": 40},
    },

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
