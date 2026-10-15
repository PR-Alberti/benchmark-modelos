# benchmark-modelos

Decodificação da imagem vista a partir do fMRI (NSD, sujeito 1): MindEye2, FRR e MindEye1, com o
mesmo treino, o mesmo teste (1.000 imagens compartilhadas) e as mesmas métricas.

| Modelo | Código | Saída |
|---|---|---|
| **MindEye2** ridge-only ([Scotti et al., 2024](https://arxiv.org/abs/2403.11207)) | `src/train_ridgeonly.py` | embedding CLIP ViT-bigG + imagem (SDXL unCLIP) |
| **FRR**, ridge fracionária ([Rokem & Kay, 2020](https://doi.org/10.1093/gigascience/giaa133)) | `src/run_frr.py` | embedding CLIP ViT-bigG |
| **MindEye1** ([Scotti et al., 2023](https://arxiv.org/abs/2305.18274)) | `mindeye1/src/Train_MindEye.py` | embedding CLIP ViT-L/14 + imagem (Versatile Diffusion) |

Site com início, uso e resultados interativos de todos os modelos: [`results/site/index.html`](results/site/index.html)
(abre direto do disco; regenerado ao fim de cada treino e avaliação).

## Instalação

```bash
git clone https://github.com/PR-Alberti/benchmark-modelos.git && cd benchmark-modelos
setup/bootstrap.sh --check        # diagnóstico, não baixa nada
setup/bootstrap.sh                # tudo
setup/bootstrap.sh --minimo       # só ridge-only e FRR de 1 sessão
setup/bootstrap.sh --ckpts        # + checkpoints ridge-only do release (repositório privado)
~/envs/fmri/bin/python -m unittest discover -s tests
```

| Preparação | Download | Disco |
|---|---|---|
| `--minimo` | ~35 GB | 40 GB |
| FRR + MindEye2 nas 40 sessões (`setup_env.sh --with-extras` + `download_data.py --stage finetune --subj 1`) | ~43 GB | 60 GB |
| completo | ~130 GB | 200 GB |

| Requisito | |
|---|---|
| GPU | NVIDIA, 20 GB (16 GB para FRR e MindEye2) |
| RAM | 30 GB |
| Python | conda ou python3.11 (3.12+ não serve) |
| Dados | [termos do NSD](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) + [formulário](https://forms.gle/xue2bCdM9LaFNMeb7) |

| Variável | Padrão |
|---|---|
| `MINDEYE_ENV` | `~/envs/fmri` |
| `MINDEYE_DATA` | `~/mindeyev2` |
| `ME1_DATA` | `~/mindeye1` |

Máquina só com terminal, passo a passo: [docs/NOVO_COMPUTADOR.md](docs/NOVO_COMPUTADOR.md).

## Uso

### Treino por dicionário

```bash
cd src
python treina.py --lista                  # experimentos de src/experimentos.py
python treina.py <nome> --so_dataset      # monta e resume o dataset
python treina.py <nome> --dry_run         # mostra o comando
python treina.py <nome> [<nome> ...]      # treina
python treina.py <nome> --retoma          # MindEye2: continua do last.pth
```

```python
import treina                               # de dentro de src/
treina.roda("meu_modelo", {
    "modelo": "frr",                        # frr | mindeye2 | mindeye1
    "hiper": {"seed": 42},
    "dataset": {"sessoes": 40, "classes": {"animal": ["sup_animal"]}, "minimo": 10},
    "rotulo": "FRR · só animais",           # opcional: o nome no site
}, dry_run=True)
```

Notebook: [notebooks/treino.ipynb](notebooks/treino.ipynb) (kernel `~/envs/fmri/bin/python`).

### Scripts

```bash
# MindEye2 ridge-only
scripts/run_ridgeonly_prior.sh                                        # 1 sessão
MODEL_NAME=subj01_ridgeonly_40sess_prior NUM_SESSIONS=40 NUM_EPOCHS=20 FROZEN=1 scripts/run_ridgeonly_prior.sh
scripts/run_recon.sh <nome> 1024 noblurry                             # reconstrução + refinamento
scripts/run_evals.sh <nome> enhanced                                  # métricas

# FRR
scripts/run_frr.sh                                                    # 1 sessão
NUM_SESSIONS=40 scripts/run_frr.sh                                    # 40 sessões

# MindEye1
scripts/run_me1_benchmark.sh                                          # 1 e 40 sessões, treino → métricas
SESSOES=1 scripts/run_me1_benchmark.sh                                # só 1 sessão
LOWLEVEL=0 scripts/run_me1_benchmark.sh                               # sem low-level
NUM_SESSIONS=1 scripts/me1_run.sh train|lowlevel|recon|metrics        # uma etapa

# Dataset controlado direto nos scripts
MODEL_NAME=<nome> DATASET=train_logs/<nome>/dataset.json NUM_SESSIONS=40 scripts/run_frr.sh

# Benchmark
scripts/run_benchmark.sh                                              # MindEye2 + FRR, tudo
python src/make_benchmark.py                                          # remonta BENCHMARK.md e a página
python src/make_site.py                                               # regenera o site (results/site)
```

### Resultado de um modelo

```bash
cd src && python -c "from mindeye_ridge import resultados; print(resultados.retrieval('<nome>'))"
```

| Onde | O quê |
|---|---|
| `train_logs/<nome>/` | checkpoint, logs, `dataset.json`, `experimento.json` (MindEye1: `mindeye1/train_logs/<nome>/`) |
| `results/tables/` | métricas finais |
| `results/evals/<nome>/` | tensores de embedding e reconstrução |
| `results/site/index.html` | o site: todos os modelos, com tabelas, gráficos, curvas, galeria e fichas |
| `results/site/modelos/<nome>.json` | o registro do modelo no site (o que foi extraído de cada resultado) |
| [BENCHMARK.md](BENCHMARK.md), `results/benchmark/index.html` | tabelas e página do benchmark |

## Combinações possíveis

### Dataset (`"dataset"`)

| Campo | Valores | Padrão |
|---|---|---|
| `sessoes` | 1 a 40 | 40 |
| `n_imagens` | inteiro (por classe com `balanceado`) | todas |
| `repeticoes` | 1, 2, 3 | todas as do pool |
| `classes` | `{nome: [colunas sup_*/cat_*]}` (exclusiva) ou lista de `pessoa`, `animal`, `outro` (prioridade) | nenhuma |
| `regra` | `exclusiva`, `prioridade` | `exclusiva` |
| `minimo` | % da tela da classe | 10 |
| `maximo_outros` | % da tela das outras classes listadas | 2 |
| `balanceado` | `True`, `False` | `False` |
| `semente` | inteiro | 0 |
| `agregacao` | `exibicoes`, `media`, `sorteio`, `combinacao` | `None` (nativo do modelo) |

`{}` = dataset completo (40 sessões), igual ao benchmark.

### Agregação × modelo

| `agregacao` | FRR | MindEye2 | MindEye1 |
|---|---|---|---|
| `None` | = `exibicoes` | = `exibicoes` | rodízio das repetições |
| `exibicoes` | ✓ | ✓ | ✓ |
| `media` | ✓ | ✓ | ✓ |
| `sorteio` | — | ✓ | ✓ |
| `combinacao` | — | ✓ | ✓ |

### Hiperparâmetros (`"hiper"`)

| Modelo | Chave e padrão |
|---|---|
| `frr` | `folds=5`, `grid="doerig"` (ou `"extended"`), `fracs=None`, `global_fraction=False`, `seed=42`, `chunk=8192` |
| `mindeye2` | `hidden_dim=1024` (ou 4096), `num_epochs=150`, `batch_size=None`, `max_lr=3e-4`, `prior=True`, `blurry=False`, `frozen_fp16=None`, `seed=42`, `ckpt_interval=None` |
| `mindeye1` | `num_epochs=240`, `max_lr=3e-4`, `seed=42`, `mixup_pct=0.33`, `batch_size=None` (16), `paper=False`, `save_every=1` |

| Restrição | |
|---|---|
| `mindeye2`, `mindeye1` | `num_epochs >= 2` |
| `mindeye2` | `blurry=True` só com `hidden_dim=4096` |
| `mindeye1` | imagens únicas >= `batch_size` |
| `frr` | `agregacao` só `exibicoes` ou `media` |
| todos | sujeito 1; nome já treinado é recusado |

## Combinações testadas

Treinadas e avaliadas (retrieval top-1 imagem → cérebro / cérebro → imagem, em %). Todas com o
dataset completo das N primeiras sessões; os dicionários estão em `BENCHMARK` (`src/experimentos.py`).

| Modelo | Nome | Sessões | Hiperparâmetros | fwd | bwd |
|---|---|---|---|---|---|
| FRR | `subj01_frr_1sess` | 1 | padrão | 55,0 | 3,1 |
| FRR | `subj01_frr_1sess_global` | 1 | `global_fraction` | 33,2 | 1,3 |
| FRR | `subj01_frr_1sess_ext` | 1 | `grid="extended"` | 55,7 | 2,0 |
| FRR | `subj01_frr_40sess` | 40 | padrão | 96,4 | 37,7 |
| FRR | `subj01_frr_40sess_global` | 40 | `global_fraction` | 96,3 | 37,7 |
| FRR | `subj01_frr_40sess_ext` | 40 | `grid="extended"` | 92,4 | 21,3 |
| MindEye2 | `subj01_ridgeonly_1sess_prior` | 1 | padrão | 92,8 | 88,1 |
| MindEye2 | `subj01_ridgeonly_1sess` | 1 | `prior=False` | 78,5 | 72,1 |
| MindEye2 | `subj01_ridgeonly_1sess_4096blurry` | 1 | `hidden_dim=4096`, `blurry` | 95,5 | 92,0 |
| MindEye2 | `subj01_ridgeonly_1sess_prior_seed42` | 1 | `seed=42` | 92,9 | 88,1 |
| MindEye2 | `subj01_ridgeonly_1sess_prior_seed1` | 1 | `seed=1` | 93,1 | 87,8 |
| MindEye2 | `subj01_ridgeonly_1sess_prior_seed2` | 1 | `seed=2` | 93,2 | 88,1 |
| MindEye2 | `subj01_ridgeonly_1sess_noprior_seed42` | 1 | `prior=False`, `seed=42` | 78,4 | 72,6 |
| MindEye2 | `subj01_ridgeonly_40sess_prior` | 40 | `num_epochs=20`, `frozen_fp16` | 99,9 | 99,5 |
| MindEye2 | `subj01_ridgeonly_40sess_4096blurry` | 40 | `hidden_dim=4096`, `blurry`, `num_epochs=20` | 100,0 | 99,4 |
| MindEye1 | `subj01_me1_1sess` | 1 | padrão + low-level | 30,6 | 23,2 |
| MindEye1 | `subj01_me1_1sess_semlowlevel` | 1 | padrão, sem low-level | 30,6 | 23,2 |
| MindEye1 | `subj01_me1_40sess` | 40 | padrão + low-level | 94,5 | 87,6 |
| MindEye1 | `subj01_me1_40sess_semlowlevel` | 40 | padrão, sem low-level | 94,5 | 87,6 |
| MindEye2 (artigo) | `final_subj01_pretrained_1sess_24bs` | 1 | fine-tune completo, baixado | 93,9 | 77,6 |
| MindEye2 (artigo) | `final_subj01_pretrained_40sess_24bs` | 40 | fine-tune completo, baixado | 100 | 99,9 |

Definidas e ainda não rodadas (`EXPERIMENTOS`): 5 classes semânticas, pessoa + animal, 1.000 imagens × 3
repetições nos quatro modos de agregação, e 3.000 × 1, 1.500 × 2, 1.000 × 3 imagens × repetições.

## Estrutura

| Pasta | |
|---|---|
| `setup/` | preparar a máquina |
| `scripts/` | rodar os modelos |
| `src/` | MindEye2, FRR, `treina.py`, `experimentos.py`, o site (`make_site.py`) e a página do benchmark |
| `src/mindeye_ridge/` | biblioteca: dados, FRR, métricas, semântica, dataset controlado, agregação, resultados |
| `mindeye1/` | MindEye1 adaptado |
| `notebooks/` | `treino.ipynb`, `classificacao_semantica.ipynb`, originais do MindEye2 |
| `results/` | tabelas, site e página (no git); tensores e logs (fora) |
| `train_logs/` | checkpoints (fora do git) |
| `tests/` | testes |
| `docs/` | [NOVO_COMPUTADOR](docs/NOVO_COMPUTADOR.md), [SETUP](docs/SETUP.md), [MINDEYE1](docs/MINDEYE1.md), [EXPERIMENTO](docs/EXPERIMENTO.md), [ESTAGIO](docs/ESTAGIO.md), [REEXECUTAR](docs/REEXECUTAR.md) |

## Licença

MIT. `src/generative_models/` e `src/autoencoder/` mantêm as licenças próprias.
