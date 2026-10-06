# benchmark-modelos

Benchmark de modelos que decodificam a imagem vista a partir do fMRI, no sujeito 1 do
[Natural Scenes Dataset](https://naturalscenesdataset.org/) (NSD), com o mesmo protocolo de
avaliação para todos. Três modelos:

| Modelo | O que é | Código | Situação |
|---|---|---|---|
| **MindEye2** | MLP + diffusion prior → CLIP ViT-bigG, imagem por SDXL unCLIP ([Scotti et al., 2024](https://arxiv.org/abs/2403.11207)). Aqui: fine-tune só da camada ridge (1 e 40 sessões) e os modelos publicados | `src/` | treinado, reconstruído e avaliado — [BENCHMARK.md](BENCHMARK.md) |
| **FRR** | regressão ridge fracionária, linear, dos voxels direto para o embedding CLIP ([Rokem & Kay, 2020](https://doi.org/10.1093/gigascience/giaa133); como em Doerig et al., 2025) | `src/run_frr.py`, `src/mindeye_ridge/frr.py` | rodado com 1 e 40 sessões — [BENCHMARK.md](BENCHMARK.md) |
| **MindEye1** | MLP + diffusion prior → CLIP ViT-L/14, imagem por Versatile Diffusion ([Scotti et al., 2023](https://arxiv.org/abs/2305.18274)). Aqui: treinado do zero com 1 e 40 sessões, nos mesmos dados dos outros | `mindeye1/` | adaptado para 20 GB; treinado, reconstruído e avaliado com 1 e 40 sessões — [BENCHMARK.md](BENCHMARK.md), [MINDEYE1.md](docs/MINDEYE1.md) |

Este repositório junta o trabalho de [`mindeye2-ridge`](https://github.com/PR-Alberti/mindeye2-ridge)
(que guarda o histórico completo do MindEye2 e do FRR) com o MindEye1. O que falta do plano de
estágio está em [ESTAGIO.md](docs/ESTAGIO.md).

## Requisitos

| | |
|---|---|
| GPU NVIDIA | 20 GB de VRAM (testado numa RTX A4500). O MindEye2 ridge-only e o FRR cabem em 16 GB; o treino do MindEye1 precisa dos 20 |
| RAM | 30 GB (o FRR de 40 sessões chega a ~25 GB) |
| Disco | ~85 GB para o MindEye2 e o FRR, mais ~70 GB para o MindEye1 completo (~31 GB só para a inferência) |
| Software | conda (ou python3.11) e git. Python 3.12+ não serve: o `torch==2.1.0` não tem wheel |
| Dados | aceitar os [termos do NSD](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) e preencher o [formulário de acesso](https://forms.gle/xue2bCdM9LaFNMeb7) |

## Instalação

Um comando prepara a máquina para treinar e avaliar os três modelos:

```bash
git clone https://github.com/PR-Alberti/benchmark-modelos.git
cd benchmark-modelos
setup/bootstrap.sh --check     # diagnóstico: GPU, ambiente, dados, pesos; não baixa nada
setup/bootstrap.sh             # tudo (~130 GB, horas, quase tudo download)
```

O `bootstrap.sh` cria o ambiente Python (`~/envs/fmri`, o mesmo para os três), baixa os dados do
MindEye2 com as 40 sessões (que o FRR e o MindEye1 também usam), pré-baixa os pesos que os modelos
buscariam na primeira execução (CLIP, Versatile Diffusion, redes das métricas), liga o MindEye1 a
esses pesos e, no fim, importa o código de cada modelo para confirmar que ele carrega. Termina
com o comando de treino de cada um. `--ckpts` traz também os ridge-only já treinados;
`--minimo` prepara só o ridge-only de 1 sessão (~30 GB).

Tudo é idempotente: interrompa e rode de novo que continua. Bibliotecas, cada arquivo baixado,
onde fica e os erros conhecidos estão no [SETUP.md](docs/SETUP.md) (MindEye2 e FRR) e no
[MINDEYE1.md](docs/MINDEYE1.md).

Os caminhos padrão podem ser trocados por variáveis de ambiente:

| Variável | Padrão | O quê |
|---|---|---|
| `MINDEYE_ENV` | `~/envs/fmri` | ambiente Python |
| `MINDEYE_DATA` | `~/mindeyev2` | dados do MindEye2 e cache de pesos do HuggingFace (`.cache/`) |
| `ME1_DATA` | `~/mindeye1` | dados e modelos publicados do MindEye1 |

## Preparar e rodar cada modelo

Todos os scripts rodam de qualquer diretório.

### MindEye2 (ridge-only)

Precisa de: `setup/bootstrap.sh` (ou só `setup/bootstrap.sh --minimo` para o treino de 1 sessão).

```bash
scripts/run_ridgeonly_prior.sh                                   # treino, 1 sessão (~1h50)
scripts/run_recon.sh subj01_ridgeonly_1sess_prior 1024 noblurry  # reconstruções + refinamento
scripts/run_evals.sh subj01_ridgeonly_1sess_prior enhanced       # métricas
```

40 sessões (~8 h de treino; precisa de `setup/download_data.py --stage finetune --subj 1`, sem limitar
as sessões):

```bash
MODEL_NAME=subj01_ridgeonly_40sess_prior NUM_SESSIONS=40 NUM_EPOCHS=20 FROZEN=1 scripts/run_ridgeonly_prior.sh
```

Sem treinar: `setup/bootstrap.sh --ckpts` baixa os checkpoints treinados do release `checkpoints-v1` de
`mindeye2-ridge` (repositório privado: precisa estar logado no GitHub). Os modelos publicados no
artigo vêm de `setup/download_data.py --stage paper paper40`.

### FRR

Precisa de: os dados de treino do MindEye2 e as imagens de teste
(`setup/download_data.py --stage train evals --subj 1`, que já traz as 40 sessões).
Não tem treino por gradiente: a validação cruzada e o ajuste final rodam de uma vez.

```bash
scripts/run_frr.sh                       # 1 sessão  (~1 min na A4500)
NUM_SESSIONS=40 scripts/run_frr.sh       # 40 sessões (~30 min; ~25 GB de RAM)
```

Na primeira execução calcula os embeddings CLIP das imagens de treino (8 GB, ~3 min) e os guarda
em `$MINDEYE_DATA/clip_targets/`. Só produz o embedding, então só o retrieval é comparável com os
outros modelos. Detalhes no [SETUP.md](docs/SETUP.md#frr-baseline-linear-plano-de-estágio).

### MindEye1

Precisa de: `setup/bootstrap.sh`, que já roda o `scripts/me1_setup.sh` e baixa o Versatile Diffusion. No benchmark, o
MindEye1 usa os dados do MindEye2 — as mesmas sessões de treino e o mesmo teste dos outros modelos —
e não precisa dos 39 GB de treino do `webdataset_avg_split`.

```bash
scripts/run_me1_benchmark.sh               # 1 e 40 sessões: treino, reconstrução e métricas (~37 h)
SESSOES=1 scripts/run_me1_benchmark.sh     # só 1 sessão (~5 h)
NUM_SESSIONS=1 scripts/me1_run.sh train    # uma etapa: train | lowlevel | recon | metrics
```

O treino como no artigo (batch 32, AdamW) não cabe em 20 GB; aqui ele usa batch 16 e Adam de
8 bits, o que muda o treino. O protocolo do benchmark, os ajustes e as medições de memória e tempo
estão no [MINDEYE1.md](docs/MINDEYE1.md). Sem `NUM_SESSIONS`, o `me1_run.sh` roda o MindEye1 original
(dados do `webdataset_avg_split`, modelos publicados).

### Treino por dicionário e dataset controlado

O jeito mais simples de treinar é o [`notebooks/treino.ipynb`](notebooks/treino.ipynb): cada
treino é um dicionário com o modelo, os hiperparâmetros e o dataset, e os modelos já treinados do
benchmark estão lá como exemplo (o dicionário `BENCHMARK` de `src/experimentos.py`, que reproduz o
treino de cada um). Os scripts acima continuam valendo; o dicionário só chama o script certo com as
variáveis certas.

Os três modelos também treinam num subconjunto controlado do treino, em vez das N primeiras
sessões. Cada experimento é um dicionário em `src/experimentos.py` (`EXPERIMENTOS`):

```python
"subj01_frr_5classes_300img_1rep": {
    "modelo": "frr",                       # frr | mindeye2 | mindeye1
    "hiper": {"folds": 5, "seed": 42},     # o que faltar fica no padrão do modelo
    "dataset": {"sessoes": 40,             # pool: as primeiras N sessões
                "n_imagens": 300,          # imagens únicas (por classe, com balanceado)
                "repeticoes": 1,           # exibições por imagem: 1, 2 ou 3
                "classes": {"pessoa": ["sup_person"], "animal": ["sup_animal"]},
                "minimo": 10, "maximo_outros": 2, "balanceado": True,
                "agregacao": "media",      # como usar as repetições (abaixo)
                "semente": 0},
},
```

```bash
python src/treina.py --lista                       # os experimentos definidos
python src/treina.py <nome> --so_dataset           # sorteia e resume o dataset, sem treinar
python src/treina.py <nome> --dry_run              # mostra o comando que rodaria
python src/treina.py <nome> [<nome> ...]           # treina, em sequência (o que já terminou é pulado)
```

O sorteio é feito sobre as exibições (`src/mindeye_ridge/dataset_controlado.py`), com semente:
mesmo dicionário, mesmo subconjunto. As classes são listas de colunas da tabela de frações da tela
(`src/mindeye_ridge/semantica.py`): cada categoria COCO (`cat_*`) ou supercategoria (`sup_*`), em %
da tela que o NSD mostrou. A imagem é da classe se ela ocupa pelo menos `minimo`% e as outras
classes ficam abaixo de `maximo_outros`%. A exploração que calibrou esses limiares está em
[`notebooks/classificacao_semantica.ipynb`](notebooks/classificacao_semantica.ipynb). Na primeira
vez, a tabela baixa as anotações COCO 2017 (~250 MB) para `$MINDEYE_DATA/semantica/`.

O `treina.py` grava o manifesto do dataset e a configuração completa (com o commit do código) em
`train_logs/<nome>/` e chama o script de treino do modelo com `DATASET` apontando para o
manifesto. O dicionário é a configuração inteira: variáveis exportadas no shell (`BATCH_SIZE`,
`RESUME`...) não chegam ao treino.

`agregacao` escolhe como o treino usa as repetições de cada imagem
(`src/mindeye_ridge/agregacao.py`). As quatro opções são os esquemas que os próprios modelos do
benchmark usam:

| `agregacao` | Amostra de treino | De onde vem |
|---|---|---|
| `"exibicoes"` | cada trial, separado | MindEye2 e FRR |
| `"media"` | a média dos betas das repetições, uma por imagem | o protocolo de teste de todos |
| `"sorteio"` | uma repetição sorteada por imagem, a cada época | o alto nível do MindEye1 (que faz rodízio) |
| `"combinacao"` | combinação aleatória das repetições, a cada época | o baixo nível do MindEye1 (`voxel_select`) |

`None` (o padrão) é o que cada modelo faz no benchmark: trials separados no FRR e no MindEye2,
rodízio no MindEye1. O FRR só aceita `exibicoes` e `media`, porque ajusta numa passada só. O
teste é sempre a média das 3 repetições. Nos modos por imagem a época tem uma amostra por imagem,
não por trial: para comparar com `exibicoes` no mesmo número de passos, multiplique `num_epochs`
(os experimentos `subj01_me2_1000img_3rep_*` fazem isso).

Com o dicionário `"dataset"` vazio, o padrão, o treino é o dataset completo (as 40 sessões) e
roda exatamente como o benchmark, o que foi conferido treinando com e sem o manifesto. Num
subconjunto, uma época encolhe junto com ele. No MindEye2, a época são 750 × sessões amostras
(os trials nominais, como no original) vezes a fração das exibições mantidas. No MindEye1, ela é
uma passada pelas imagens, cada uma com as repetições empilhadas. Com o mesmo `num_epochs`, um
subconjunto menor treina menos passos. No MindEye1 isso pesa também na comparação entre imagens e
repetições: 1.000 imagens × 3 repetições dão um terço dos passos por época de 3.000 × 1. Para
comparar com o mesmo número de passos, ajuste `num_epochs`. O FRR não tem épocas. Os scripts aceitam a variável também quando chamados direto, com `MODEL_NAME`
obrigatório (os nomes padrão são os dos modelos do benchmark), como
`MODEL_NAME=<nome> DATASET=train_logs/<nome>/dataset.json NUM_SESSIONS=40 scripts/run_frr.sh`.
O teste não muda: são sempre as 1.000 imagens compartilhadas. Um nome de experimento não muda de
configuração: se `train_logs/<nome>` já tem outra, o treino para; e um modelo não retoma com
outro manifesto (`dataset.sha1` no diretório do modelo). O `run_benchmark.sh` e o
`run_me1_benchmark.sh` ignoram `DATASET`: o benchmark é sempre o dataset completo.

## Resultados

[BENCHMARK.md](BENCHMARK.md) traz as tabelas do MindEye2 (quatro ridge-only e os dois modelos do
artigo), do FRR e do MindEye1 (1 e 40 sessões), todas com o mesmo pipeline e as mesmas métricas. `benchmark/index.html` é a
página completa (galeria de reconstruções, curvas de treino, ruído entre sementes; um arquivo
só, abre offline). O MindEye1 não tem refinamento: a mesma reconstrução aparece nas duas
abas e nas duas tabelas, e ele não entra nas métricas de legenda.

```bash
scripts/run_benchmark.sh                 # regenera tudo do MindEye2 e do FRR (~50 h numa A4500)
python src/make_benchmark.py             # só remonta a página com o que já existe
python -m unittest discover -s tests     # testes do FRR, da página, dos caminhos e do dataset controlado
```

A página e o `BENCHMARK.md` versionados já estão prontos. Para remontá-los, o
`make_benchmark.py` precisa dos tensores em `results/evals/` e das curvas em `train_logs/`, que
ficam fora do git: num clone novo, sem eles, a página sai sem a galeria e sem as curvas.

## Onde está o quê

| | |
|---|---|
| `src/` | MindEye2 e FRR: `train_ridgeonly.py`, `recon_inference.py`, `enhanced_recon_inference.py`, `final_evaluations.py`, `verify_retrieval.py`, `run_frr.py`, `make_benchmark.py`, `make_comparison.py`; treino com dataset controlado: `experimentos.py` (os dicionários) e `treina.py` |
| `src/mindeye_ridge/` | biblioteca: `utils`, `models` e `modeling_git` (do MindEye2) e o que foi acrescentado — `paths`, `nsd_data`, `clip_targets`, `frr`, `embedding_metrics`, `semantica` (fração da tela e rótulos), `dataset_controlado` (subconjuntos de treino), `agregacao` (como usar as repetições), `resultados` (o retrieval de um modelo pelo nome) |
| `src/report/` | código da página do benchmark e do `BENCHMARK.md` |
| `src/generative_models/`, `src/autoencoder/` | código de terceiros usado como está (Stability AI; ConvNeXt) |
| `mindeye1/` | MindEye1: `src/` (código original + ajustes marcados `# benchmark-modelos`), `download.py`, `README-original.md` |
| `scripts/` | `run_*.sh` (MindEye2 e FRR), `me1_setup.sh`, `me1_run.sh` e `run_me1_benchmark.sh` (MindEye1); `common.sh` guarda caminhos e ambiente |
| `results/` | `tables/` (métricas finais, no git), `metrics/`, `figs/`, `evals/` (tensores, fora do git) |
| `benchmark/` | a página do benchmark |
| `tests/` | testes |
| `tools/` | `me1_fake_data.py` (dados sintéticos do MindEye1), `inspect_ckpt.py` |
| `train_logs/`, `mindeye1/train_logs/`, `logs/` | checkpoints e logs (fora do git) |
| `setup/` | preparam a máquina: `bootstrap.sh` (tudo de uma vez), `setup_env.sh` (ambiente), `download_data.py` (dados do MindEye2), `restore_ckpt.py` |
| `docs/` | [SETUP.md](docs/SETUP.md) e [MINDEYE1.md](docs/MINDEYE1.md) (instalação e operação), [EXPERIMENTO.md](docs/EXPERIMENTO.md) (o que foi feito e por quê), [ESTAGIO.md](docs/ESTAGIO.md) (plano de estágio), `README-mindeye2-original.md` |
| [BENCHMARK.md](BENCHMARK.md) | resultados (gerado pelo `make_benchmark.py`) |
| `notebooks/` | `treino.ipynb` (como treinar: os modelos do benchmark e experimentos novos como dicionários), `classificacao_semantica.ipynb` (exploração dos rótulos semânticos); em `mindeye2_original/`, os notebooks do MindEye2, só de referência |
| `legacy/` | scripts antigos, só de referência |

## Licença

MIT, como os dois repositórios de origem (MedARC). O código de terceiros em `src/generative_models/`
e `src/autoencoder/` mantém as licenças próprias.
