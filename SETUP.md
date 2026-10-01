# MindEye2 — preparar uma máquina do zero

Este é o **único arquivo** que você precisa ler para deixar um computador pronto
para rodar o MindEye2. Ele lista todas as bibliotecas, todos os arquivos externos
e a ordem em que tudo é instalado.

Índice: [1. Requisitos](#1-requisitos) · [2. Instalação](#2-instalação-em-2-comandos) ·
[3. Bibliotecas](#3-todas-as-bibliotecas) · [4. Arquivos externos](#4-todos-os-arquivos-externos) ·
[5. Onde cada coisa fica](#5-onde-cada-coisa-fica) · [6. Rodar o pipeline](#6-rodar-o-pipeline-completo) ·
[7. Backup e restauração](#7-backup-e-restauração) · [8. Erros conhecidos](#8-erros-conhecidos)

---

## 1. Requisitos

| Item | Mínimo | Como conferir |
|---|---|---|
| GPU NVIDIA | 16 GB de VRAM (testado em RTX A4500, 20 GB) | `nvidia-smi` |
| conda **ou** python3.11 | qualquer versão do conda | `conda --version` |
| Disco livre | **~85 GB** (ver a conta abaixo) | `df -h ~` |
| git | qualquer versão | `git --version` |

Python 3.12+ **não funciona**: o `torch==2.1.0` não tem wheel para essa versão.
O `setup_env.sh` cria um ambiente 3.11 isolado, então a versão do sistema não importa.

**A conta dos 85 GB:**

| O quê | Tamanho |
|---|---|
| Dados e checkpoints do HuggingFace (seção 4.1) | 62 GB |
| Modelos baixados sozinhos na 1ª execução (seção 4.2) | ~15 GB |
| Ambiente Python (`~/envs/fmri`) | 6,5 GB |
| Checkpoints que **você** vai treinar (~2–3 GB cada) | 5 GB |

---

## 2. Instalação em 2 comandos

```bash
git clone https://github.com/PR-Alberti/mindeye2-ridge.git
cd mindeye2-ridge && ./bootstrap.sh --all --ckpts
```

É isso. O `bootstrap.sh` cria o ambiente, baixa os 62 GB de dados e traz os
checkpoints já treinados. Leva algumas horas, quase tudo em download.

**Antes de começar**, se quiser só ver o que falta na máquina sem baixar nada:

```bash
./bootstrap.sh --check
```

Ele imprime, seção por seção, o que já existe e o que falta — incluindo um
diagnóstico de GPU que distingue "driver não responde" de "a placa não está no
barramento PCI" (os dois se parecem, e a diferença muda completamente o que fazer).

**Variações:**

| Comando | O que faz |
|---|---|
| `./bootstrap.sh` | ambiente + dados de treino (~30 GB) |
| `./bootstrap.sh --all` | ambiente + todos os dados (~62 GB) |
| `./bootstrap.sh --all --ckpts` | + os modelos já treinados, para rodar sem treinar |
| `./bootstrap.sh --check` | só diagnostica, não baixa nada |
| `--env-path X` / `--data-path Y` | muda onde o ambiente e os dados ficam |

Toda etapa é idempotente: pode interromper com Ctrl+C e rodar de novo que ele
continua de onde parou.

### Se preferir passo a passo

O `bootstrap.sh` só encadeia estes dois, que continuam funcionando sozinhos:

```bash
./setup_env.sh --with-extras       # ambiente Python (~15 min)
./download_data.py --stage finetune recon enhanced evals paper --subj 1 --num-sessions 1
```

O `setup_env.sh` termina imprimindo o que encontrou — confira que a GPU aparece:

```
torch 2.1.0+cu121 | CUDA disponivel: True | NVIDIA RTX A4500
modulos do repo (sgm, mindeye_ridge): OK
```

Se sair `CUDA disponivel: False`, o problema é a GPU, não o ambiente — veja a
seção 7.

Outras opções do `setup_env.sh`: `--path /outro/lugar`, `--venv` (usa
`python3.11 -m venv` em vez de conda). Do `download_data.py`: `--dry-run` para ver
o tamanho antes, e `HF_TOKEN` no ambiente para ganhar banda (opcional; o
repositório é público).

---

## 3. Todas as bibliotecas

Tudo isto é instalado pelo `./setup_env.sh`. A tabela existe para você saber o que
está lá dentro e conseguir remontar na mão, se precisar.

### 3.1 Stack principal

```bash
pip install "numpy<2" \
    torch==2.1.0 torchvision==0.16.0 xformers==0.0.22.post7 \
    matplotlib==3.8.2 tqdm scikit-image==0.22.0 pandas==2.2.0 einops ftfy regex \
    accelerate==0.24.1 webdataset==0.2.73 kornia==0.7.1 h5py==3.10.0 \
    open_clip_torch==2.24.0 transformers==4.37.2 torchmetrics==1.3.0.post0 \
    diffusers==0.23.0 omegaconf==2.3.0 pytorch-lightning==2.0.1 wandb \
    jupyter ipykernel
```

| Biblioteca | Versão | Para quê |
|---|---|---|
| `torch`, `torchvision` | 2.1.0 / 0.16.0 | base de tudo |
| `xformers` | 0.0.22.post7 | atenção eficiente (economiza VRAM) |
| `numpy` | `<2` | o torch 2.1.0 foi compilado contra a série 1.x |
| `open_clip_torch` | 2.24.0 | embedder CLIP ViT-bigG-14 |
| `transformers` | 4.37.2 | GIT (legendas) e CLIP da HuggingFace |
| `diffusers` | 0.23.0 | schedulers de difusão |
| `pytorch-lightning`, `omegaconf` | 2.0.1 / 2.3.0 | exigidos pelo `sgm` (generative_models) |
| `webdataset` | 0.2.73 | lê os `.tar` com os metadados dos trials |
| `h5py` | 3.10.0 | lê os `.hdf5` de betas e imagens |
| `kornia` | 0.7.1 | augmentations |
| `accelerate` | 0.24.1 | treino multi-GPU / fp16 |
| `torchmetrics`, `scikit-image` | 1.3.0.post0 / 0.22.0 | métricas |
| `wandb` | qualquer | logging opcional (usamos `--no-wandb_log`) |

### 3.2 Pacotes que vêm de repositório git

```bash
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install dalle2-pytorch
pip install torch==2.1.0 torchvision==0.16.0 "numpy<2"   # dalle2 sobe o torch; reafirme os pins
```

O `mindeye_ridge/models.py` importa os dois no topo do módulo — são obrigatórios, não opcionais.

### 3.3 Extras (só para `final_evaluations.py`)

```bash
pip install sentence-transformers==2.5.1 evaluate==0.4.1 nltk==3.8.1 \
    rouge_score==0.1.2 "datasets==2.16.1" umap-learn
```

Instalados por `./setup_env.sh --with-extras`. Dois detalhes:

- `datasets` fica em 2.x porque a série 5.x exige `huggingface_hub>=0.25`, que quebra o `diffusers` 0.23.0.
- o `setup.sh` original (agora em `legacy/`) pede `umap==0.1.1`, que não existe mais no PyPI; `umap-learn` fornece o mesmo módulo `umap`.

### 3.4 Pins de compatibilidade (aplicados por último)

```bash
pip install "huggingface_hub==0.20.3" "setuptools<81" deepspeed==0.13.1
```

| Pin | Por quê |
|---|---|
| `huggingface_hub==0.20.3` | `diffusers==0.23.0` importa `cached_download`, removido na 0.26 |
| `setuptools<81` | o CLIP da OpenAI faz `from pkg_resources import packaging`, removido no setuptools 81 |
| `deepspeed==0.13.1` | os checkpoints publicados têm objetos do deepspeed no pickle; sem ele o `torch.load` falha |

---

## 4. Todos os arquivos externos

### 4.1 Baixados pelo `download_data.py` (62 GB)

Vêm do dataset [`pscotti/mindeyev2`](https://huggingface.co/datasets/pscotti/mindeyev2)
e vão para `~/mindeyev2` (mude com `--data-path`). A coluna **etapa** é o valor de
`--stage` que traz o arquivo.

| Arquivo | Tamanho | Etapa | Para quê |
|---|---|---|---|
| `coco_images_224_float16.hdf5` | 20,5 GB | train, finetune | as 73k imagens do NSD que os sujeitos viram |
| `betas_all_subj01_fp32_renorm.hdf5` | 1,8 GB | todas com sujeito | ativações de fMRI do sujeito 1 |
| `wds/subj01/train/0.tar` | 8,4 MB | todas com sujeito | metadados dos trials de treino (1 sessão) |
| `wds/subj01/test/0.tar` | 33,8 MB | todas com sujeito | metadados do test set |
| `wds/subj01/new_test/0.tar` | 36,6 MB | todas com sujeito | test set alternativo (`--new_test`) |
| `train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth` | 2,7 GB | finetune | ponto de partida: pré-treino nos outros 7 sujeitos |
| `unclip6_epoch0_step110000.ckpt` | 16,7 GB | recon | SDXL unCLIP: gera imagem a partir do embedding CLIP |
| `bigG_to_L_epoch8.pth` | 6,8 MB | recon | projeta embeddings bigG → L |
| `sd_image_var_autoenc.pth` | 319,2 MB | recon, blurry | autoencoder (VAE) |
| `evals/all_images.pt` | 574,2 MB | recon, evals | as 1000 imagens de teste, como tensor |
| `zavychromaxl_v30.safetensors` | 6,5 GB | enhanced | SDXL usado no refinamento das reconstruções |
| `gnet_multisubject.pt` | 1,2 GB | evals | GNet, para a métrica de correlação cerebral |
| `evals/all_captions.pt` | 794 KB | evals | legendas COCO de referência |
| `evals/all_git_generated_captions.pt` | 53,8 KB | evals | legendas geradas pelo GIT, de referência |
| `brain_region_masks.hdf5` | 822 KB | evals | máscaras das regiões cerebrais |
| `train_logs/final_subj01_pretrained_1sess_24bs/last.pth` | 8,3 GB | paper | modelo publicado no artigo (subj01, 1 sessão) |
| `evals/final_subj01_.../..._all_enhancedrecons.pt` | 2,9 GB | paper | reconstruções publicadas, para comparação |

**Etapas disponíveis** (combináveis: `--stage recon evals`):

| `--stage` | Serve para | Tamanho |
|---|---|---|
| `train` | treinar do zero | ~24 GB |
| `finetune` | fine-tune a partir do ckpt multi-sujeito (o que usamos) | ~27 GB |
| `recon` | `recon_inference.py` | ~18 GB |
| `enhanced` | `enhanced_recon_inference.py` | ~7 GB |
| `evals` | `final_evaluations.py` | ~2 GB |
| `paper` | comparar com o modelo publicado | ~11 GB |
| `paper40` | comparar com o modelo publicado de 40 sessões (benchmark) | ~12 GB |
| `blurry` | **só** se usar `--blurry_recon` | ~17 GB |

A etapa `blurry` traz `convnext_xlarge_alpha0.75_fullckpt.pth` (6,6 GB) e o ponto de
partida `final_multisubject_subj01` (10,3 GB). O braço padrão roda com
`--no-blurry_recon` e não precisa dela; o braço 4096 + blurry e o benchmark precisam.

Outras opções: `--subj 1 2 5 7` (vários sujeitos), `--num-sessions N` (só os N
primeiros tars de treino), `--dry-run` (lista sem baixar).

### 4.2 Baixados sozinhos na primeira execução (~15 GB)

Não precisa fazer nada — os scripts buscam e guardam em cache. Só reserve o disco.
O cache fica em `$HF_HOME`, que os scripts apontam para `~/mindeyev2/.cache`.

| Modelo | Tamanho | Quem baixa | Para quê |
|---|---|---|---|
| `ViT-bigG-14` / `laion2b_s39b_b160k` (open_clip) | ~10 GB | treino e recon | o embedder CLIP principal do MindEye2 |
| `microsoft/git-large-coco` | ~1,6 GB | `recon_inference.py` | gera legendas das reconstruções |
| `openai/clip-vit-large-patch14` | ~1,7 GB | enhanced e evals | métrica CLIP e refinamento |
| `CLIP ViT-L/14` (OpenAI, via `clip.load`) | ~900 MB | `final_evaluations.py` | métrica CLIP |
| `openai/clip-vit-base-patch32` | ~600 MB | `final_evaluations.py` | métrica de legendas |
| AlexNet, Inception V3, EfficientNet-B1 (torchvision) + SwAV ResNet50 (torch.hub) | ~500 MB | `final_evaluations.py` | métricas de baixo e alto nível |

---

## 5. Onde cada coisa fica

```
~/mindeye2-ridge/                 ← este repositório (código)
├── README.md                    o que é e como rodar
├── SETUP.md                     este arquivo
├── EXPERIMENTO.md               o que foi feito e por quê
├── BENCHMARK.md                 tabelas do benchmark (gerado)
├── README-original.md           README do MindEye2, como referência
├── bootstrap.sh                 prepara a máquina inteira (chama os dois abaixo)
├── setup_env.sh                 cria o ambiente Python
├── download_data.py             baixa os arquivos da seção 4.1
├── restore_ckpt.py              conserta .pth que o Drive descompactou (seção 7)
├── scripts/
│   ├── common.sh                caminhos e ambiente, lidos por todos os run_*.sh
│   ├── run_ridgeonly_prior.sh   treino (fine-tune só da camada ridge)
│   ├── run_recon.sh             reconstruções (recon + refinamento)
│   ├── run_evals.sh             métricas finais
│   ├── run_frr.sh               baseline linear FRR
│   └── run_benchmark.sh         roda tudo o que o benchmark precisa (seção 6)
├── src/                         pontos de entrada (rodam de qualquer diretório)
│   ├── train_ridgeonly.py       treino
│   ├── recon_inference.py       gera as reconstruções
│   ├── enhanced_recon_inference.py   refina as reconstruções
│   ├── final_evaluations.py     calcula as métricas
│   ├── verify_retrieval.py      só o retrieval, sem difusão
│   ├── run_frr.py               FRR: voxels → embedding CLIP
│   ├── make_benchmark.py        monta benchmark/index.html e BENCHMARK.md
│   ├── make_comparison.py       figura imagem vista × reconstruções
│   ├── mindeye_ridge/           biblioteca (utils, models, modeling_git, paths, nsd_data,
│   │                            clip_targets, frr, embedding_metrics)
│   ├── report/                  o código da página e do BENCHMARK.md (dados, tabelas, textos,
│   │                            markdown e, em assets/, o CSS e o JavaScript)
│   ├── generative_models/       código da Stability AI, como está
│   └── autoencoder/             ConvNeXt do MindEye2
├── results/
│   ├── tables/                  CSVs e JSONs com as métricas finais        (no git)
│   ├── metrics/  figs/          curvas e figuras de corridas antigas       (no git)
│   └── evals/<modelo>/          tensores de reconstrução e de embedding    (fora do git)
├── benchmark/index.html         benchmark completo com galeria (gerado, abre offline)
├── tests/                       testes (python -m unittest discover -s tests)
├── notebooks/                   notebooks do MindEye2 original, de referência
├── legacy/                      o que já não é usado, com um README dizendo o que era
├── tools/                       utilitários avulsos
├── train_logs/<modelo>/         saída: last.pth + metrics.csv + train.log  (fora do git)
└── logs/                        logs de execução                            (fora do git)

~/mindeyev2/                 ← dados externos (seção 4.1), 62 GB
├── coco_images_224_float16.hdf5
├── betas_all_subj01_fp32_renorm.hdf5
├── wds/subj01/{train,test,new_test}/0.tar
├── train_logs/                  checkpoints prontos (multi-sujeito e do artigo)
├── evals/                       imagens e legendas de referência
└── .cache/                      modelos da seção 4.2

~/envs/fmri/                 ← ambiente Python, 6,5 GB
```

`train_logs/` e `logs/` estão no `.gitignore`: são checkpoints e logs, não código.

Os scripts em `scripts/` leem os caminhos de `scripts/common.sh`, que usa `~/envs/fmri` e
`~/mindeyev2` por padrão. Em outro lugar, exporte antes de rodar:

```bash
export MINDEYE_ENV=/home/SEU_USUARIO/envs/fmri      # ambiente criado no passo 2
export MINDEYE_DATA=/home/SEU_USUARIO/mindeyev2     # dados baixados no passo 3
```

---

## 6. Rodar o pipeline completo

Os comandos rodam de qualquer diretório: o `scripts/common.sh` entra em `src/` sozinho.

```bash
# 1. treino — fine-tune só da camada ridge, 1 sessão, com diffusion prior
scripts/run_ridgeonly_prior.sh
# saída: train_logs/subj01_ridgeonly_1sess_prior/{last.pth,metrics.csv}
# ~35 s por época numa RTX A4500; 150 épocas por padrão

# 2. reconstruções (recon + refinamento)
scripts/run_recon.sh subj01_ridgeonly_1sess_prior 1024 noblurry
# saída: results/evals/subj01_ridgeonly_1sess_prior/*_all_recons.pt e *_all_enhancedrecons.pt

# 3. métricas
scripts/run_evals.sh subj01_ridgeonly_1sess_prior enhanced
# saída: results/tables/subj01_ridgeonly_1sess_prior_all_enhancedrecons.csv
```

**Variantes úteis:**

```bash
BATCH_SIZE=12 scripts/run_ridgeonly_prior.sh          # se der out of memory
NUM_EPOCHS=10 scripts/run_ridgeonly_prior.sh          # teste rápido
PRIOR=0 scripts/run_ridgeonly_prior.sh                # sem o diffusion prior: ~45% mais rápido, mas bem pior (BENCHMARK.md)
scripts/run_recon.sh <modelo> 1024 noblurry --skip-enhanced   # só a recon base
scripts/run_evals.sh <modelo> base                    # métricas da recon sem refinamento
```

**As métricas que interessam** ficam em `train_logs/<modelo>/metrics.csv`:
`test/test_fwd_pct_correct` e `test/test_bwd_pct_correct` são retrieval top-1 entre
300 candidatos — acaso = 0,33%. **fwd**: dada cada imagem, achar o seu cérebro entre as 300
previsões; **bwd**: dado cada cérebro, achar a sua imagem. É o que o código calcula, e é o
oposto do que o comentário do `final_evaluations.py` (`fwd: brain, clip`) e o artigo
(Image Retrieval = cérebro → imagem) dizem; o `BENCHMARK.md` rotula as colunas por essa direção
(Img→cér, Cér→img).

**Alavancas de VRAM**, se a GPU for menor:

| Alavanca | Efeito |
|---|---|
| `BATCH_SIZE=12` | menos memória por passo |
| `--embedder_fp16` | guarda o ViT-bigG em fp16, libera ~3,6 GB (já ligado nos `run_*.sh`) |
| `--no-use_prior` | remove o diffusion prior, mas a ridge treinada sem a loss do prior perde ~15 pontos de retrieval e reconstrói bem pior (BENCHMARK.md, seção de ruído) |
| `--no-blurry_recon` | remove o submódulo de baixo nível (já é o padrão nos nossos scripts) |
| `GNET_BATCH_SIZE=20` | lote do GNet nas métricas (padrão já é 20; o original usava 100 e estourava 20 GB) |

### Benchmark de todos os modelos

```bash
scripts/run_benchmark.sh              # ~40 h numa A4500; uma linha por etapa em logs/benchmark.log
python src/make_benchmark.py             # só remonta a página com o que já existe
```

Treina o que falta (4096 + blurry e 40 sessões), reconstrói e avalia os quatro ridge-only
e os dois modelos do artigo (1 e 40 sessões; `download_data.py --stage paper paper40 blurry`),
roda o baseline linear FRR (próxima seção), e monta `benchmark/index.html` e `BENCHMARK.md`.
Pode ser interrompido e rodado de novo:

| Onde | Como retoma |
|---|---|
| treino | `RESUME=1 scripts/run_ridgeonly_prior.sh` continua do `last.pth` (época, ridge, otimizador, scheduler); `CKPT_INTERVAL=1` grava a cada época |
| reconstrução | `recon_inference.py` grava um parcial a cada 100 imagens, com o estado dos geradores aleatórios |
| refinamento | idem, já existia no `enhanced_recon_inference.py` |
| métricas e etapas prontas | puladas; `FORCE=1 scripts/run_evals.sh ...` recalcula |

Opções novas que ele usa, úteis também sozinhas:

| Opção | Para quê |
|---|---|
| `FROZEN_CKPT=<ckpt de partida> scripts/run_recon.sh ...` | backbone e prior vêm do ckpt de partida, só a ridge do modelo treinado. Necessário no treinado sem prior (não tem prior no ckpt) e recomendado nos treinados com `--frozen_fp16` (guardaram os congelados arredondados, inclusive o agendamento de ruído do prior) |
| `BACKBONE_FP16=1` | backbone em fp16 na reconstrução; ligado sozinho com `hidden_dim=4096`, que em fp32 estoura 20 GB junto com o unCLIP. Mesmo resultado: é o arredondamento que o autocast já fazia |
| `scripts/run_evals.sh <modelo> published` | avalia as reconstruções publicadas pelos autores (`published-base` para as sem refinamento, onde existirem) |

### FRR: baseline linear (plano de estágio)

Regressão ridge fracionária (Rokem & Kay, 2020) dos 15.724 voxels direto para o embedding CLIP
ViT-bigG/14 achatado (256 × 1.664), seguindo Doerig et al. (2025): validação cruzada de 5
dobras agrupada por imagem e uma fração escolhida para cada uma das 425.984 dimensões do alvo.
Sem rede, sem prior e sem reconstrução; mesmo sujeito, mesmas sessões de treino e mesmo teste
(1.000 imagens) do ridge-only, avaliado com o mesmo retrieval (`embedding_metrics.py`).

```bash
scripts/run_frr.sh                                   # 1 sessão   -> results/evals/subj01_frr_1sess   (~1 min)
NUM_SESSIONS=40 scripts/run_frr.sh                   # 40 sessões -> results/evals/subj01_frr_40sess (~30 min)
MODEL_NAME=subj01_frr_1sess_ext EXTRA="--grid extended" scripts/run_frr.sh    # grade estendida (sensibilidade)
python -m unittest discover -s tests                            # testes
```

Grava `results/evals/<modelo>/<modelo>_{all_clipvoxels.pt,retrieval.json,frr.json}` e uma cópia do
`frr.json` em `results/tables/` (é a que o `make_benchmark.py` lê). A grade de frações é a de Doerig
(0,05 a 1), como o plano especifica; `--grid extended` acrescenta dez menores (`FRACS_ESTENDIDA` em
`frr.py`) como análise de sensibilidade: o ótimo do erro quadrático cai abaixo da grade de Doerig,
mas o retrieval não o acompanha (veja o `BENCHMARK.md`).

| Detalhe | |
|---|---|
| Alvos CLIP de treino | calculados na primeira execução (8 GB e ~3 min para as 9.000 imagens) e guardados em `<dados>/clip_targets/subj01/`; o alvo de teste é recalculado a cada corrida, como nas outras avaliações |
| Retomada | cada dobra da validação cruzada vai para `train_logs/<modelo>/frr_cv/`; uma corrida interrompida recomeça da primeira que falta. `--cv_dir` aponta para as dobras de outra corrida (por exemplo, `--global_fraction` troca só a escolha da fração) |
| Memória | GPU ~4 GB; RAM ~25 GB de pico nas 40 sessões (diagonalização de 15.724 × 15.724 em fp64 na CPU), ~3 GB com 1 sessão |

### Treino original do artigo (não o ridge-only)

Os notebooks de `notebooks/` são os do MindEye2 original e importam `utils` e `models` do layout antigo (tudo em `src/`). Para rodar esta receita, converta o notebook para dentro de `src/` e troque os imports por `from mindeye_ridge import utils` e `from mindeye_ridge.models import ...`; ou use o `src/train_ridgeonly.py`, que já faz isso e treina tudo quando chamado com `--no-ridge_only` (veja `legacy/run_fulltune_probe.sh`).

```bash
jupyter nbconvert notebooks/Train.ipynb --to python --output-dir src
GLOBAL_BATCH_SIZE=24 ~/envs/fmri/bin/python Train.py \
    --data_path=$HOME/mindeyev2 --cache_dir=$HOME/mindeyev2/.cache \
    --model_name=meu_teste --subj=1 --num_sessions=1 --batch_size=24 \
    --hidden_dim=1024 --n_blocks=4 --clip_scale=1. \
    --use_prior --prior_scale=30 --no-blurry_recon \
    --max_lr=3e-4 --num_epochs=150 --no-wandb_log
```

⚠️ Passe `--batch_size` **explicitamente**. O `Train.ipynb` calcula `batch_size` a
partir de `GLOBAL_BATCH_SIZE` no início, mas o argparse depois sobrescreve tudo com o
default de `--batch_size` (16) — por isso a variável de ambiente sozinha não tem efeito.
O `accel.slurm` do repositório original passa `--batch_size` explicitamente pelo mesmo motivo.

---

## 7. Backup e restauração

Esta é a parte que a gente aprendeu na marra: **em máquina que é resetada, só
sobrevive o que está no GitHub.**

### O que já está seguro (não precisa backup)

Tudo que é código está no repositório: `src/`, `scripts/`, `tests/`, `bootstrap.sh`,
`setup_env.sh`, `download_data.py`, este guia e as tabelas de métricas em
`results/tables/`. Uma máquina nova recupera tudo isso com `git clone`.

Os 62 GB de dados externos também não precisam de backup — o `bootstrap.sh` os
rebaixa do HuggingFace.

### O que se perde se você não guardar

| O quê | Tamanho | Dá para regerar? |
|---|---|---|
| `train_logs/<modelo>/last.pth` | 2–3 GB cada | sim, retreinando (~90 min) |
| `train_logs/<modelo>/metrics.csv` e `train.log` | KB | não, sem retreinar |
| `results/evals/<modelo>/*.pt` (reconstruções) | ~2,3 GB | sim, ~3 h de inferência |
| `<dados>/clip_targets/` (cache dos alvos CLIP do FRR) | 8 GB | sim, ~3 min |
| `train_logs/<modelo>/frr_cv/` (dobras da validação cruzada do FRR) | 70 MB cada | sim, ~20 min nas 40 sessões |

Os `.pth` **e as reconstruções** estão publicados no release
[`checkpoints-v1`](https://github.com/PR-Alberti/mindeye2-ridge/releases), e o
`./bootstrap.sh --ckpts` traz os dois de volta automaticamente — o que poupa
~3 h de GPU a cada máquina nova. Para publicar um modelo novo, veja abaixo.

### Cuidado ao usar Google Drive como backup

Um `.pth` é um zip, e o Drive às vezes o **descompacta**: você baixa de volta uma
pasta com `data.pkl`, `data/`, `version` e `byteorder` em vez de um arquivo. O
torch não carrega isso, e **re-zipar na mão não resolve** — ele exige que cada
registro comece alinhado em 64 bytes, coisa que o `zipfile` do Python não faz.

Para consertar:

```bash
./restore_ckpt.py pasta_do_checkpoint/ -o train_logs/meu_modelo/last.pth
./restore_ckpt.py train_logs/meu_modelo/last.pth    # só verifica se está legível
```

Ele lê as storages e deixa o próprio torch regravar. Ao final confirma com um
`torch.load` de verdade — só isso prova que o arquivo está bom.

Para evitar o problema, prefira compactar antes de subir:
`tar czf train_logs.tar.gz train_logs/`.

### Publicar um modelo treinado no GitHub

Assets de release têm limite de 2 GB por arquivo, então checkpoints maiores vão
partidos (o `bootstrap.sh` junta de volta sozinho):

```bash
M=subj01_ridgeonly_1sess_prior
split -b 1900M -d -a 2 --additional-suffix="" \
      train_logs/$M/last.pth "$M.last.pth.part-"
gh release upload checkpoints-v1 "$M.last.pth.part-"*  # ou pela interface web
```

Para arquivos abaixo de 2 GB, suba direto como `<modelo>.last.pth`.

As reconstruções vão num tar (o `bootstrap.sh` extrai sozinho para
`results/evals/<modelo>/`) — e compactar também evita que o Drive as descompacte:

```bash
cd results/evals/$M && tar czf ~/$M_recons.tar.gz *.pt
gh release upload checkpoints-v1 ~/${M}_recons.tar.gz
```

---

## 8. Erros conhecidos

| Erro | Causa | Correção |
|---|---|---|
| `cannot import name 'cached_download'` | `diffusers==0.23.0` usa função removida na `huggingface_hub` 0.26 | `pip install huggingface_hub==0.20.3` |
| `No module named 'pkg_resources'` | o CLIP da OpenAI importa `pkg_resources`, removido do setuptools 81 | `pip install "setuptools<81"` |
| `No module named 'deepspeed'` ao carregar um ckpt | os checkpoints publicados têm objetos do deepspeed no pickle | `pip install deepspeed==0.13.1` |
| Erros estranhos de dtype no numpy | `torch==2.1.0` foi compilado contra numpy 1.x | `pip install "numpy<2"` |
| `CUDA out of memory` no treino, com "reserved but unallocated" grande | fragmentação do allocator; o batch 16 fica sem margem quando o desktop também roda na GPU | já corrigido: os `run_ridgeonly*.sh` exportam `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `CUDA out of memory` no treino, mesmo com o acima | batch grande demais para a placa | `BATCH_SIZE=12 scripts/run_ridgeonly_prior.sh` |
| Treino cabia antes e agora não cabe | o monitor foi ligado na placa dedicada; Xorg/gnome-shell/VS Code passam a ocupar ~350 MiB de VRAM | ligue o monitor na saída de vídeo da placa-mãe (iGPU), devolvendo a VRAM inteira ao treino |
| `CUDA out of memory` nas métricas | GNet prevendo 1000 recons de uma vez | já corrigido em `models.py`; ajuste com `GNET_BATCH_SIZE=10` |
| `--stage` não baixa nada e diz "ja esta no lugar" | arquivos completos já existem | normal; use `--dry-run` para conferir |
| Python 3.12 recusa instalar torch | não existe wheel do `torch==2.1.0` para 3.12 | use o ambiente 3.11 do `setup_env.sh` |

---

## Referências

- Artigo (ICML 2024): https://arxiv.org/abs/2403.11207
- Repositório original: https://github.com/MedARC-AI/MindEyeV2
- Dataset no HuggingFace: https://huggingface.co/datasets/pscotti/mindeyev2
- Termos do Natural Scenes Dataset: https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions
