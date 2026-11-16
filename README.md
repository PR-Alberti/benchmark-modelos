# benchmark-modelos

Decodificação da imagem vista a partir do fMRI (NSD, sujeito 1): MindEye2, FRR e MindEye1, com o
mesmo treino, o mesmo teste (1.000 imagens compartilhadas) e as mesmas métricas.

| Modelo | Código | Saída |
|---|---|---|
| **MindEye2** ridge-only ([Scotti et al., 2024](https://arxiv.org/abs/2403.11207)) | `src/etapas/train_ridgeonly.py` | embedding CLIP ViT-bigG + imagem (SDXL unCLIP) |
| **FRR**, ridge fracionária ([Rokem & Kay, 2020](https://doi.org/10.1093/gigascience/giaa133)) | `src/etapas/run_frr.py` | embedding CLIP ViT-bigG |
| **FRR multi-sujeito**: espaço compartilhado linear, pré-treino nos sujeitos 2–8 ([docs/FRR_MULTISUJEITO.md](docs/FRR_MULTISUJEITO.md)) | `src/etapas/run_frr_ms.py` | embedding CLIP ViT-bigG |
| **MindEye1** ([Scotti et al., 2023](https://arxiv.org/abs/2305.18274)) | `mindeye1/src/Train_MindEye.py` | embedding CLIP ViT-L/14 + imagem (Versatile Diffusion) |

Em andamento, ainda fora do `treina.py` e do site: o **MindEye1 multi-sujeito**, a rede do MindEye1 treinada
com o protocolo do MindEye2 (pré-treino nos outros 7 sujeitos e ajuste no sujeito 1), em
`mindeye1/src/Train_MindEye_multisubject.py` ([docs/MINDEYE1.md](docs/MINDEYE1.md#mindeye1-multi-sujeito)).

Novo no código? Comece pelo [docs/GUIA.md](docs/GUIA.md): o problema em uma página, o glossário, o
pipeline de ponta a ponta e onde fica cada coisa. O índice da documentação está em [docs/README.md](docs/README.md).

Site com início, uso e resultados interativos de todos os modelos (abas Resumo, Tabela, Tempos, Gráficos,
Dispersão, Treino, Galeria, FRR e Fichas): [`results/site/index.html`](results/site/index.html); apresentação do
trabalho em slides: [`results/site/apresentacao.html`](results/site/apresentacao.html). Os dois abrem direto do
disco e são regenerados ao fim de cada treino e avaliação, então estão sempre em dia. O [BENCHMARK.md](BENCHMARK.md)
não: é a foto de 19/10/2026, só com 1 e 40 sessões, e regenerá-lo pede a máquina com `train_logs/` e `results/evals/`.

## Instalação

```bash
git clone https://github.com/PR-Alberti/benchmark-modelos.git && cd benchmark-modelos
setup/bootstrap.sh --check        # diagnóstico, não baixa nada
setup/bootstrap.sh                # tudo
setup/bootstrap.sh --minimo       # só ridge-only e FRR de 1 sessão
setup/bootstrap.sh --ckpts        # + checkpoints ridge-only do release (repositório privado)
~/envs/fmri/bin/python -m unittest discover -s tests
```

O setup instala o código de `src/` no ambiente em modo editável (`pip install -e .`, configurado no
[`pyproject.toml`](pyproject.toml)): `import mindeye_ridge`, `import treina` e `import report` funcionam
de qualquer pasta, e uma mudança no código vale na hora. Num ambiente criado antes disso, rode
`setup/bootstrap.sh` de novo, ou só `~/envs/fmri/bin/pip install --no-deps --no-build-isolation -e .`.

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
python treina.py <nome> --reconstrucao unclip|refino   # FRR e FRR multi-sujeito: depois do treino, reconstrói (padrão: none)
```

No FRR (e no `frr_ms`), `--reconstrucao` passa o embedding previsto pelo próprio modelo direto para o unCLIP
(`unclip`) e, com `refino`, também pelo refinamento SDXL guiado pela legenda que o GIT tira do mesmo
embedding; as métricas saem das duas. Nada vem do MindEye2 treinado: unCLIP, conversor bigG→ViT-L,
GIT e SDXL foram treinados só com imagens e texto. Com o treino já concluído, só a reconstrução roda.

```python
import treina                               # de dentro de src/
treina.roda("meu_modelo", {
    "modelo": "frr",                        # frr | frr_ms | mindeye2 | mindeye1
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
NUM_SESSIONS=1 MODEL_NAME=<nome> scripts/run_frr_ms.sh                # FRR multi-sujeito (sujeitos 2-8: bootstrap.sh --multisujeito)
scripts/run_recon_embeddings.sh <nome> unclip|refino                  # reconstrução a partir do embedding + métricas

# MindEye1
scripts/run_me1_benchmark.sh                                          # 1 e 40 sessões, treino → métricas
SESSOES=1 scripts/run_me1_benchmark.sh                                # só 1 sessão
LOWLEVEL=0 scripts/run_me1_benchmark.sh                               # sem low-level
NUM_SESSIONS=1 scripts/me1_run.sh train|lowlevel|recon|metrics        # uma etapa

# Dataset controlado direto nos scripts
MODEL_NAME=<nome> DATASET=train_logs/<nome>/dataset.json NUM_SESSIONS=40 scripts/run_frr.sh

# Lotes (retomáveis; uma linha por etapa em results/logs/)
scripts/run_reexecutar.sh                                             # os modelos do benchmark, 1 e 40 sessões (repro_<nome>)
scripts/run_sessoes_intermediarias.sh                                 # MindEye2 com 2, 5, 10 e 20 sessões, com e sem low-level
scripts/run_me1_sessoes.sh                                            # MindEye1 com 5, 10 e 20 sessões, com e sem low-level
python src/treina.py subj01_frr_5sess --reconstrucao refino           # FRR numa sessão intermediária, com imagens e métricas

# Site e benchmark
python src/make_site.py                                               # regenera o site e a apresentação (results/site)
python src/make_site.py --exemplos                                    # refaz os exemplos semânticos da apresentação
python src/make_benchmark.py                                          # remonta BENCHMARK.md e a página (precisa de train_logs/ e results/evals/)
scripts/mede_tempos.sh --fonte results/evals/<modelo>                 # estimativa de tempo por etapa, para quem não gravou os próprios
```

O `scripts/run_benchmark.sh` é o lote original, com os nomes sem `repro_`; os resultados dele foram
substituídos pelas reproduções e ele fica só como registro ([docs/REEXECUTAR.md](docs/REEXECUTAR.md)).

### Resultado de um modelo

```bash
python -c "from mindeye_ridge import resultados; print(resultados.retrieval('<nome>'))"
```

| Onde | O quê |
|---|---|
| `train_logs/<nome>/` | checkpoint, logs, `dataset.json`, `experimento.json` (MindEye1: `mindeye1/train_logs/<nome>/`) |
| `results/tables/` | métricas finais |
| `results/evals/<nome>/` | tensores de embedding e reconstrução |
| `results/site/index.html` | o site: todos os modelos, com tabelas, tempos de cada etapa, gráficos, curvas, galeria e fichas |
| `results/site/apresentacao.html` | a apresentação: a pergunta, os modelos ilustrados, os resultados e o controle semântico, com exemplos de imagens por classe e % da tela (`results/site/exemplos/`) |
| `results/site/modelos/<nome>.json` | o registro do modelo no site (o que foi extraído de cada resultado) |
| `train_logs/<nome>/tempos.jsonl` | o tempo de cada etapa depois do treino (o site o copia para o registro do modelo) |
| [BENCHMARK.md](BENCHMARK.md), `results/benchmark/index.html` | tabelas e página do benchmark (foto de 19/10/2026, 1 e 40 sessões) |

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
| `frr_ms` | `k=1024`, `pretreino=[2..8]` (ou `"proprio"`, o controle de posto reduzido), `pesos="sujeito"` (ou `"exibicao"`), `iteracoes=10`, `tolerancia=1e-3`, `validacao=0.1`, `folds=5`, `grid="doerig"`, `fracs=None`, `seed=42`, `chunk=8192` |
| `mindeye2` | `hidden_dim=1024` (ou 4096), `num_epochs=150`, `batch_size=None`, `max_lr=3e-4`, `prior=True`, `blurry=False`, `frozen_fp16=None`, `seed=42`, `ckpt_interval=None` |
| `mindeye1` | `num_epochs=240`, `max_lr=3e-4`, `seed=42`, `mixup_pct=0.33`, `batch_size=None` (16), `paper=False`, `save_every=1` |

| Restrição | |
|---|---|
| `mindeye2`, `mindeye1` | `num_epochs >= 2` |
| `mindeye2` | `blurry=True` só com `hidden_dim=4096` |
| `mindeye1` | imagens únicas >= `batch_size` |
| `frr` | `agregacao` só `exibicoes` ou `media` |
| `frr_ms` | `agregacao` só `exibicoes`; os sujeitos 2–8 baixados (`setup/bootstrap.sh --multisujeito`) |
| todos | sujeito 1; nome já treinado é recusado |

## Combinações testadas

Treinadas e avaliadas, todas com o dataset completo das N primeiras sessões e a semente 42. Retrieval
top-1 entre 300, em %: imagem → cérebro / cérebro → imagem (média de 30 sorteios). Os números são os do
site em 14/11/2026; as linhas de 1 e 40 sessões vêm das reproduções `repro_<nome>`, que o site mostra com o
nome sem prefixo. Os dicionários estão em `src/experimentos.py` (`BENCHMARK`, `EXPERIMENTOS`, `REPRODUCOES`).

| Modelo | Nome | 1 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| FRR | `subj01_frr_<N>sess` | 55,0 / 3,1 | 88,6 / 20,4 | 92,7 / 29,7 | 94,3 / 53,3 | 96,4 / 37,7 |
| FRR multi-sujeito (C) | `subj01_frr_multisujeito_<N>sess` | 62,9 / 12,4 | — | — | 94,1 / 50,2 | 96,1 / 38,7 |
| FRR posto reduzido (B, controle) | `subj01_frr_postoreduzido_<N>sess` | 41,1 / 9,6 | — | — | 94,5 / 54,7 | 97,0 / 43,9 |
| MindEye2 (4096 + low-level) | `subj01_ridgeonly_<N>sess_4096blurry` | 95,5 / 92,2 | 99,6 / 98,4 | 99,6 / 98,6 | 99,6 / 99,0 | 99,9 / 99,6 |
| MindEye2 sem low-level (1024 + prior) | `subj01_ridgeonly_<N>sess_prior` | 92,9 / 88,1 | 99,2 / 98,0 | 99,4 / 98,7 | 99,4 / 98,9 | 99,9 / 99,2 |
| MindEye1 | `subj01_me1_<N>sess` | 30,6 / 23,2 | 63,1 / 48,3 | 74,7 / 62,8 | 87,0 / 78,4 | 94,5 / 87,6 |
| MindEye2 do artigo (fine-tune completo, baixado) | `final_subj01_pretrained_<N>sess_24bs` | 94,1 / 78,2 | — | — | — | 100,0 / 99,9 |

O MindEye1 sem low-level (`subj01_me1_<N>sess_semlowlevel`) usa o mesmo checkpoint do alto nível e só muda a
reconstrução (sem o img2img a partir da imagem borrada), então o retrieval é o mesmo. Variantes só com 1 e 40
sessões:

| Modelo | Nome | Hiperparâmetros | 1 | 40 |
|---|---|---|---|---|
| FRR, uma fração para todas as dimensões | `subj01_frr_<N>sess_global` | `global_fraction` | 33,2 / 1,3 | 96,3 / 37,7 |
| FRR, grade estendida | `subj01_frr_<N>sess_ext` | `grid="extended"` | 55,7 / 2,0 | 92,3 / 21,3 |
| MindEye2 sem low-level nem prior | `subj01_ridgeonly_1sess` | `prior=False` | 78,4 / 72,6 | — |
| MindEye2 sem low-level, sementes 1 e 2 | `subj01_ridgeonly_1sess_prior_seed{1,2}` | `seed` | 93,1 / 87,8 e 93,2 / 88,1 | — |

Definidas e ainda não rodadas (`EXPERIMENTOS`): o MindEye2 com 2 sessões (com e sem low-level) e os de dataset
controlado: 5 classes semânticas, pessoa + animal, 1.000 imagens × 3 repetições nos quatro modos de agregação, e
3.000 × 1, 1.500 × 2, 1.000 × 3 imagens × repetições. O site os mostra em Fichas como planejados.

## Estrutura

| Pasta | |
|---|---|
| `setup/` | preparar a máquina (`bootstrap.sh` faz tudo) |
| `scripts/` | os comandos de cada etapa, em bash ([scripts/README.md](scripts/README.md)) |
| `src/` | `treina.py` e `experimentos.py` (o ponto de entrada), `make_site.py` e `make_benchmark.py` ([src/README.md](src/README.md)) |
| `src/etapas/` | as etapas do pipeline: treino, reconstrução, refino, métricas |
| `src/mindeye_ridge/` | biblioteca: dados, FRR, métricas, semântica, dataset controlado, agregação, resultados |
| `src/report/` | o código do site e do `BENCHMARK.md` |
| `vendor/` | código de terceiros (os geradores de imagem), como veio ([vendor/README.md](vendor/README.md)) |
| `mindeye1/` | MindEye1 adaptado, e o MindEye1 multi-sujeito ([mindeye1/README.md](mindeye1/README.md)) |
| `notebooks/` | `treino.ipynb`, `classificacao_semantica.ipynb`, originais do MindEye2 |
| `results/` | tabelas, site e página (no git); tensores e logs (fora) |
| `train_logs/` | checkpoints (fora do git) |
| `tests/` | 221 testes, sem GPU, ~35 s |
| `docs/` | documentação, com índice em [docs/README.md](docs/README.md) |

## Licença

MIT. O código de terceiros em `vendor/` (`generative_models/`, `autoencoder/`) mantém as licenças próprias.
