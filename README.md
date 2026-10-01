# mindeye2-ridge

MindEye2 (fMRI → imagem) com fine-tune apenas da camada ridge. Sujeito 1 do NSD,
1 sessão de dados. Baseado em
[MedARC-AI/MindEyeV2](https://github.com/MedARC-AI/MindEyeV2).

## Preparar uma máquina

```bash
git clone https://github.com/PR-Alberti/mindeye2-ridge.git
cd mindeye2-ridge && ./bootstrap.sh --all --ckpts
```

Cria o ambiente Python, baixa os 62 GB de dados e traz os checkpoints já
treinados. `./bootstrap.sh --check` mostra o que falta sem baixar nada.

Detalhes — bibliotecas, arquivos externos, backup, erros conhecidos — em
[SETUP.md](SETUP.md).

## Rodar

```bash
scripts/run_ridgeonly_prior.sh                                   # treino (~1h50)
scripts/run_recon.sh subj01_ridgeonly_1sess_prior 1024 noblurry  # reconstruções
scripts/run_evals.sh subj01_ridgeonly_1sess_prior enhanced       # métricas
scripts/run_frr.sh                                               # baseline linear (FRR), ~1 min
```

Roda de qualquer diretório. Métricas por época em `train_logs/<modelo>/metrics.csv`;
tabelas finais em `results/tables/`.

## Resultados

Os quatro experimentos ridge-only (1024 com e sem prior, 4096 + blurry, 40
sessões) e o baseline linear FRR contra os modelos publicados no artigo, com o
mesmo pipeline e as mesmas métricas: [BENCHMARK.md](BENCHMARK.md) traz as tabelas;
`benchmark/index.html` é a página completa, com galeria de reconstruções lado a
lado, curvas de treino e ruído entre sementes (um arquivo só, abre offline).
O que falta do plano de estágio está em [ESTAGIO.md](ESTAGIO.md).

```bash
scripts/run_benchmark.sh    # regenera tudo (~50 h numa A4500)
```

## Onde está o quê

| | |
|---|---|
| `scripts/` | comandos do dia a dia (`run_*.sh`); `common.sh` guarda os caminhos e o ambiente |
| `src/` | pontos de entrada: `train_ridgeonly.py` (`--ridge_only` congela tudo menos a ridge), `recon_inference.py`, `enhanced_recon_inference.py`, `final_evaluations.py`, `verify_retrieval.py`, `run_frr.py`, `make_benchmark.py`, `make_comparison.py` |
| `src/mindeye_ridge/` | biblioteca: `utils`, `models` e `modeling_git` (do MindEye2) e o que foi acrescentado — `paths`, `nsd_data`, `clip_targets`, `frr`, `embedding_metrics` |
| `src/report/` | o código da página do benchmark e do `BENCHMARK.md`: dados, tabelas, textos e, em `assets/`, o CSS e o JavaScript |
| `src/generative_models/`, `src/autoencoder/` | código de terceiros usado como está (Stability AI; ConvNeXt) |
| `results/` | `tables/` (métricas finais, no git), `metrics/`, `figs/` e `evals/` (tensores que cada modelo produz, fora do git) |
| `benchmark/` | a página do benchmark, gerada por `src/make_benchmark.py` |
| `tests/` | testes do FRR, da página do benchmark e dos caminhos: `python -m unittest discover -s tests` |
| `train_logs/` | checkpoints e logs (fora do git — release `checkpoints-v1`) |
| `notebooks/` | notebooks do MindEye2 original, só de referência |
| `legacy/` | geradores antigos, jobs SLURM e wrappers que já não são usados (leia o `legacy/README.md` antes de rodar qualquer coisa de lá) |
| `tools/` | utilitários avulsos |
| `bootstrap.sh`, `setup_env.sh` | preparam a máquina: ambiente, dados e checkpoints |
| `download_data.py` | baixa do HuggingFace só o que cada etapa precisa |
| `restore_ckpt.py` | conserta `.pth` que backup descompactou em pasta |
| [BENCHMARK.md](BENCHMARK.md) | resultados de todos os modelos |
| [SETUP.md](SETUP.md) | instalação e operação |
| [EXPERIMENTO.md](EXPERIMENTO.md) | o que foi feito e por quê |
| [README-original.md](README-original.md) | README do MindEye2, como referência |
