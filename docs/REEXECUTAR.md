# O que rodar de novo para o benchmark ficar completo

Levantamento de 12/10/2026, atualizado em 15/10/2026 depois da reexecução. Parte do que está no git: os
registros de cada modelo em `results/site/modelos/*.json` (o que o site conseguiu extrair de cada corrida)
e o [BENCHMARK.md](../BENCHMARK.md).

## Situação em 15/10/2026

A reexecução rodou numa RTX A4500 (a máquina de treino original não estava disponível), pelo
`scripts/run_reexecutar.sh`: 52 h 19 de GPU, de 12/10 às 18h09 a 14/10 às 22h29. Cada modelo foi treinado
pelo `src/treina.py` com o dicionário do benchmark, com outro nome (`repro_<nome>`), porque o `treina.py`
não sobrescreve um modelo do benchmark. Cada um ficou com configuração, commit limpo do código, manifesto
do dataset, curva, tempo, retrieval com intervalo de 95% e, nos ridge-only, reconstrução e métricas. Os
resultados de cada modelo foram commitados quando ele terminou. Nenhuma tabela do benchmark foi alterada: as
reproduções aparecem no site, no grupo Experimentos, com o rótulo "· reprodução".

| Modelo (`repro_subj01_…`) | Treino medido | Treino original | Situação |
|---|---|---|---|
| `frr_1sess`, `_global`, `_ext` | 1 min cada | 10 s | completo |
| `frr_40sess`, `_global`, `_ext` | 24, 20 e 25 min | 21 min | completo |
| `ridgeonly_1sess` (1024 sem prior) | 1 h 00 | 1 h 02 (semente 42) | completo |
| `ridgeonly_1sess_prior_seed1`, `_seed2` | 1 h 51 e 1 h 54 | 1 h 52 e 1 h 53 | completo |
| `ridgeonly_1sess_4096blurry` | 2 h 45 | 2 h 41 | completo |
| `ridgeonly_40sess_prior` | 8 h 03 | 7 h 51 | completo |
| `ridgeonly_40sess_4096blurry` | 12 h 40 | 12 h 22 | completo |
| **`ridgeonly_1sess_prior`** (1024 + prior) | 1 h 51 | 1 h 53 (semente 42) | **só treino e retrieval** |
| `final_subj01_pretrained_1sess_24bs`, `_40sess_24bs` | — (dos autores) | — | retrieval com intervalo |

Medido nesta máquina: reconstrução de 186 a 194 min por modelo, 6 min para cada tabela de métricas (unCLIP e
refinada) e 3 min para o retrieval.

### Quanto as reproduções se afastam dos originais

| Modelo | Maior diferença nas métricas de imagem |
|---|---|
| FRR, as seis variantes | 0 (idêntico, inclusive o R² da validação cruzada) |
| Ridge 4096 + blurry, 1 sessão | 0 (idêntico, época por época) |
| Ridge 1024 + prior, 40 sessões | 0,89 p.p. nas porcentagens; 0,0023 nas demais |
| Ridge 4096 + blurry, 40 sessões | 0,59 p.p.; 0,0064 (PixCorr 0,337 contra 0,339) |
| Ridge 1024 sem prior, contra o release | 1,24 p.p.; 0,0008 |
| Sementes 1 e 2, contra o release do 1024 + prior | 1,19 e 0,95 p.p.; 0,0056 e 0,0038 |

O retrieval do 1024 sem prior (78,41% / 72,59%) e o do 1024 + prior (92,91% / 88,12%) saem iguais aos dos
retreinos de semente 42 feitos antes. As diferenças da tabela são um pouco maiores que o ruído de
reconstrução medido antes (até 0,6 p.p. nas identificações 2-way). A causa não foi investigada: os originais
foram treinados em outra máquina e, no caso dos checkpoints do release, com uma versão anterior do código.

## O que ainda falta, em ordem

**Uma única semente (42) por modelo.** Não se treinam nem se reconstroem outras sementes. As sementes 1 e 2 do
Ridge 1024 + prior e a reconstrução com a semente 7 (`…_rseed7`, só no benchmark original) são a única
medida de ruído, feita uma vez; não a repita.

1. **Reconstrução e métricas do Ridge 1024 + prior de 1 sessão** (`repro_subj01_ridgeonly_1sess_prior`),
   ~3 h 30 numa A4500. O treino e o retrieval dele já existem. Foi interrompida em 15/10, por volta das 16h30, a pedido,
   sem gravar nenhuma imagem. A primeira tentativa falhou por um defeito do `run_recon.sh`, já corrigido
   (`6e7e078`).

   ```bash
   SO=1sess COMMITA=1 scripts/run_reexecutar.sh
   ```

   As etapas concluídas são puladas pelos marcadores em `train_logs/<modelo>/.reexec_*`, então só a
   reconstrução e as métricas rodam. `COMMITA=1` faz um commit dos resultados ao fim; esta máquina não tem
   `user.email` no git, então exporte `GIT_AUTHOR_EMAIL` e `GIT_COMMITTER_EMAIL` ou configure o e-mail.

2. **Passar todos os modelos de referência para as reproduções.** Decidido: as linhas do benchmark
   (`report/config.py`, `MODELOS`; `report/site.py`, `PRESETS`, `PARES`, `OCULTOS`) passam a ser os `repro_…`,
   que têm commit, configuração, curva, tempo e retrieval com intervalo. O que falta para cada modelo:

   | Modelo de referência | Reprodução | Falta | Tempo (A4500) |
   |---|---|---|---|
   | FRR 1 e 40 sessões, e as variantes `_global` e `_ext` | completa, idêntica ao original | só trocar a referência | 0 |
   | Ridge 1024 sem prior, 1 sessão | completa | só trocar a referência | 0 |
   | Ridge 4096 + blurry, 1 sessão | completa | só trocar a referência | 0 |
   | Ridge 1024 + prior, 40 sessões | completa | só trocar a referência | 0 |
   | Ridge 4096 + blurry, 40 sessões | completa | só trocar a referência | 0 |
   | **Ridge 1024 + prior, 1 sessão** | treino e retrieval | reconstrução e métricas (item 1) | ~3 h 30 |
   | **MindEye1, 1 sessão** (e o low-level) | não existe | treino, reconstrução, métricas e a curva | ~5 h |
   | **MindEye1, 40 sessões** (e o low-level) | não existe | idem | ~32 h |
   | Artigo, 1 e 40 sessões | só o retrieval com intervalo | nada: os pesos são dos autores e não se retreinam | 0 |

   Total de GPU que falta: ~3 h 30 (item 1) mais ~37 h do MindEye1 (`scripts/run_me1_benchmark.sh`, que treina,
   reconstrói e avalia), ou seja, ~40 h 30. Sem o MindEye1, ~3 h 30. Os modelos `…_semlowlevel` do MindEye1
   (`LOWLEVEL=0`) pedem outro treino do alto nível cada um, que não está nessa soma e não foi medido.

   Trabalho de código, sem GPU: o `Train_MindEye.py` precisa gravar `metrics.csv` no formato que o
   `report/catalogo.py` lê (`CURVA`); os dois dicionários do MindEye1 entram em `REPRODUCOES` de
   `src/experimentos.py`; e a troca das referências obriga a reler as conclusões escritas à mão em
   `report/texts.py`, porque os números mudam até 1,24 p.p. (o FRR e o 4096 de 1 sessão não mudam).

Fora do escopo, por decisão: análise com várias sementes (o texto "o efeito da semente do treino nas
métricas de imagem não foi medido" fica como está) e a reconstrução com a semente 7.

## O que não dá para fazer sem a máquina de treino original

- **Retrieval com intervalo dos cinco ridge-only de referência** (`subj01_ridgeonly_1sess_prior`,
  `_1sess`, `_1sess_4096blurry`, `_40sess_prior`, `_40sess_4096blurry`): precisa dos checkpoints deles, que
  não estão aqui. Os registros originais seguem com a média sem intervalo (`final_evaluations`); as
  reproduções têm o intervalo.
- **Completar o registro do `subj01_ridgeonly_40sess_4096blurry`** (curva, tempo, parâmetros e galeria):
  precisa do `train_logs/` e do `results/evals/` desse modelo. O `repro_subj01_ridgeonly_40sess_4096blurry`
  cobre os quatro.

## A régua: o que uma corrida completa registra

| # | O quê | Onde fica | Quem grava |
|---|---|---|---|
| 1 | configuração inteira, commit do código e manifesto do dataset | `train_logs/<nome>/experimento.json` e `dataset.json` | `src/treina.py` |
| 2 | curva por época e tempo de treino | `train_logs/<nome>/metrics.csv` | `train_ridgeonly.py` (só o MindEye2) |
| 3 | retrieval com intervalo de 95% dos 30 sorteios | `results/evals/<nome>/<nome>_retrieval.json` ou `results/tables/<nome>_frr.json` | `verify_retrieval.py`, `run_frr.py` |
| 4 | reconstruções e todas as métricas (unCLIP e refinada, legendas, correlação cerebral) | `results/tables/<nome>_all_*.csv` | `run_recon.sh` e `run_evals.sh` |
| 5 | tira da galeria | `results/site/galeria/` | `make_site.py`, a partir dos tensores em `results/evals/` |

Os modelos originais do benchmark não têm os itens 1 e 2 (o `commit` está vazio nos registros deles, e a
configuração vem do dicionário de `src/experimentos.py`, escrito depois), e os checkpoints do release não têm
curva nem tempo. Foi isso que a reexecução supriu.

## Como foi rodado

```bash
setup/bootstrap.sh                                  # ambiente, dados (~62 GB), pesos e MindEye1
COMMITA=1 scripts/run_reexecutar.sh                 # a fila inteira (~52 h); SO="frr 1sess" roda só alguns blocos
python src/treina.py --lista                        # as reproduções aparecem em "reproducoes do benchmark"
```

Blocos de `SO`: `frr`, `1sess`, `sementes`, `4096`, `artigo`, `40sess`. A fila pode ser interrompida e rodada
de novo, e a falha de um modelo não para os outros (uma linha por etapa em `results/logs/reexecutar.log`).

Dois defeitos do setup apareceram numa máquina nova e foram corrigidos: o `pyarrow` 26 exigia NumPy 2 e
quebrava as métricas de legenda e a reconstrução (agora pinado em 15.0.2 no `setup_env.sh`), e o conda recente
exige aceitar os termos da Anaconda (contornado criando o Python 3.11 pelo canal conda-forge, sem conda, com o
`venv` que o `setup_env.sh` já prevê).

Os 12 experimentos de dataset controlado de `src/experimentos.py` (`EXPERIMENTOS`) não são re-execuções, e
sim os próximos a rodar; `python src/treina.py --lista` mostra todos, e eles aparecem em Fichas, no site,
como planejados.
