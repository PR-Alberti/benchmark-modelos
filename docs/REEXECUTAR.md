# O que rodar de novo para o benchmark ficar completo

Levantamento de 12/10/2026, atualizado em 16/10/2026 depois da reexecução e em 21/10/2026 com o MindEye1 sem low-level. Parte do que está no git: os
registros de cada modelo em `results/site/modelos/*.json` (o que o site conseguiu extrair de cada corrida)
e o [BENCHMARK.md](../BENCHMARK.md).

## Situação em 16/10/2026

A reexecução rodou numa RTX A4500 (a máquina de treino original não estava disponível), pelo
`scripts/run_reexecutar.sh`: 52 h 19 de GPU, de 12/10 às 18h09 a 14/10 às 22h29. Cada modelo foi treinado
pelo `src/treina.py` com o dicionário do benchmark, com outro nome (`repro_<nome>`), porque o `treina.py`
não sobrescreve um modelo do benchmark. Cada um ficou com configuração, commit limpo do código, manifesto
do dataset, curva, tempo, retrieval com intervalo de 95% e, nos ridge-only, reconstrução e métricas. Os
resultados de cada modelo foram commitados quando ele terminou. Desde 19/10 as reproduções ocupam o lugar dos
originais no site e no BENCHMARK.md (veja **Referências**, abaixo).

| Modelo (`repro_subj01_…`) | Treino medido | Treino original | Situação |
|---|---|---|---|
| `frr_1sess`, `_global`, `_ext` | 1 min cada | 10 s | completo |
| `frr_40sess`, `_global`, `_ext` | 24, 20 e 25 min | 21 min | completo |
| `ridgeonly_1sess` (1024 sem prior) | 1 h 00 | 1 h 02 (semente 42) | completo |
| `ridgeonly_1sess_prior_seed1`, `_seed2` | 1 h 51 e 1 h 54 | 1 h 52 e 1 h 53 | completo |
| `ridgeonly_1sess_4096blurry` | 2 h 45 | 2 h 41 | completo |
| `ridgeonly_40sess_prior` | 8 h 03 | 7 h 51 | completo |
| `ridgeonly_40sess_4096blurry` | 12 h 40 | 12 h 22 | completo |
| `ridgeonly_1sess_prior` (1024 + prior) | 1 h 51 | 1 h 53 (semente 42) | completo (reconstrução e métricas em 16/10, 3 h 19) |
| `me1_1sess` (MindEye1) | 2 h 43 (1 h 40 + 1 h 02 do low-level) | 2 h 53 | completo (reconstrução 47 min, métricas 3 min) |
| `me1_40sess` (MindEye1) | 28 h 55 (16 h 03 + 12 h 51 do low-level) | 32 h 38 | completo (reconstrução 49 min, métricas 3 min) |
| `me1_1sess_semlowlevel` (MindEye1 sem low-level) | 1 h 41 | 1 h 40 | completo em 20/10 (reconstrução 50 min, métricas 3 min) |
| `me1_40sess_semlowlevel` (MindEye1 sem low-level) | 15 h 50 | 16 h 11 | completo em 21/10 (reconstrução 50 min, métricas 3 min) |
| `final_subj01_pretrained_1sess_24bs`, `_40sess_24bs` | — (dos autores) | — | retrieval com intervalo |

Medido nesta máquina: reconstrução de 186 a 194 min por modelo, 6 min para cada tabela de métricas (unCLIP e
refinada) e 3 min para o retrieval.

### Quanto as reproduções se afastam dos originais

| Modelo | Maior diferença nas métricas de imagem |
|---|---|
| FRR, as seis variantes | 0 (idêntico, inclusive o R² da validação cruzada) |
| Ridge 4096 + blurry, 1 sessão | 0 (idêntico, época por época) |
| MindEye1, 1 sessão | 0 (idêntico em todos os dígitos das métricas e do retrieval) |
| MindEye1, 40 sessões | 0,14 p.p. (Incep); retrieval idêntico; 0,0007 nas demais |
| MindEye1 sem low-level, 1 e 40 sessões | 0 (idêntico até a 4ª casa em todas as métricas e no retrieval) |
| Ridge 1024 + prior, 1 sessão, contra o release | 0,7 p.p. (Incep); 0,003 (Eff) |
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

**Referências.** Decidido em 16/10 e concluído em 19/10 (todas as linhas de referência vêm das reproduções, menos as do artigo; as `_semlowlevel` desde 21/10): as linhas do benchmark (os ids `subj01_…`) passam a ser lidas das
reproduções `repro_subj01_…`. O id público não mudou; `report/config.py` (`fonte`) aponta cada um para a
reprodução assim que ela tem as tabelas de resultado, e as linhas `repro_…` ficam escondidas do site para
não duplicar. Os registros e as tabelas dos originais seguem no histórico do git. Mudaram, pela reprodução,
as tabelas do BENCHMARK.md (até 1,24 p.p.) e as conclusões escritas à mão em `report/texts.py`, já revistas
com os números novos.

**Variantes `subj01_me1_*_semlowlevel`** (MindEye1 sem o low-level): refeitas em 20 e 21/10 com
`PREFIXO=repro_ LOWLEVEL=0 scripts/run_me1_benchmark.sh`, a partir das entradas `repro_…_semlowlevel` de
`REPRODUCOES`. Como as outras referências, as linhas `subj01_me1_*_semlowlevel` passam a ser lidas delas
(`ME1_SEM_LOWLEVEL` em `report/config.py`, já que não estão nas tabelas do BENCHMARK.md). Nada da reexecução
falta rodar nesta máquina; o que segue precisa da máquina original.

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

O low-level do MindEye1 de 40 sessões parou na época 90 de 120 (reinício da máquina em 19/10) e foi retomado do
`last.pth`. O `results/logs/me1_benchmark.log` (fora do git) ganhou uma linha de início ajustada para o tempo
contar só o treino ativo (12 h 51); sem ela o site mostraria 3 h 22.
