# O que rodar de novo para o benchmark ficar completo

Levantamento de 12/10/2026, feito a partir do que está no git: os registros de cada modelo em
`results/site/modelos/*.json` (o que o site conseguiu extrair de cada corrida) e o
[BENCHMARK.md](../BENCHMARK.md). Não foi conferido contra o disco da máquina de treino; onde isso
importa, está dito.

## O que uma corrida completa registra

O pipeline atual guarda cinco coisas de cada modelo. É a régua usada abaixo.

| # | O quê | Onde fica | Quem grava |
|---|---|---|---|
| 1 | configuração inteira, commit do código e manifesto do dataset | `train_logs/<nome>/experimento.json` e `dataset.json` | `src/treina.py` |
| 2 | curva por época e tempo de treino | `train_logs/<nome>/metrics.csv` | `train_ridgeonly.py` (só o MindEye2) |
| 3 | retrieval com intervalo de 95% dos 30 sorteios | `results/evals/<nome>/<nome>_retrieval.json` ou `results/tables/<nome>_frr.json` | `verify_retrieval.py`, `run_frr.py` |
| 4 | reconstruções e todas as métricas (unCLIP e refinada, legendas, correlação cerebral) | `results/tables/<nome>_all_*.csv` | `run_recon.sh` e `run_evals.sh` |
| 5 | tira da galeria | `results/site/galeria/` | `make_site.py`, a partir dos tensores em `results/evals/` |

## Resumo

Em ordem de prioridade. Só os dois últimos itens pedem treino.

| # | O que fazer | Modelos | Treina de novo? | Custo numa A4500 |
|---|---|---|---|---|
| 1 | reconstruir e avaliar os retreinos com o código atual | `subj01_ridgeonly_1sess_prior_seed42`, `subj01_ridgeonly_1sess_noprior_seed42` | não | ~3 h de inferência cada, mais as métricas |
| 2 | reconstruir e avaliar as outras sementes | `subj01_ridgeonly_1sess_prior_seed1`, `..._seed2` | não | ~3 h de inferência cada, mais as métricas |
| 3 | regenerar o site onde estão os arquivos do treino | `subj01_ridgeonly_40sess_4096blurry` | não, se o `metrics.csv` ainda existir | minutos (12 h 22 se precisar treinar) |
| 4 | medir o retrieval com intervalo de confiança | os cinco ridge-only de referência e os dois do artigo | não | minutos cada, sem difusão |
| 5 | refazer pelo `treina.py`, para ter configuração e commit registrados | todos; vale a pena começar pelo FRR | sim, com outro nome | FRR: 10 s a 25 min; os outros, o tempo do treino |
| 6 | gravar a curva de treino do MindEye1 | `subj01_me1_1sess`, `subj01_me1_40sess` | sim, depois de mudar o código | 2 h 53 e 32 h 38 |

O que **não** precisa ser refeito: o FRR (seis variantes, com tempo, intervalo e retrieval centrado),
o `subj01_ridgeonly_1sess_4096blurry` e o `subj01_ridgeonly_40sess_prior` (curva, tempo e métricas
completas) e os quatro MindEye1 (métricas e tempo). Nesses só valem os itens 4 a 6.

## 1. Os dois modelos de 1 sessão que vieram do release

`subj01_ridgeonly_1sess_prior` e `subj01_ridgeonly_1sess` são as linhas "Ridge 1024 + prior" e "Ridge
1024 sem prior" do benchmark, e são os dois checkpoints do release `checkpoints-v1`: foram treinados com
uma versão anterior do código, e o sem prior com outro script. Por isso não têm curva de treino, o
tempo do primeiro é uma estimativa (`~1 h 50 (corrida anterior)`) e o do segundo não existe.

O retreino com o código atual já foi feito, com a mesma semente: `subj01_ridgeonly_1sess_prior_seed42`
(1 h 53) e `subj01_ridgeonly_1sess_noprior_seed42` (1 h 02). Os dois têm curva, tempo e retrieval com
intervalo, e o retrieval bate com o do release (92,9% / 88,1% contra 92,8% / 88,1%; 78,4% / 72,6%
contra 78,5% / 72,1%). Falta só o item 4 da régua: eles nunca foram reconstruídos, então não têm
métricas de imagem, legendas, correlação cerebral nem galeria.

```bash
MS1024=~/mindeyev2/train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth
for m in subj01_ridgeonly_1sess_prior_seed42 subj01_ridgeonly_1sess_noprior_seed42; do
    FROZEN_CKPT=$MS1024 scripts/run_recon.sh $m 1024 noblurry
    scripts/run_evals.sh $m base
    scripts/run_evals.sh $m enhanced
done
```

O `FROZEN_CKPT` é obrigatório no sem prior (o checkpoint dele não tem prior) e inofensivo no outro: no
ridge-only, backbone e prior ficam congelados e são os do ponto de partida.

Depois disso, a decisão é trocar as linhas de referência do benchmark pelos retreinos. Hoje o
`report/config.py` (`MODELOS`) e o `report/site.py` (`PRESETS`, `PARES`, `OCULTOS`) apontam para os
checkpoints do release, e o site esconde os `_seed42`. A semente 7 da reconstrução
(`subj01_ridgeonly_1sess_prior_rseed7`) também usa o checkpoint do release e teria de ser refeita sobre
o retreino para a régua de ruído continuar valendo.

## 2. O efeito da semente do treino nas métricas de imagem

As sementes 1 e 2 do Ridge 1024 + prior só têm retrieval. A régua de ruído do benchmark tem, por isso,
um buraco que o próprio BENCHMARK.md registra: "O efeito da semente do treino nas métricas de imagem
não foi medido." Hoje dá para dizer que trocar a semente move o retrieval em menos de 0,5 ponto, mas
não quanto move PixCorr, SSIM ou as identificações 2-way.

```bash
for s in 1 2; do
    m=subj01_ridgeonly_1sess_prior_seed$s
    FROZEN_CKPT=$MS1024 scripts/run_recon.sh $m 1024 noblurry
    scripts/run_evals.sh $m base
    scripts/run_evals.sh $m enhanced
done
```

Com o item 1, ficam três sementes de treino com todas as métricas.

## 3. Ridge 4096 + blurry de 40 sessões: o treino existe, o registro não

As tabelas deste modelo estão completas e o BENCHMARK.md traz o tempo (12 h 22) e os parâmetros
treinados (64,4 M de 2.227,3 M). No site, porém, ele aparece sem curva de treino, sem tempo, sem
parâmetros e sem tira na galeria: o registro `results/site/modelos/subj01_ridgeonly_40sess_4096blurry.json`
foi extraído numa máquina sem o `train_logs/` e o `results/evals/` dele.

Nada a treinar se esses arquivos ainda estiverem na máquina onde o modelo rodou:

```bash
ls train_logs/subj01_ridgeonly_40sess_4096blurry/metrics.csv \
   results/evals/subj01_ridgeonly_40sess_4096blurry/*recons.pt
python src/make_site.py          # completa o registro e gera a tira da galeria
git add results/site
```

Se o `metrics.csv` tiver sido apagado, a curva só volta treinando de novo (item 5, 12 h 22). Se
faltarem só os tensores, basta reconstruir (`FROZEN_CKPT` do `final_multisubject_subj01`, `4096 blurry`).

## 4. Retrieval sem intervalo de confiança

O `final_evaluations.py` faz os 30 sorteios de 300 mas guarda só a média. Têm intervalo de 95% apenas
o FRR e os modelos medidos pelo `verify_retrieval.py` (as sementes do item 1 e 2). Os outros aparecem
no site com o ponto e sem a barra: os cinco ridge-only de referência, os dois modelos do artigo e os
quatro MindEye1.

Para os sete do MindEye2 basta o `verify_retrieval.py`, que não passa pela difusão. A chamada é a da
função `retrieval` do `scripts/run_benchmark.sh`, de dentro de `src/`:

```bash
MS4096=~/mindeyev2/train_logs/final_multisubject_subj01/last.pth
ret() {   # ret <modelo> <hidden_dim> <--blurry_recon|--no-blurry_recon> [ckpt de partida]
    ~/envs/fmri/bin/python verify_retrieval.py --model_name=$1 --data_path=$HOME/mindeyev2 \
        --cache_dir=$HOME/mindeyev2 --subj=1 --hidden_dim=$2 --n_blocks=4 $3 ${4:+--frozen_ckpt=$4} --new_test
}
ret subj01_ridgeonly_1sess_prior        1024 --no-blurry_recon
ret subj01_ridgeonly_1sess              1024 --no-blurry_recon $MS1024
ret subj01_ridgeonly_40sess_prior       1024 --no-blurry_recon $MS1024
ret subj01_ridgeonly_1sess_4096blurry   4096 --blurry_recon    $MS4096
ret subj01_ridgeonly_40sess_4096blurry  4096 --blurry_recon    $MS4096
ret final_subj01_pretrained_1sess_24bs  4096 --blurry_recon
ret final_subj01_pretrained_40sess_24bs 4096 --blurry_recon
```

O site passa a usar o `_retrieval.json` no lugar da média das tabelas assim que ele existe. Estas sete
chamadas não foram testadas com estes modelos; no benchmark a função só rodou nas sementes do 1024.

O MindEye1 fica de fora: o `scripts/me1_run.sh retrieval` imprime o resultado e não grava o JSON que o
site lê. Para ele ter a barra é preciso antes fazer o `Retrievals_testset.py` gravar o arquivo.

## 5. Nenhum modelo do benchmark tem configuração e commit registrados

Os 24 registros do site têm `commit` vazio, e a configuração de todos vem do dicionário em
`src/experimentos.py`, escrito depois. Nenhum foi treinado pelo `treina.py`, que é quem grava o
`experimento.json` (hiperparâmetros completos, comando, commit) e o `dataset.json` (as exibições usadas).
Os dicionários foram conferidos (o FRR de 1 sessão refeito pelo `treina.py` dá os mesmos números, e o
MindEye2 com e sem o manifesto dá as mesmas métricas época a época), mas não há como saber, pelo
registro, com que versão do código cada modelo foi treinado.

O `treina.py` não treina de novo um nome que já tem modelo. Para refazer, use outro nome com o mesmo
dicionário (de dentro de `src/`):

```python
import treina
from experimentos import BENCHMARK
for nome in ("subj01_frr_1sess", "subj01_frr_40sess"):
    treina.roda("v2_" + nome, BENCHMARK[nome])
```

Onde vale a pena, do mais barato ao mais caro:

| Modelos | Tempo | Vale a pena? |
|---|---|---|
| FRR, as seis variantes | 10 s (1 sessão) a 25 min (40 sessões) | sim: custa menos de 1 h 30 no total e fecha o registro do baseline |
| Ridge 1024 de 1 sessão, com e sem prior | 1 h 53 e 1 h 02 | sim, se o item 1 for feito pelo `treina.py` em vez de aproveitar os `_seed42` |
| Ridge 4096 + blurry de 1 sessão | 2 h 41 | só se o commit importar: o treino é determinístico e já repetiu época por época |
| Ridge de 40 sessões (1024 e 4096) | 7 h 51 e 12 h 22 | idem |
| MindEye1 de 1 e 40 sessões | 2 h 53 e 32 h 38 | só junto com o item 6 |

Um modelo refeito com outro nome entra no site como modelo novo, com reconstrução e avaliação à parte.
Os dois do artigo não entram nesta conta: foram treinados pelos autores.

## 6. O MindEye1 não tem curva de treino

O `mindeye1/src/Train_MindEye.py` não grava `metrics.csv`, então a aba Treino do site só mostra o
MindEye2. Para ter a curva é preciso primeiro fazer o treino gravar o arquivo, no formato que o
`report/catalogo.py` lê (`CURVA`), e depois treinar de novo: 2 h 53 com 1 sessão e 32 h 38 com 40. É o
item mais caro e o de menor retorno; só compensa se a comparação da dinâmica de treino entre o MindEye1
e o MindEye2 entrar no trabalho.

## O que ainda não rodou nenhuma vez

Fora deste levantamento ficam os 12 experimentos de dataset controlado de `src/experimentos.py`
(`EXPERIMENTOS`): não são re-execuções, e sim os próximos a rodar. `python src/treina.py --lista`
mostra todos, e eles aparecem no site em Fichas, como planejados. Rodados pelo `treina.py`, já saem com
os cinco itens da régua, menos a reconstrução e a avaliação, que continuam à parte (`run_recon.sh` e
`run_evals.sh`).
