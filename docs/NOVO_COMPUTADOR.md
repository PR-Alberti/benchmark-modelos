# Do computador zerado ao primeiro treino (só terminal)

Passo a passo para uma máquina Linux sem interface gráfica, acessada por SSH. Os comandos foram
conferidos neste repositório; os tamanhos vêm do `--dry-run` do `setup/download_data.py`.

## 0. O que a máquina precisa ter

| | |
|---|---|
| GPU NVIDIA | 20 GB de VRAM (testado numa RTX A4500). O FRR e o MindEye2 ridge-only cabem em 16 GB; o MindEye1 precisa dos 20 |
| Driver NVIDIA | só o driver (525 ou mais novo); o CUDA vem dentro do `pip install torch` |
| RAM | 30 GB (o FRR de 40 sessões chega a ~25 GB de pico) |
| Software | `git`, `curl`, `tmux` e conda (ou `python3.11`). Python 3.12+ não serve: o `torch==2.1.0` não tem wheel |
| Conta | aceitar os [termos do NSD](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) e preencher o [formulário](https://forms.gle/xue2bCdM9LaFNMeb7). O download em si usa o dataset público `pscotti/mindeyev2` do HuggingFace, sem token (um `HF_TOKEN` só ajuda se o HuggingFace limitar o ritmo) |

### Quanto baixar (e quanto de disco reservar)

| O que você vai fazer | Download | Disco recomendado |
|---|---|---|
| **Mínimo** (`--minimo`): MindEye2 ridge-only e FRR com 1 sessão | 25 GB de dados + ~10 GB do CLIP bigG, baixado no primeiro treino | 40 GB |
| **FRR e MindEye2 ridge-only em qualquer subconjunto das 40 sessões** (caminho B, abaixo) | ~25 GB de dados + ~10 GB do CLIP bigG + 0,5 GB de anotações do COCO + 7,7 GB de alvos CLIP calculados na 1ª vez | 60 GB |
| **Completo** (`bootstrap.sh`): tudo, inclusive reconstrução, métricas e MindEye1 | **~130 GB**: 89 GB de dados e checkpoints + ~40 GB de pesos de terceiros (o Versatile Diffusion do MindEye1 sozinho tem 23 GB) | 200 GB |

Cada treino do MindEye1 grava ~17 GB por modelo, então o disco do completo cresce com o que você treina.

```bash
df -h ~          # confira o espaço livre antes
nvidia-smi       # a GPU e o driver respondem?
```

## 1. Ferramentas básicas

```bash
sudo apt update && sudo apt install -y git curl tmux build-essential
# conda, se a máquina não tiver (o bootstrap usa python3.11 -m venv como alternativa)
curl -fsSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3 && ~/miniconda3/bin/conda init bash
exec bash
```

## 2. Clonar o repositório

```bash
git clone https://github.com/PR-Alberti/benchmark-modelos.git
cd benchmark-modelos
git checkout controlled-dataset      # enquanto a branch não estiver na main
```

## 3. Preparar a máquina

Rode dentro do `tmux`, porque o download e a instalação levam horas e a conexão SSH pode cair:

```bash
tmux new -s setup        # para sair sem matar: Ctrl-b e depois d; para voltar: tmux attach -t setup
```

**Caminho A: tudo, num comando** (os ~130 GB; é o que permite reconstruir imagens, calcular métricas e rodar o MindEye1):

```bash
setup/bootstrap.sh --check     # só diagnostica: GPU, disco, o que falta; não baixa nada
setup/bootstrap.sh             # cria o ambiente, baixa dados e pesos, prepara o MindEye1
```

Para só 1 sessão do ridge-only e do FRR (~35 GB): `setup/bootstrap.sh --minimo`.

**Caminho B: só o necessário para treinar o FRR e o MindEye2 ridge-only com dataset controlado** (~43 GB):

```bash
setup/setup_env.sh --with-extras                              # ambiente ~/envs/fmri (~15 min)
~/envs/fmri/bin/python setup/download_data.py --stage finetune --subj 1   # dados das 40 sessões + ponto de partida
```

Para reconstruir as imagens do FRR (`treina.py --reconstrucao unclip|refino`), o caminho B precisa
ainda do unCLIP, do SDXL do refinamento e das redes das métricas (~40 GB; o Versatile Diffusion do
MindEye1 não entra):

```bash
~/envs/fmri/bin/python setup/download_data.py --stage recon enhanced evals --subj 1
~/envs/fmri/bin/python setup/prefetch_weights.py      # o Versatile Diffusion (23 GB) só serve ao MindEye1
```

Se `XDG_CACHE_HOME` apontar para uma partição pequena, exporte antes `HF_HOME=~/.cache/huggingface` e
`TORCH_HOME=~/.cache/torch`: o HuggingFace e o `torch.hub` gravam lá os temporários dos downloads, e um
disco cheio derruba o download com `No space left on device`, mesmo com espaço em `~`.

Nos dois caminhos, tudo é idempotente: se cair, rode o mesmo comando e ele continua de onde parou.
Os dados ficam em `~/mindeyev2` (mude com `MINDEYE_DATA=/outro/lugar` ou `--data-path`) e o ambiente
em `~/envs/fmri` (`MINDEYE_ENV`). Exporte as mesmas variáveis em todos os comandos seguintes se mudar.

Confira no fim:

```bash
~/envs/fmri/bin/python -m unittest discover -s tests      # 134 testes, ~15 s; "OK (skipped=3)"
setup/bootstrap.sh --check                                # o resumo diz o que ainda falta
```

## 4. Rodar um treino

Em terminal, o jeito é o `src/treina.py`. Cada treino é um dicionário com o modelo (`frr`, `mindeye2`
ou `mindeye1`), os hiperparâmetros e o dataset; o que faltar fica no padrão do modelo.

```bash
tmux new -s treino
cd ~/benchmark-modelos/src
~/envs/fmri/bin/python treina.py --lista        # os experimentos prontos, e os modelos do benchmark
```

### Exemplo genérico: treinar só com imagens de animais

Escreva o dicionário num arquivo (não precisa editar o repositório):

```bash
cat > ~/meu_experimento.py <<'EOF'
import sys
sys.path.insert(0, "/home/SEU_USUARIO/benchmark-modelos/src")    # troque pelo caminho do clone
import treina

exp = {
    "modelo": "frr",                       # frr | mindeye2 | mindeye1
    "hiper": {"seed": 42},                 # hiperparâmetros do modelo; o resto é padrão
    "dataset": {
        "sessoes": 40,                     # pool: as primeiras N sessões
        "classes": {"animal": ["sup_animal"]},   # só imagens em que animal ocupa >= minimo% da tela
        "minimo": 10,
        # "n_imagens": 500, "repeticoes": 3,     # opcional: imagens únicas e exibições por imagem
        # "agregacao": "media",                  # opcional: média das repetições (frr: exibicoes | media)
    },
}
treina.roda("meu_frr_animais", exp, dry_run=True)      # troque por dry_run=False para treinar
EOF
cd ~/benchmark-modelos/src
~/envs/fmri/bin/python ~/meu_experimento.py
```

1. **Com `dry_run=True`** ele monta o dataset, imprime quantas imagens sobraram (1.240 imagens, 3.720
   exibições, no caso dos animais) e mostra o comando que rodaria. Nada é gravado.
2. **Com `dry_run=False`** treina de verdade. O modelo fica em `train_logs/meu_frr_animais/`, com o
   manifesto do dataset (`dataset.json`), a configuração completa (`experimento.json`) e o log.
3. **O mesmo exemplo com outro modelo:** troque `"modelo"` e os hiperparâmetros. Para o MindEye2:
   `"hiper": {"num_epochs": 20, "frozen_fp16": True, "seed": 42}`. Num subconjunto pequeno as épocas
   encolhem junto com os dados, então aumente `num_epochs` para ter passos parecidos com o benchmark.
4. **Nome:** é o nome do modelo. Um nome já treinado não é sobrescrito; mude o nome para repetir.

Os hiperparâmetros aceitos, os padrões e as combinações recusadas estão em `treina.py` (`MODELOS`); as
opções do dataset, em `src/mindeye_ridge/dataset_controlado.py`. O [notebooks/treino.ipynb](../notebooks/treino.ipynb)
mostra todos os campos com exemplos, mas num terminal puro o arquivo acima faz o mesmo.

Para ver as categorias disponíveis (as colunas `sup_*` e `cat_*`) e quantas imagens cada escolha deixa:

```bash
cd ~/benchmark-modelos/src
~/envs/fmri/bin/python - <<'EOF'
from mindeye_ridge import paths, semantica
fr = semantica.tabela_fracoes(paths.DATA, 1)      # baixa as anotações do COCO (~250 MB) na 1ª vez
print([c for c in fr.columns if c.startswith("sup_")])
EOF
```

### Treinos longos

Rode em `tmux` ou com `nohup`, para sobreviver ao fechamento do SSH:

```bash
cd ~/benchmark-modelos/src
nohup ~/envs/fmri/bin/python ~/meu_experimento.py > ~/meu_experimento.log 2>&1 &
```

Para acompanhar:

```bash
tail -f ~/benchmark-modelos/train_logs/meu_frr_animais/frr.log        # FRR
tail -f ~/benchmark-modelos/train_logs/<nome>/train.log               # MindEye2 e MindEye1 (MindEye1: mindeye1/train_logs/<nome>/)
watch -n 5 nvidia-smi                                                 # uso da GPU
```

Se interromper, rode o mesmo comando de novo: o FRR retoma da última dobra da validação cruzada e o
MindEye1, do `last.pth`. O MindEye2 só retoma com `treina.roda(..., retoma=True)` (na linha de comando,
`--retoma`); sem isso ele recomeça da época 0. Se o nome já tem **outro**
dataset ou outros hiperparâmetros, ele para, para não misturar configurações.

Quanto leva, numa A4500: FRR ~1 min (1 sessão) a ~30 min (40 sessões); MindEye2 ridge-only ~2 h (1 sessão)
a ~8 h (40 sessões, 20 épocas); MindEye1 ~5 h (1 sessão) a ~20 h (40 sessões), sem contar o low-level.
Num subconjunto, o tempo cai com o número de exibições.

## 5. Ver o resultado

Todos os modelos treinados aparecem no site `results/site/index.html`, regenerado ao fim de cada
treino e de cada avaliação. Numa máquina sem interface, copie a pasta para o seu computador e abra o
`index.html` no navegador:

```bash
scp -r usuario@maquina:~/benchmark-modelos/results/site ./site-benchmark
```

Pelo terminal, o retrieval de um modelo:

```bash
cd ~/benchmark-modelos/src
~/envs/fmri/bin/python -c "
from mindeye_ridge import resultados
print(resultados.retrieval('meu_frr_animais'))
"
```

Devolve o retrieval top-1 (`fwd`: imagem → cérebro, `bwd`: cérebro → imagem) e de onde veio o número. O FRR
já sai com isso calculado, em `results/tables/<nome>_frr.json`. O MindEye2 e o MindEye1 precisam de mais
duas etapas para as imagens e as métricas completas (precisam do caminho A, que baixa os pesos de
reconstrução):

```bash
cd ~/benchmark-modelos
scripts/run_recon.sh <nome> 1024 noblurry     # MindEye2: reconstruções + refinamento
scripts/run_evals.sh <nome> enhanced          # MindEye2: métricas
scripts/me1_run.sh lowlevel                   # MindEye1: low-level (com as mesmas variáveis do treino)
scripts/me1_run.sh recon && scripts/me1_run.sh metrics
```

Com dataset controlado, o MindEye1 e o MindEye2 pedem as mesmas variáveis (`MODEL_NAME`, `DATASET`,
`NUM_SESSIONS`) em cada etapa; o README tem os detalhes.

## 6. Se algo der errado

| Sintoma | O que fazer |
|---|---|
| `nenhuma GPU NVIDIA utilizavel` | `nvidia-smi` falha: o driver não está instalado ou carregado |
| o download caiu | rode `setup/bootstrap.sh` (ou o `download_data.py`) de novo: continua de onde parou |
| `CUDA out of memory` | feche outros processos da GPU (`nvidia-smi`); o MindEye1 só cabe em 20 GB com batch 16 |
| `pkg_resources`, `cached_download` ou `deepspeed` no import | o ambiente saiu das versões do `setup_env.sh`; refaça com ele (os pins estão no [SETUP.md](SETUP.md), seção 3) |
| o treino recusa o nome | já existe modelo com esse nome (ou com outra configuração): use outro nome |
| faltou `semantica/subj01_fracao_tela.csv` | rode o `tabela_fracoes` acima uma vez; ele baixa o COCO e calcula (~20 s) |

Referências: [SETUP.md](SETUP.md) (cada arquivo baixado e os erros conhecidos), [MINDEYE1.md](MINDEYE1.md)
(o MindEye1) e o [README](../README.md).
