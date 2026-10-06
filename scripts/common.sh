# Configuracao comum dos scripts: caminhos, ambiente Python e GPU. Cada script faz
#     source "$(dirname "$0")/common.sh"
# e passa a rodar de dentro de src/, com os caminhos abaixo definidos. Para outra maquina,
# exporte MINDEYE_ENV (ambiente Python do setup_env.sh) e MINDEYE_DATA (dataset + cache de pesos).

# caminho absoluto de quem chamou: depois do cd abaixo, um $0 relativo deixa de existir
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/src"
SCRIPTS="$REPO/scripts"
ENVP="${MINDEYE_ENV:-$HOME/envs/fmri}"
DATA="${MINDEYE_DATA:-$HOME/mindeyev2}"
TRAIN_LOGS="$REPO/train_logs"
EVALS="$REPO/results/evals"
TABLES="$REPO/results/tables"
LOGS="$REPO/results/logs"

export MINDEYE_DATA="$DATA"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$DATA/.cache"                 # pesos do open_clip bigG vao para ca (~10 GB)
export TRANSFORMERS_CACHE="$DATA/.cache"
# Sem isto o treino em batch 16 nao cabe quando o desktop tambem roda na GPU: o allocator
# deixa ~2 GB reservados mas nao alocados e o backward estoura.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Dataset controlado (src/mindeye_ridge/dataset_controlado.py). Os scripts de treino aceitam
# DATASET=<manifesto.json> e chamam prepara_dataset antes de definir os nomes padrao: o caminho
# vira absoluto (o cd abaixo quebraria um relativo) e MODEL_NAME passa a ser obrigatorio, porque
# os nomes padrao sao os dos modelos do benchmark, e um treino controlado com eles sobrescreveria
# (ou retomaria) os modelos e as tabelas do benchmark.
ORIGEM="$PWD"
prepara_dataset() {
    [ -n "${DATASET:-}" ] || return 0
    case "$DATASET" in /*) ;; *) DATASET="$ORIGEM/$DATASET" ;; esac
    [ -f "$DATASET" ] || { echo "ERRO: DATASET=$DATASET nao existe" >&2; exit 1; }
    if [ -z "${MODEL_NAME:-}" ]; then
        echo "ERRO: com DATASET, defina MODEL_NAME (os nomes padrao sao os dos modelos do benchmark)" >&2
        exit 1
    fi
}

# Um modelo nao troca de dataset numa retomada. O hash do manifesto (ou "completo", sem DATASET)
# fica em <dir>/dataset.sha1; retomar com outro para o treino. Um diretorio com last.pth e sem o
# registro e de antes do dataset controlado, quando todo treino era nas sessoes inteiras.
#   confere_dataset <diretorio do modelo> <1 se o treino vai retomar o que esta nele, senao 0>
confere_dataset() {
    local dir=$1 retoma=$2 atual=completo antes=""
    [ -n "${DATASET:-}" ] && atual=$(sha1sum "$DATASET" | cut -d' ' -f1)
    if [ -f "$dir/dataset.sha1" ]; then antes=$(cat "$dir/dataset.sha1")
    elif [ -f "$dir/last.pth" ]; then antes=completo; fi
    if [ "$retoma" = 1 ] && [ -n "$antes" ] && [ "$antes" != "$atual" ]; then
        echo "ERRO: $dir foi treinado com outro dataset ($antes), e a retomada viria com $atual." >&2
        echo "      Use outro MODEL_NAME, ou apague o diretorio para treinar do zero." >&2
        exit 1
    fi
    mkdir -p "$dir"
    echo "$atual" > "$dir/dataset.sha1"
}

cd "$SRC"
