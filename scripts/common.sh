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
LOGS="$REPO/logs"

export MINDEYE_DATA="$DATA"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$DATA/.cache"                 # pesos do open_clip bigG vao para ca (~10 GB)
export TRANSFORMERS_CACHE="$DATA/.cache"
# Sem isto o treino em batch 16 nao cabe quando o desktop tambem roda na GPU: o allocator
# deixa ~2 GB reservados mas nao alocados e o backward estoura.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$SRC"
