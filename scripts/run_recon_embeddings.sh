#!/bin/bash
# Reconstrucao e metricas de um modelo que preve o embedding CLIP bigG direto (o FRR), sem rede do
# MindEye2: o embedding do proprio modelo vai para o unCLIP (src/recon_embeddings.py).
#
#   scripts/run_recon_embeddings.sh <model_name> unclip    # unCLIP + metricas das reconstrucoes
#   scripts/run_recon_embeddings.sh <model_name> refino    # o mesmo, mais o refinamento (SDXL) e as metricas dele
#
# Precisa de results/evals/<modelo>/<modelo>_all_clipvoxels.pt (o run_frr.py grava). O treina.py
# chama este script com --reconstrucao. Sem ramo low-level: nao ha reconstrucao borrada.
#
# Etapa cuja saida ja existe e pulada, e a reconstrucao interrompida retoma do parcial; tabela de
# metricas ja existente tambem (FORCE=1 recalcula). SEED muda a semente da amostragem (padrao 42).
set -e
set -o pipefail
source "$(dirname "$0")/common.sh"

MODEL_NAME="$1"
MODO="$2"
[ -z "$MODEL_NAME" ] && { sed -n '2,12p' "$SELF"; exit 1; }
case "$MODO" in
    unclip|refino) ;;
    *) echo "ERRO: modo '$MODO': use unclip ou refino" >&2; exit 1 ;;
esac

LOGDIR=$TRAIN_LOGS/$MODEL_NAME; mkdir -p "$LOGDIR"
OUT=$EVALS/$MODEL_NAME/$MODEL_NAME
[ -f "${OUT}_all_clipvoxels.pt" ] || { echo "ERRO: ${OUT}_all_clipvoxels.pt nao existe -- rode o modelo antes" >&2; exit 1; }

if [ -f "${OUT}_all_recons.pt" ] && [ ! -f "${OUT}_recon_partial.pt" ]; then
    echo "=== recon_embeddings: $MODEL_NAME ja feito, pulando ==="
else
    echo "=== recon_embeddings: $MODEL_NAME ==="
    $ENVP/bin/python recon_embeddings.py --model_name="$MODEL_NAME" --cache_dir="$DATA" --seed="${SEED:-42}" \
        2>&1 | tee -a "$LOGDIR/recon.log"
fi
"$SCRIPTS/run_evals.sh" "$MODEL_NAME" base

[ "$MODO" = refino ] || exit 0
if [ -f "${OUT}_all_enhancedrecons.pt" ]; then
    echo "=== enhanced_recon_inference: $MODEL_NAME ja feito, pulando ==="
else
    echo "=== enhanced_recon_inference: $MODEL_NAME ==="
    $ENVP/bin/python enhanced_recon_inference.py --model_name="$MODEL_NAME" \
        --data_path=$DATA --cache_dir=$DATA --subj=1 --seed="${SEED:-42}" \
        2>&1 | tee -a "$LOGDIR/enhanced.log"
fi
"$SCRIPTS/run_evals.sh" "$MODEL_NAME" enhanced
