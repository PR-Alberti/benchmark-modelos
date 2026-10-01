#!/bin/bash
# Calcula as metricas finais sobre as reconstrucoes de um modelo.
#
#   scripts/run_evals.sh <model_name> [base|enhanced|published|published-base]
#
# "base"     -> avalia {model}_all_recons.pt         (saida do recon_inference)
# "enhanced" -> avalia {model}_all_enhancedrecons.pt (saida do enhanced_recon)
# "published" / "published-base" -> as reconstrucoes publicadas no HuggingFace
#               (so os modelos do paper), linkadas como {model}_published_all_*.pt
#               para a tabela sair com outro nome
#
# Tabela ja existente e pulada; FORCE=1 recalcula.
set -e
set -o pipefail
source "$(dirname "$0")/common.sh"

MODEL_NAME="$1"
KIND="${2:-enhanced}"
[ -z "$MODEL_NAME" ] && { sed -n '2,12p' "$SELF"; exit 1; }


case "$KIND" in
    base)           SUFIXO=all_recons ;;
    enhanced)       SUFIXO=all_enhancedrecons ;;
    published)      SUFIXO=published_all_enhancedrecons ;;
    published-base) SUFIXO=published_all_recons ;;
    *) echo "ERRO: tipo desconhecido: $KIND"; exit 1 ;;
esac
RECONS="$EVALS/$MODEL_NAME/${MODEL_NAME}_${SUFIXO}.pt"
if [ "${KIND#published}" != "$KIND" ] && [ ! -e "$RECONS" ]; then
    PUB="$DATA/evals/$MODEL_NAME/${MODEL_NAME}_${SUFIXO#published_}.pt"
    [ -f "$PUB" ] && ln -s "$PUB" "$RECONS"
fi
[ -f "$RECONS" ] || { echo "ERRO: $RECONS nao existe -- rode o recon antes"; exit 1; }

TABELA="$TABLES/${MODEL_NAME}_${SUFIXO}"
if [ "${FORCE:-0}" != "1" ] && [ -f "$TABELA.csv" ] && [ -f "${TABELA}_caption_metrics.csv" ]; then
    echo "=== $TABELA.csv ja existe, pulando (FORCE=1 recalcula) ==="; exit 0
fi

LOGDIR=$TRAIN_LOGS/$MODEL_NAME; mkdir -p "$LOGDIR"
echo "=== final_evaluations: $MODEL_NAME ($KIND) ==="
$ENVP/bin/python -u final_evaluations.py \
    --model_name="$MODEL_NAME" --all_recons_path="$RECONS" \
    --data_path=$DATA --cache_dir=$DATA --subj=1 \
    2>&1 | tee -a "$LOGDIR/evals_$KIND.log"

echo "=== tabela salva em $TABLES/ ==="
ls -la $TABLES/ 2>/dev/null | tail -5
