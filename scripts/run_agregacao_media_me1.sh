#!/bin/bash
# Media das repeticoes no treino (branch agregation), fila do ME1: 1, 5, 10, 20 e 40 sessoes, com e sem low-level.
# Experimentos em src/experimentos.py (sufixo _media). A GPU de 20 GB so comporta uma fila por vez:
# rode a do ME1 (run_agregacao_media_me1.sh) depois que a do ME2 terminar.
#
#   scripts/run_agregacao_media_me1.sh                 as 4 quantidades de sessoes, da menor para a maior
#   SESSOES="5 10" scripts/run_agregacao_media_me1.sh  so algumas
#
# Pode ser interrompido e rodado de novo: etapas concluidas sao puladas e o treino retoma do last.pth.
set -o pipefail
source "$(dirname "$0")/common.sh"
unset DATASET
export SUFIXO=_media
LOG="$LOGS/agregacao_media_me1.log"; mkdir -p "$LOGS"
FALHAS=0
etapa() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
etapa "=== inicio ME1: sessoes [${SESSOES:-1 5 10 20 40}] ==="
for n in ${SESSOES:-1 5 10 20 40}; do
    livre=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
    [ "$livre" -ge "${MIN_LIVRE_GB:-60}" ] || { etapa "PARADO antes de $n sessoes: $livre GB livres"; exit 1; }
    etapa "$n sessoes ($livre GB livres)"
    SESSOES=$n "$SCRIPTS/run_me1_sessoes.sh" || { FALHAS=$((FALHAS + 1)); etapa "$n sessoes FALHOU -- veja $LOGS/me1_benchmark.log.err"; }
    git -C "$REPO" add results/site results/tables >> "$LOG" 2>&1
    git -C "$REPO" diff --cached --quiet || git -C "$REPO" commit -q -m "Media das repeticoes: ME1 com $n sessoes, resultados e site" >> "$LOG" 2>&1 \
        || etapa "$n sessoes: nao consegui commitar (veja $LOG)"
done
etapa "=== fim ME1: $FALHAS falha(s) ==="
[ "$FALHAS" = 0 ]
