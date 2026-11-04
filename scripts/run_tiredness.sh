#!/bin/bash
# O decoding muda ao longo da sessao? MindEye2 ridge-only treinado nas 10 primeiras sessoes e avaliado,
# trial a trial e por run, nas 30 seguintes, sem os trials de imagens vistas no treino. O protocolo e a
# leitura dos resultados estao em docs/TIREDNESS.md; os resultados, em results/tables/tiredness/.
#
#   scripts/run_tiredness.sh                                  treina (~4 h numa A4500) e avalia
#   scripts/run_tiredness.sh <modelo>                         outro modelo de src/experimentos.py
#
# Pode ser interrompido e rodado de novo: o treino retoma do last.pth, e um modelo ja treinado so e avaliado.
set -o pipefail
source "$(dirname "$0")/common.sh"
# o treino define o dataset pelo proprio experimento: um DATASET exportado no shell nao pode vazar
unset DATASET

MODELO=${1:-subj01_ridgeonly_10sess_prior}
mkdir -p "$TRAIN_LOGS/$MODELO"

r=""
[ -f "$TRAIN_LOGS/$MODELO/last.pth" ] && r=--retoma
"$ENVP/bin/python" treina.py "$MODELO" $r || exit 1
"$ENVP/bin/python" etapas/tiredness_por_run.py --model_name="$MODELO" --data_path="$DATA" \
    2>&1 | tee "$TRAIN_LOGS/$MODELO/tiredness.log"
