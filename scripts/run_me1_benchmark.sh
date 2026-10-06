#!/bin/bash
# MindEye1 no benchmark: treina, reconstroi e avalia com 1 e com 40 sessoes, nas mesmas exibicoes
# de treino e no mesmo teste dos outros modelos (docs/MINDEYE1.md, "No benchmark").
#
#   scripts/run_me1_benchmark.sh              tudo, 1 sessao primeiro (~37 h numa A4500)
#   SESSOES="1" scripts/run_me1_benchmark.sh  so 1 sessao (~5 h)
#
# Pode ser interrompido e rodado de novo: etapa concluida e pulada (marcador .completo) e treino
# pela metade retoma do last.pth. Uma linha por etapa em logs/me1_benchmark.log.
set -e
set -o pipefail
source "$(dirname "$0")/common.sh"
# o benchmark e o dataset completo: um DATASET exportado no shell nao pode vazar para estes treinos
unset DATASET
LOG="$LOGS/me1_benchmark.log"; mkdir -p "$LOGS"
ME1_LOGS="$REPO/mindeye1/train_logs"
etapa() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# roda <marcador> <descricao> <comando...>: pula se o marcador existe, cria quando o comando acaba
roda() {
    local marca=$1 desc=$2; shift 2
    if [ -f "$marca" ]; then etapa "$desc: ja feito"; return; fi
    etapa "$desc: comecando"
    "$@" > /dev/null 2>> "$LOG.err"
    mkdir -p "$(dirname "$marca")"; touch "$marca"
    etapa "$desc: feito"
}

for N in ${SESSOES:-1 40}; do
    M=subj01_me1_${N}sess; AE=subj01_me1_lowlevel_${N}sess
    # o last.pth so e gravado a cada SAVE_EVERY epocas; com 1 sessao a epoca leva ~30 s e cada
    # gravacao ~6 GB, entao grava a cada 10
    [ "$N" -le 1 ] && SE=10 || SE=1
    roda "$ME1_LOGS/$M/.completo"             "$M: treino do alto nivel (240 epocas)" \
        env NUM_SESSIONS=$N SAVE_EVERY=$SE "$SCRIPTS/me1_run.sh" train
    roda "$ME1_LOGS/models/$AE/.completo"     "$AE: treino do low-level (120 epocas)" \
        env NUM_SESSIONS=$N "$SCRIPTS/me1_run.sh" lowlevel
    roda "$EVALS/$M/.recon_completo"           "$M: reconstrucoes (1.000 imagens)" \
        env NUM_SESSIONS=$N "$SCRIPTS/me1_run.sh" recon
    roda "$EVALS/$M/.metricas_completas"       "$M: metricas" \
        env NUM_SESSIONS=$N "$SCRIPTS/me1_run.sh" metrics
done
etapa "MindEye1: tudo pronto"
