#!/bin/bash
# MindEye1 em tamanhos intermediarios de sessoes (5, 10 e 20), com e sem o low-level, nas mesmas
# condicoes do benchmark (as N primeiras sessoes, teste de 1.000 imagens, mesmas metricas).
#
#   scripts/run_me1_sessoes.sh                  5, 10 e 20 sessoes
#   SESSOES="5" scripts/run_me1_sessoes.sh      so alguns tamanhos
#
# Por tamanho N, nesta ordem (as entradas estao em src/experimentos.py, EXPERIMENTOS):
#   1. alto nivel, pelo src/treina.py (configuracao, commit, manifesto e curva em train_logs/)
#   2. low-level, no mesmo manifesto
#   3. reconstrucao e metricas COM o low-level        -> subj01_me1_<N>sess
#   4. reconstrucao e metricas SEM o low-level        -> subj01_me1_<N>sess_semlowlevel
# O alto nivel nao e treinado de novo para o "sem low-level": e o mesmo checkpoint, e so a reconstrucao
# muda (sem o img2img a partir da imagem borrada); isso poupa um treino por tamanho.
#
# Pode ser interrompido e rodado de novo: etapa concluida e pulada (marcador) e treino pela metade
# retoma do last.pth. Uma linha por etapa em results/logs/me1_benchmark.log, o mesmo do
# run_me1_benchmark.sh (o site tira o tempo de treino dali).
set -e
set -o pipefail
source "$(dirname "$0")/common.sh"
unset DATASET
LOG="$LOGS/me1_benchmark.log"; mkdir -p "$LOGS"
ME1_LOGS="$REPO/mindeye1/train_logs"
etapa() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

roda() {   # roda <marcador> <descricao> <comando...>
    local marca=$1 desc=$2; shift 2
    if [ -f "$marca" ]; then etapa "$desc: ja feito"; return; fi
    etapa "$desc: comecando"
    "$@" > /dev/null 2>> "$LOG.err"
    mkdir -p "$(dirname "$marca")"; touch "$marca"
    etapa "$desc: feito"
}

for N in ${SESSOES:-5 10 20}; do
    M=subj01_me1_${N}sess; S=${M}_semlowlevel; AE=subj01_me1_lowlevel_${N}sess
    roda "$ME1_LOGS/$M/.completo"         "$M: treino do alto nivel (240 epocas)" \
        "$ENVP/bin/python" treina.py "$M"
    roda "$ME1_LOGS/models/$AE/.completo" "$AE: treino do low-level (120 epocas)" \
        env NUM_SESSIONS=$N AE_NAME=$AE DATASET="$REPO/train_logs/$M/dataset.json" MODEL_NAME=$M \
            "$SCRIPTS/me1_run.sh" lowlevel
    roda "$EVALS/$M/.recon_completo"      "$M: reconstrucoes (1.000 imagens)" \
        env NUM_SESSIONS=$N MODEL_NAME=$M AE_NAME=$AE "$SCRIPTS/me1_run.sh" recon
    roda "$EVALS/$M/.metricas_completas"  "$M: metricas" \
        env NUM_SESSIONS=$N MODEL_NAME=$M "$SCRIPTS/me1_run.sh" metrics
    # sem o low-level: o checkpoint e o de $M; so o nome da saida muda (a recon aceita --out_name depois)
    roda "$EVALS/$S/.recon_completo"      "$S: reconstrucoes sem low-level (1.000 imagens)" \
        env NUM_SESSIONS=$N MODEL_NAME=$M "$SCRIPTS/me1_run.sh" recon \
            --autoencoder_name=None --img2img_strength=1 --out_name="$S"
    roda "$EVALS/$S/.metricas_completas"  "$S: metricas" \
        env NUM_SESSIONS=$N MODEL_NAME=$S "$SCRIPTS/me1_run.sh" metrics
done
etapa "MindEye1 em sessoes intermediarias: tudo pronto"
