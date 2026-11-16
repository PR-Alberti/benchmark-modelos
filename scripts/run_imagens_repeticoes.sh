#!/bin/bash
# MindEye2 ridge-only com imagens unicas x repeticoes a 6.000 exibicoes fixas (branch Image-per-repetition):
# 6.000 imagens x 1, 3.000 x 2 e 2.000 x 3, com trials separados e com a media das repeticoes, com e sem
# low-level. Os experimentos estao em src/experimentos.py (IMAGENS_REPETICOES); cada um e treinado pelo
# src/treina.py, depois vem o retrieval com intervalo de 95% (verify_retrieval.py), a reconstrucao e as
# metricas, como no scripts/run_sessoes_intermediarias.sh.
#
#   scripts/run_imagens_repeticoes.sh                        os 10 modelos
#   CONFIGS="6000img_1rep 3000img_2rep_media" ...            so algumas configuracoes
#   LOWLEVEL="0" ...                                         so sem low-level (1024); "1" so com (4096 + blurry)
#   RECON=0 ...                                              so treino e retrieval, sem reconstrucao
#   COMMITA=1 ...                                            um commit com os resultados de cada modelo
#
# Pode ser interrompido e rodado de novo: etapa concluida deixa um marcador em
# train_logs/<modelo>/.imgrep_<etapa> e e pulada; o treino retoma do last.pth. A falha de um modelo nao
# para a fila (uma linha por etapa em results/logs/imagens_repeticoes.log).
set -o pipefail
source "$(dirname "$0")/common.sh"
# estes treinos definem o dataset pelo proprio experimento: um DATASET exportado no shell nao pode vazar
unset DATASET

mkdir -p "$LOGS"
LOG=$LOGS/imagens_repeticoes.log
CONFIGS=${CONFIGS:-6000img_1rep 3000img_2rep_exibicoes 3000img_2rep_media 2000img_3rep_exibicoes 2000img_3rep_media}
LOWLEVEL=${LOWLEVEL:-0 1}
RECON=${RECON:-1}
MS1024="$DATA/train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth"
MS4096="$DATA/train_logs/final_multisubject_subj01/last.pth"
FALHAS=0

etapa() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# passo <modelo> <etapa> <descricao> <comando...>: pula se o marcador existe, cria quando o comando acaba
passo() {
    local m=$1 e=$2 d=$3 marca="$TRAIN_LOGS/$1/.imgrep_$2" t0=$SECONDS; shift 3
    if [ -f "$marca" ]; then etapa "$m: $d ja feito"; return 0; fi
    mkdir -p "$TRAIN_LOGS/$m"
    etapa "$m: $d"
    if "$@" >> "$TRAIN_LOGS/$m/imagens_repeticoes.log" 2>&1; then
        touch "$marca"
        etapa "$m: $d concluido em $(( (SECONDS - t0) / 60 )) min"
    else
        etapa "$m: $d FALHOU -- veja $TRAIN_LOGS/$m/imagens_repeticoes.log"
        return 1
    fi
}

treina() {
    local m=$1 r=""
    [ -f "$TRAIN_LOGS/$m/last.pth" ] && r=--retoma
    passo "$m" treino "treino" "$ENVP/bin/python" treina.py "$m" $r
}

# modelo <nome> <hidden_dim> <blurry|noblurry> <ckpt de partida>
modelo() {
    local m=$1 hd=$2 bl=$3 fz=$4 bf=--no-blurry_recon
    [ "$bl" = blurry ] && bf=--blurry_recon
    treina "$m" || { FALHAS=$((FALHAS + 1)); etapa "$m: interrompido, a fila segue"; return 0; }
    passo "$m" retrieval "retrieval com intervalo" cronometra "$m" retrieval "$ENVP/bin/python" etapas/verify_retrieval.py --model_name="$m" \
        --data_path="$DATA" --cache_dir="$DATA" --subj=1 --hidden_dim="$hd" --n_blocks=4 $bf \
        --frozen_ckpt="$fz" --new_test || FALHAS=$((FALHAS + 1))
    if [ "$RECON" = 1 ]; then
        passo "$m" recon "reconstrucoes" env FROZEN_CKPT="$fz" "$SCRIPTS/run_recon.sh" "$m" "$hd" "$bl" \
            && passo "$m" evals_base "metricas (base)" "$SCRIPTS/run_evals.sh" "$m" base \
            && passo "$m" evals_enhanced "metricas (enhanced)" "$SCRIPTS/run_evals.sh" "$m" enhanced \
            || { FALHAS=$((FALHAS + 1)); etapa "$m: reconstrucao ou metricas interrompidas, a fila segue"; }
    fi
    guarda "$m"
}

# guarda <modelo>: com COMMITA=1, um commit com o que o modelo acrescentou aos resultados
guarda() {
    [ "${COMMITA:-0}" = 1 ] || return 0
    git -C "$REPO" add results/site results/tables >> "$LOG" 2>&1
    git -C "$REPO" diff --cached --quiet && return 0
    git -C "$REPO" commit -q -m "Imagens x repeticoes: resultados de $1" >> "$LOG" 2>&1 \
        || etapa "$1: nao consegui commitar os resultados (veja $LOG)"
}

etapa "=== inicio: configuracoes [$CONFIGS], low-level [$LOWLEVEL], recon $RECON ==="
for c in $CONFIGS; do
    for ll in $LOWLEVEL; do
        if [ "$ll" = 1 ]; then modelo "subj01_me2_${c}_4096blurry" 4096 blurry "$MS4096"
        else modelo "subj01_me2_${c}_prior" 1024 noblurry "$MS1024"; fi
    done
done

atualiza_site
guarda "o site"
etapa "=== fim: $FALHAS falha(s) ==="
