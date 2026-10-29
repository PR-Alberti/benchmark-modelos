#!/bin/bash
# Refaz, numa maquina sem os checkpoints, os modelos do benchmark que o docs/REEXECUTAR.md lista.
# Cada modelo e treinado pelo src/treina.py com o dicionario de src/experimentos.py (REPRODUCOES), para
# ficarem registrados a configuracao, o commit, o manifesto do dataset, a curva e o tempo; depois vem
# o retrieval com intervalo de 95% (verify_retrieval.py), a reconstrucao e as metricas.
#
#   scripts/run_reexecutar.sh                     tudo, do mais barato ao mais caro (~55 h numa A4500)
#   SO="frr 1sess" scripts/run_reexecutar.sh      so alguns blocos: frr 1sess sementes 4096 artigo 40sess
#
# Os modelos refeitos se chamam repro_<nome do benchmark>: o treina.py nao sobrescreve um modelo do
# benchmark, e assim nenhuma tabela de results/tables/ e alterada; o original e a reproducao ficam
# lado a lado no site. O Ridge 1024 de 1 sessao (com e sem prior) refeito aqui e o treino com o codigo
# atual, e nao o checkpoint do release; com a semente 42, e a mesma corrida dos _prior_seed42 e
# _noprior_seed42 do benchmark, que por isso nao tem reproducao propria.
#
# COMMITA=1 guarda no git os resultados de cada modelo quando ele termina (results/site e
# results/tables). Sem isso o site regenerado deixa a arvore com alteracoes, e o treina.py registra o
# commit dos treinos seguintes como "<hash>+alteracoes", sem dar para saber se o codigo mudou.
#
# Pode ser interrompido e rodado de novo: etapa concluida deixa um marcador em
# train_logs/<modelo>/.reexec_<etapa> e e pulada; o treino do MindEye2 retoma do last.pth e o FRR da
# ultima dobra. A falha de um modelo nao para a fila: os outros seguem, e a falha fica no log.
# Uma linha por etapa em results/logs/reexecutar.log.
set -o pipefail
source "$(dirname "$0")/common.sh"
# o benchmark e o dataset completo: um DATASET exportado no shell nao pode vazar para estes treinos
unset DATASET

mkdir -p "$LOGS"
LOG=$LOGS/reexecutar.log
SO=${SO:-frr 1sess sementes 4096 artigo 40sess}
MS1024="$DATA/train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth"
MS4096="$DATA/train_logs/final_multisubject_subj01/last.pth"
FALHAS=0

etapa() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# passo <modelo> <etapa> <descricao> <comando...>: pula se o marcador existe, cria quando o comando acaba
passo() {
    local m=$1 e=$2 d=$3 marca="$TRAIN_LOGS/$1/.reexec_$2" t0=$SECONDS; shift 3
    if [ -f "$marca" ]; then etapa "$m: $d ja feito"; return 0; fi
    mkdir -p "$TRAIN_LOGS/$m"
    etapa "$m: $d"
    if "$@" >> "$TRAIN_LOGS/$m/reexecutar.log" 2>&1; then
        touch "$marca"
        etapa "$m: $d concluido em $(( (SECONDS - t0) / 60 )) min"
    else
        etapa "$m: $d FALHOU -- veja $TRAIN_LOGS/$m/reexecutar.log"
        return 1
    fi
}

# o treina.py pula o que ja terminou; --retoma so vale para o MindEye2 com um last.pth no disco
treina() {
    local m=$1 r=""
    [ -f "$TRAIN_LOGS/$m/last.pth" ] && r=--retoma
    passo "$m" treino "treino" "$ENVP/bin/python" treina.py "$m" $r
}

# retrieval <modelo> <hidden_dim> <blurry|noblurry> [ckpt de onde vem backbone e prior]: sem difusao.
# O retrieval tambem sai nas tabelas (sem o intervalo), entao a falha aqui nao interrompe o modelo.
retrieval() {
    local m=$1 hd=$2 bf=--no-blurry_recon fz=""
    [ "$3" = blurry ] && bf=--blurry_recon
    [ -n "${4:-}" ] && fz="--frozen_ckpt=$4"
    passo "$m" retrieval "retrieval com intervalo" "$ENVP/bin/python" etapas/verify_retrieval.py --model_name="$m" \
        --data_path="$DATA" --cache_dir="$DATA" --subj=1 --hidden_dim="$hd" --n_blocks=4 $bf $fz --new_test \
        || FALHAS=$((FALHAS + 1))
    return 0
}

# reconstroi <modelo> <hidden_dim> <blurry|noblurry> [VAR=valor ...]
reconstroi() {
    local m=$1 hd=$2 bl=$3; shift 3
    passo "$m" recon "reconstrucoes" env "$@" "$SCRIPTS/run_recon.sh" "$m" "$hd" "$bl"
}

avalia() {
    passo "$1" "evals_$2" "metricas ($2)" "$SCRIPTS/run_evals.sh" "$1" "$2"
}

# modelo <nome> <hidden_dim> <blurry|noblurry> [ckpt de partida]: treino, retrieval, reconstrucao, metricas
modelo() {
    local m=$1 hd=$2 bl=$3 fz=${4:-}
    treina "$m" && retrieval "$m" "$hd" "$bl" "$fz" \
        && reconstroi "$m" "$hd" "$bl" ${fz:+FROZEN_CKPT="$fz"} \
        && avalia "$m" base && avalia "$m" enhanced \
        || { FALHAS=$((FALHAS + 1)); etapa "$m: interrompido, a fila segue"; }
    guarda "$m"
}

# guarda <modelo>: com COMMITA=1, um commit com o que o modelo acrescentou aos resultados
guarda() {
    [ "${COMMITA:-0}" = 1 ] || return 0
    git -C "$REPO" add results/site results/tables >> "$LOG" 2>&1
    git -C "$REPO" diff --cached --quiet && return 0
    git -C "$REPO" commit -q -m "Reexecucao: resultados de $1" >> "$LOG" 2>&1 \
        || etapa "$1: nao consegui commitar os resultados (veja $LOG)"
}

quer() { [[ " $SO " == *" $1 "* ]]; }

etapa "=== inicio: $SO ==="

if quer frr; then
    for m in frr_1sess frr_1sess_global frr_1sess_ext frr_40sess frr_40sess_global frr_40sess_ext; do
        treina "repro_subj01_$m" || FALHAS=$((FALHAS + 1))
        guarda "repro_subj01_$m"
    done
fi

if quer 1sess; then
    # com prior e em fp32, backbone e prior do proprio checkpoint servem; o sem prior nao tem prior
    modelo repro_subj01_ridgeonly_1sess_prior 1024 noblurry
    modelo repro_subj01_ridgeonly_1sess 1024 noblurry "$MS1024"
fi

# Semente: a unica de treino e a 42, a mesma em todos os modelos. Este bloco (1 e 2) e a excecao, feita
# uma vez para medir o ruido do treino; nao acrescente sementes a outros modelos.
if quer sementes; then      # ruido do treino: agora com todas as metricas, e nao so o retrieval
    for s in 1 2; do modelo repro_subj01_ridgeonly_1sess_prior_seed$s 1024 noblurry; done
fi

if quer 4096; then
    modelo repro_subj01_ridgeonly_1sess_4096blurry 4096 blurry "$MS4096"
fi

if quer artigo; then        # treinados pelos autores: so falta o intervalo do retrieval
    for m in final_subj01_pretrained_1sess_24bs final_subj01_pretrained_40sess_24bs; do
        mkdir -p "$TRAIN_LOGS/$m"
        ln -sfn "$DATA/train_logs/$m/last.pth" "$TRAIN_LOGS/$m/last.pth"
        retrieval "$m" 4096 blurry
        guarda "$m"
    done
fi

if quer 40sess; then
    modelo repro_subj01_ridgeonly_40sess_prior 1024 noblurry "$MS1024"
    modelo repro_subj01_ridgeonly_40sess_4096blurry 4096 blurry "$MS4096"
fi

atualiza_site
guarda "o site"
etapa "=== fim: $FALHAS falha(s) ==="
