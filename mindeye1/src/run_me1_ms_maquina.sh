#!/bin/bash
# Protocolos multi-sujeito do MindEye1: uma maquina = um pre-treino + os 5 ajustes no subj01.
# Ordem de servico completa em docs/PROCEDIMENTO_ME1_MULTISUJ.md.
#
#   mindeye1/src/run_me1_ms_maquina.sh --lista          as 6 maquinas e o que ja terminou nesta
#   mindeye1/src/run_me1_ms_maquina.sh 3                tudo da maquina 3
#   SMOKE=1 mindeye1/src/run_me1_ms_maquina.sh 3        teste de fumaca (3 epocas de 2 passos, nomes _smoke)
#   FT_SESSOES="1 5" mindeye1/src/run_me1_ms_maquina.sh 3   so alguns ajustes
#
# Variaveis: DATA (padrao ~/mindeyev2, formato do MindEye2), PY (padrao ~/envs/fmri/bin/python),
# OUT (padrao ~/me1_multisubj; checkpoints em $OUT/train_logs), MAX_RESTARTS (5), COMMITA (1 = commit e push
# dos resultados desta maquina em results/me1_multisubj/ ao fim de cada fase).
# Pode ser interrompido e rodado de novo: fase com final_retrieval.json e pulada, fase pela metade retoma do last.pth.
set -u
cd "$(dirname "$0")"
SRC=$(pwd -P); REPO=$(cd ../.. && pwd -P)
DATA=${DATA:-${MINDEYE_DATA:-$HOME/mindeyev2}}
PY=${PY:-${MINDEYE_ENV:-$HOME/envs/fmri}/bin/python}
OUT=${OUT:-$HOME/me1_multisubj}
MAX_RESTARTS=${MAX_RESTARTS:-5}
COMMITA=${COMMITA:-1}
SMOKE=${SMOKE:-0}
FT_SESSOES=${FT_SESSOES:-1 5 10 20 40}

# maquina -> "nome | sujeitos do pre-treino | sessoes por sujeito | batch global"
config() {
    case "$1" in
        1) echo "2subj_10sess|2 5|10|24" ;;
        2) echo "2subj_20sess|2 5|20|24" ;;
        3) echo "4subj_5sess|2 5 7 3|5|24" ;;
        4) echo "4subj_10sess|2 5 7 3|10|24" ;;
        5) echo "7subj_1sess|2 3 4 5 6 7 8|1|21" ;;
        6) echo "7subj_5sess|2 3 4 5 6 7 8|5|21" ;;
        *) return 1 ;;
    esac
}

feito() { [ -f "$OUT/train_logs/$1/final_retrieval.json" ] || [ -f "$REPO/results/me1_multisubj/$1/final_retrieval.json" ]; }

if [ "${1:-}" = "--lista" ] || [ -z "${1:-}" ]; then
    printf '%-8s %-14s %-16s %-6s %s\n' maquina pre-treino sujeitos sessoes "feito (pre + ajustes subj01 1/5/10/20/40)"
    for m in 1 2 3 4 5 6; do
        IFS='|' read -r nome subjs nsess bs <<< "$(config $m)"
        st=""; feito "me1ms_pre_${nome}" && st="pre" || st="-"
        for M in 1 5 10 20 40; do feito "me1ms_ft_${nome}__subj01_${M}sess" && st="$st ${M}" || st="$st ."; done
        printf '%-8s %-14s %-16s %-6s %s\n' "$m" "$nome" "$subjs" "$nsess" "$st"
    done
    [ -z "${1:-}" ] && { echo; echo "uso: $0 <maquina 1-6> | --lista"; exit 1; }
    exit 0
fi

MAQ=$1
CFG=$(config "$MAQ") || { echo "maquina $MAQ nao existe: use 1 a 6 (--lista)" >&2; exit 1; }
IFS='|' read -r NOME SUBJS NSESS BS <<< "$CFG"
SUF=""; EXTRA=()
if [ "$SMOKE" = 1 ]; then SUF=_smoke; EXTRA=(--num_epochs=3 --max_iters_per_epoch=2 --ckpt_interval=1); COMMITA=0; fi
PRE=me1ms_pre_${NOME}${SUF}
mkdir -p "$OUT/train_logs" "$REPO/results/logs"
LOG=$REPO/results/logs/me1_multisubj_maquina${MAQ}${SUF}.log

etapa() { echo "[$(date '+%F %T')] maquina $MAQ: $*" | tee -a "$LOG"; }

# guarda os resultados leves (sem last.pth) no repositorio e sobe; cada maquina so escreve na propria pasta
guarda() {
    local name=$1 dst=$REPO/results/me1_multisubj/$1
    mkdir -p "$dst"
    cp "$OUT/train_logs/$name/"{args.json,log.jsonl,final_retrieval.json} "$dst/" 2>/dev/null
    cp "$OUT/train_logs/$name/stdout.log" "$dst/" 2>/dev/null
    [ "$COMMITA" = 1 ] || return 0
    ( cd "$REPO" && git add "results/me1_multisubj/$name" "results/logs/$(basename "$LOG")" \
        && git commit -q -m "ME1 multi-sujeito, maquina $MAQ: resultados de $name" \
        && { git pull -q --rebase --autostash && git push -q || etapa "push falhou (o commit ficou local; rode git push depois)"; } )
}

fase() { # fase <nome> <argumentos...>
    local name=$1 t0=$SECONDS; shift
    if [ -f "$OUT/train_logs/$name/final_retrieval.json" ]; then etapa "$name ja terminado"; return 0; fi
    mkdir -p "$OUT/train_logs/$name"
    for attempt in $(seq 0 "$MAX_RESTARTS"); do
        etapa "$name: tentativa $attempt"
        if "$PY" Train_MindEye_multisubject.py --model_name="$name" --data_path="$DATA" --logs_path="$OUT/train_logs" \
            --subj=1 --hidden_dim=1024 --n_blocks=4 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --prior_scale=30 \
            --clip_scale=1 --ckpt_interval=5 "$@" "${EXTRA[@]}" >> "$OUT/train_logs/$name/stdout.log" 2>&1; then
            etapa "$name: concluido em $(( (SECONDS - t0) / 60 )) min -- $(tr -d '\n ' < "$OUT/train_logs/$name/final_retrieval.json" | cut -c1-160)"
            guarda "$name"
            return 0
        fi
        etapa "$name: caiu (veja $OUT/train_logs/$name/stdout.log)"; sleep 60
    done
    etapa "$name: desisti depois de $MAX_RESTARTS tentativas"; return 1
}

etapa "config $NOME: pre-treino nos sujeitos [$SUBJS], $NSESS sessoes de cada, batch $BS; ajustes no subj01: $FT_SESSOES"
# shellcheck disable=SC2086
fase "$PRE" --multi_subject --pretrain_subjs $SUBJS --pretrain_sessions="$NSESS" --batch_size="$BS" --num_sessions=40 \
    || exit 1
FALHAS=0
for M in $FT_SESSOES; do
    fase "me1ms_ft_${NOME}__subj01_${M}sess${SUF}" --no-multi_subject --num_sessions="$M" --batch_size=24 \
        --multisubject_ckpt="$OUT/train_logs/$PRE" || FALHAS=$((FALHAS + 1))
done
etapa "fim: $FALHAS falha(s)"
[ "$FALHAS" = 0 ] && etapa "MAQUINA $MAQ: TUDO PRONTO"
exit "$FALHAS"
