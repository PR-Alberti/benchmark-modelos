#!/bin/bash
# Gera tudo o que o benchmark (benchmark/) precisa: treina o que falta,
# reconstroi, avalia e monta a pagina. Uma GPU, tudo em sequencia.
#
#   scripts/run_benchmark.sh            # ~50 h numa A4500
#   tail -f logs/benchmark.log # uma linha por etapa
#
# Pode ser interrompido e rodado de novo: treinos retomam do last.pth,
# reconstrucoes do parcial, e o que ja terminou e pulado.
#
# Experimentos (todos subj01, fine-tune so da ridge a partir do multi-sujeito):
#   subj01_ridgeonly_1sess_prior        1024, com prior, 1 sessao
#   subj01_ridgeonly_1sess_4096blurry   4096 + blurry, com prior, 1 sessao
#   subj01_ridgeonly_1sess              1024, sem prior, 1 sessao
#   subj01_ridgeonly_40sess_prior       1024, com prior, 40 sessoes
# Referencias (fine-tune completo, checkpoints publicados):
#   final_subj01_pretrained_1sess_24bs  e  final_subj01_pretrained_40sess_24bs
set -e
set -o pipefail
source "$(dirname "$0")/common.sh"

mkdir -p $LOGS
LOG=$LOGS/benchmark.log

etapa() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
trap 'etapa "FALHOU (linha $LINENO) -- veja os logs em $TRAIN_LOGS/<modelo>/"' ERR

# treina <modelo> [VAR=valor ...]; o log completo fica em train_logs/<modelo>/train.log
treina() {
    local m=$1; shift
    if [ -f "$TRAIN_LOGS/$m/.treino_completo" ]; then
        etapa "$m: treino ja completo"; return
    fi
    etapa "$m: treinando ($*)"
    env "$@" RESUME=1 MODEL_NAME="$m" "$SCRIPTS/run_ridgeonly_prior.sh" > /dev/null 2>> "$LOG"
    touch "$TRAIN_LOGS/$m/.treino_completo"
    etapa "$m: treino completo"
}

# reconstroi <modelo> <hidden_dim> <blurry|noblurry> [VAR=valor ...]
reconstroi() {
    local m=$1 hd=$2 bl=$3; shift 3
    etapa "$m: reconstrucoes"
    env "$@" "$SCRIPTS/run_recon.sh" "$m" "$hd" "$bl" > /dev/null 2>> "$LOG"
}

# avalia <modelo> <base|enhanced|published|published-base>
avalia() {
    etapa "$1: metricas ($2)"
    "$SCRIPTS/run_evals.sh" "$1" "$2" > /dev/null 2>> "$LOG"
}

# retrieval <modelo> <hidden_dim> <blurry|noblurry>: so clipvoxels + retrieval,
# sem difusao (minutos, em vez das horas de uma reconstrucao)
retrieval() {   # 4o argumento opcional: ckpt de onde vem backbone e prior
    local m=$1 hd=$2 bf=--no-blurry_recon fz=""
    [ "$3" = blurry ] && bf=--blurry_recon
    [ -n "${4:-}" ] && fz="--frozen_ckpt=$4"
    if [ -f "$EVALS/$m/${m}_retrieval.json" ]; then etapa "$m: retrieval ja medido"; return; fi
    etapa "$m: retrieval"
    $ENVP/bin/python verify_retrieval.py --model_name="$m" --data_path="$DATA" \
        --cache_dir="$DATA" --subj=1 --hidden_dim="$hd" --n_blocks=4 $bf $fz --new_test \
        >> "$TRAIN_LOGS/$m/retrieval.log" 2>&1
}

# frr <modelo> <sessoes> [opcoes do run_frr.py]: baseline linear (sem rede, sem reconstrucao);
# so retrieval e similaridade com o embedding CLIP. 1 sessao leva ~1 min; 40, ~1 h
frr() {
    local m=$1 ns=$2; shift 2
    if [ -f "$TABLES/${m}_frr.json" ]; then etapa "$m: FRR ja medido"; return; fi
    etapa "$m: FRR ($ns sessoes)"
    env NUM_SESSIONS=$ns MODEL_NAME=$m EXTRA="$*" "$SCRIPTS/run_frr.sh" > /dev/null 2>> "$LOG"
}

# os checkpoints do paper ficam nos dados; os scripts procuram em train_logs
for m in final_subj01_pretrained_1sess_24bs final_subj01_pretrained_40sess_24bs; do
    mkdir -p "$TRAIN_LOGS/$m"
    ln -sfn "$DATA/train_logs/$m/last.pth" "$TRAIN_LOGS/$m/last.pth"
done

etapa "=== inicio ==="

# 1024 + prior: reconstrucoes vieram do release; so recalcula as metricas
avalia subj01_ridgeonly_1sess_prior base
avalia subj01_ridgeonly_1sess_prior enhanced

# FRR: o principal e o que o plano especifica (Doerig et al.: fracoes de 0,05 a 1, uma por
# dimensao). As variantes medem a sensibilidade: uma fracao so para todas as dimensoes (reaproveita
# as dobras da CV do principal) e a grade estendida a fracoes menores.
frr subj01_frr_1sess 1
frr subj01_frr_1sess_global 1 --global_fraction --cv_dir $TRAIN_LOGS/subj01_frr_1sess/frr_cv
frr subj01_frr_1sess_ext 1 --grid extended
frr subj01_frr_40sess 40
frr subj01_frr_40sess_global 40 --global_fraction --cv_dir $TRAIN_LOGS/subj01_frr_40sess/frr_cv
frr subj01_frr_40sess_ext 40 --grid extended

# treinos primeiro: sao os passos com mais risco (memoria), melhor falhar cedo
treina subj01_ridgeonly_1sess_4096blurry BLURRY=1 HIDDEN_DIM=4096 CKPT_INTERVAL=10
treina subj01_ridgeonly_40sess_prior NUM_SESSIONS=40 NUM_EPOCHS=20 CKPT_INTERVAL=1 FROZEN=1

# Nos ridge-only, backbone e prior sao os do ckpt de partida (FROZEN_CKPT):
# o sem prior nao tem prior no ckpt, e os treinados com congelados em fp16
# guardaram esses pesos arredondados. Do modelo treinado so vem a ridge.
MS1024="$DATA/train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth"
MS4096="$DATA/train_logs/final_multisubject_subj01/last.pth"

reconstroi subj01_ridgeonly_1sess 1024 noblurry FROZEN_CKPT="$MS1024"
avalia subj01_ridgeonly_1sess base
avalia subj01_ridgeonly_1sess enhanced

reconstroi final_subj01_pretrained_1sess_24bs 4096 blurry
avalia final_subj01_pretrained_1sess_24bs base
avalia final_subj01_pretrained_1sess_24bs enhanced
avalia final_subj01_pretrained_1sess_24bs published

reconstroi subj01_ridgeonly_1sess_4096blurry 4096 blurry FROZEN_CKPT="$MS4096"
avalia subj01_ridgeonly_1sess_4096blurry base
avalia subj01_ridgeonly_1sess_4096blurry enhanced

reconstroi subj01_ridgeonly_40sess_prior 1024 noblurry FROZEN_CKPT="$MS1024"
avalia subj01_ridgeonly_40sess_prior base
avalia subj01_ridgeonly_40sess_prior enhanced

reconstroi final_subj01_pretrained_40sess_24bs 4096 blurry
avalia final_subj01_pretrained_40sess_24bs base
avalia final_subj01_pretrained_40sess_24bs enhanced
avalia final_subj01_pretrained_40sess_24bs published
avalia final_subj01_pretrained_40sess_24bs published-base

etapa "montando a pagina"
$ENVP/bin/python make_benchmark.py >> "$LOG" 2>&1
etapa "pagina montada: benchmark/index.html"

# ---------------------------------------------------------------------------
# Ruido de referencia. O treino e deterministico com a mesma semente (o 4096 +
# blurry retreinado repetiu a corrida anterior epoca por epoca), entao repetir
# uma corrida nao mede ruido. Aqui muda so a semente, na config 1024 + prior:
#   treino      sementes 42 (retreino com o codigo atual), 1 e 2 -> retrieval
#   amostragem  o mesmo modelo reconstruido com a semente 7 -> todas as metricas
# ---------------------------------------------------------------------------
for s in 42 1 2; do
    treina subj01_ridgeonly_1sess_prior_seed$s SEED=$s
    retrieval subj01_ridgeonly_1sess_prior_seed$s 1024 noblurry
done
# controle do sem prior: mesmo script e semente do retreino acima, so sem o prior
# (os dois ckpts do release vieram de scripts e versoes de codigo diferentes)
treina subj01_ridgeonly_1sess_noprior_seed42 PRIOR=0 SEED=42
retrieval subj01_ridgeonly_1sess_noprior_seed42 1024 noblurry "$MS1024"
m=subj01_ridgeonly_1sess_prior_rseed7
mkdir -p "$TRAIN_LOGS/$m"
ln -sfn ../subj01_ridgeonly_1sess_prior/last.pth "$TRAIN_LOGS/$m/last.pth"
reconstroi "$m" 1024 noblurry SEED=7
avalia "$m" base
avalia "$m" enhanced

etapa "montando a pagina (com o ruido de referencia)"
$ENVP/bin/python make_benchmark.py >> "$LOG" 2>&1
etapa "=== fim: benchmark/index.html ==="
