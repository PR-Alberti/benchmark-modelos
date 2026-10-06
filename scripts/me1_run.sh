#!/bin/bash
# Roda uma etapa do MindEye1 (subj01, alto nivel em CLIP ViT-L/14 257x768, Versatile Diffusion).
#
#   scripts/me1_run.sh train      [args extras]   treino do alto nivel  -> mindeye1/train_logs/$MODEL_NAME
#   scripts/me1_run.sh lowlevel   [args extras]   treino do low-level   -> mindeye1/train_logs/models/$AE_NAME
#   scripts/me1_run.sh retrieval                  retrieval no teste (30 sorteios de 300)
#   scripts/me1_run.sh recon      [args extras]   reconstrucoes
#   scripts/me1_run.sh metrics    [recons.pt]     metricas das reconstrucoes
#
# Dois modos:
#   NUM_SESSIONS=1 (ou 40)  benchmark: treina e testa nos mesmos dados dos outros modelos (as N
#                           primeiras sessoes, teste de 1.000 imagens, dados do MindEye2 em
#                           $MINDEYE_DATA); recon e metricas vao para results/ como os do MindEye2.
#                           Nomes: subj01_me1_<N>sess e subj01_me1_lowlevel_<N>sess.
#   sem NUM_SESSIONS        o MindEye1 original: webdataset_avg_split em $ME1_DATA, modelos
#                           publicados como padrao.
#
# DATASET=<manifesto.json> (com NUM_SESSIONS) treina nas exibicoes de um dataset controlado
# (src/mindeye_ridge/dataset_controlado.py) em vez das N primeiras sessoes; o teste nao muda.
#
# Variaveis: MODEL_NAME, AE_NAME (trocam os nomes), PAPER=1 (treino com batch 32 e AdamW normal,
# como no artigo: nao cabe numa GPU de 20 GB), SAVE_EVERY (grava o last.pth a cada N epocas).
# Treinos interrompidos retomam do last.pth. Antes: scripts/me1_setup.sh. Detalhes no docs/MINDEYE1.md.
set -e
set -o pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO/mindeye1/src"
ME1_DATA="${ME1_DATA:-$HOME/mindeye1}"
export ME1_DATA
export MPLBACKEND=Agg   # o Reconstructions.py chama plt.show(); com tela, o TkAgg trava esperando a janela
N="${NUM_SESSIONS:-0}"
if [ "$N" -gt 0 ]; then
    MODEL_NAME="${MODEL_NAME:-subj01_me1_${N}sess}"
    AE_NAME="${AE_NAME:-subj01_me1_lowlevel_${N}sess}"
    DADOS="$DATA"; BENCH="--num_sessions=$N"
else
    MODEL_NAME="${MODEL_NAME:-prior_257_final_subj01_bimixco_softclip_byol}"
    AE_NAME="${AE_NAME:-autoencoder_subj01_4x_locont_no_reconst}"
    DADOS="$ME1_DATA"; BENCH=""
fi
VD=$(ls -d "$HF_HOME"/hub/models--shi-labs--versatile-diffusion/snapshots/*/ 2>/dev/null | head -1)
PY="$ENVP/bin/python"
filtra() { grep --line-buffered -v -E "Warning|pynvml|warnings.warn|pkg_resources"; }

etapa="${1:?etapa: train | lowlevel | retrieval | recon | metrics}"; shift
mkdir -p "$REPO/mindeye1/train_logs"
case "$etapa" in
train)
    if [ "${PAPER:-0}" = 1 ]; then CFG="--batch_size=32"
    else CFG="--batch_size=16 --adam8bit --val_chunk=50"; fi   # cabe em 20 GB (pico 17,5 GiB)
    [ "$MODEL_NAME" = prior_257_final_subj01_bimixco_softclip_byol ] && MODEL_NAME=subj01_me1
    SAIDA="$REPO/mindeye1/train_logs/$MODEL_NAME"; mkdir -p "$SAIDA"
    RESUME=""; [ -f "$SAIDA/last.pth" ] && RESUME="--resume_from_ckpt"
    # --save_at_end: o original grava o best.pth escolhido pela perda no teste; no benchmark vale o
    # ultimo, como nos outros modelos
    $PY Train_MindEye.py --data_path="$DADOS" $BENCH --model_name="$MODEL_NAME" --subj=1 \
        --hidden --clip_variant=ViT-L/14 --n_samples_save=0 --no-wandb_log --save_at_end \
        --save_last_every="${SAVE_EVERY:-1}" ${DATASET:+--dataset="$DATASET"} $CFG $RESUME "$@" \
        2>&1 | filtra | tee -a "$SAIDA/train.log" ;;
lowlevel)
    # subj01, batch 8, 120 epocas, como no artigo; retoma sozinho se houver last.pth
    SAIDA="$REPO/mindeye1/train_logs/models/$AE_NAME"; mkdir -p "$SAIDA"
    if [ "$N" -gt 0 ]; then ARGS="--num_sessions=$N --data_path=$DADOS ${DATASET:+--dataset=$DATASET}"; else ARGS=""; fi
    $PY train_autoencoder.py --model_name="$AE_NAME" $ARGS "$@" 2>&1 | filtra | tee -a "$SAIDA/train.log" ;;
retrieval)
    $PY Retrievals_testset.py --data_path="$ME1_DATA" --model_name="$MODEL_NAME" --subj=1 "$@" 2>&1 | filtra ;;
recon)
    [ -n "$VD" ] || { echo "falta o Versatile Diffusion: mindeye1/download.py --stage vd"; exit 1; }
    if [ "$N" -gt 0 ]; then
        $PY Reconstructions.py --benchmark --data_path="$DADOS" --model_name="$MODEL_NAME" \
            --autoencoder_name="models/$AE_NAME" --out_name="$MODEL_NAME" --subj=1 --vd_cache_dir="$VD" "$@" 2>&1 | filtra
    else
        $PY Reconstructions.py --data_path="$DADOS" --model_name="$MODEL_NAME" \
            --autoencoder_name="$AE_NAME" --subj=1 --vd_cache_dir="$VD" "$@" 2>&1 | filtra
    fi ;;
metrics)
    if [ "$N" -gt 0 ]; then
        # as mesmas metricas dos outros modelos; a reconstrucao final do MindEye1 ja inclui o
        # img2img a partir da imagem borrada, entao e avaliada como "base" (sem a mistura 75/25)
        "$SCRIPTS/run_evals.sh" "$MODEL_NAME" base
    else
        $PY Reconstruction_Metrics.py --recon_path="${1:?caminho do .pt de reconstrucoes}" \
            --all_images_path=all_images.pt 2>&1 | filtra
    fi ;;
*) echo "etapa desconhecida: $etapa"; exit 1 ;;
esac
