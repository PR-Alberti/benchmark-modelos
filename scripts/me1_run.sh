#!/bin/bash
# Roda uma etapa do MindEye1 (subj01, alto nivel em CLIP ViT-L/14 257x768, Versatile Diffusion).
#
#   scripts/me1_run.sh train      [args extras]   treino do alto nivel  -> mindeye1/train_logs/$MODEL_NAME
#   scripts/me1_run.sh lowlevel                   treino do low-level   -> mindeye1/train_logs/autoencoder
#   scripts/me1_run.sh retrieval                  retrieval no teste (30 sorteios de 300)
#   scripts/me1_run.sh recon      [args extras]   reconstrucoes         -> mindeye1/src/<modelo>_recons_*.pt
#   scripts/me1_run.sh metrics    <recons.pt>     metricas das reconstrucoes
#
# Variaveis: MODEL_NAME (modelo de alto nivel; padrao: o publicado), AE_NAME (low-level;
# padrao: o publicado), ME1_DATA (dados, padrao ~/mindeye1), PAPER=1 (treino com batch 32
# e AdamW normal, como no artigo: nao cabe numa GPU de 20 GB).
# Antes: scripts/me1_setup.sh e mindeye1/download.py. Detalhes no MINDEYE1.md.
set -e
set -o pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO/mindeye1/src"
ME1_DATA="${ME1_DATA:-$HOME/mindeye1}"
export ME1_DATA
MODEL_NAME="${MODEL_NAME:-prior_257_final_subj01_bimixco_softclip_byol}"
AE_NAME="${AE_NAME:-autoencoder_subj01_4x_locont_no_reconst}"
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
    mkdir -p "$REPO/mindeye1/train_logs/$MODEL_NAME"
    $PY Train_MindEye.py --data_path="$ME1_DATA" --model_name="$MODEL_NAME" --subj=1 \
        --hidden --clip_variant=ViT-L/14 --n_samples_save=0 --no-wandb_log $CFG "$@" \
        2>&1 | filtra | tee -a "$REPO/mindeye1/train_logs/$MODEL_NAME/train.log" ;;
lowlevel)
    # o script nao tem argumentos: subj01, batch 8, 120 epocas, como no artigo
    $PY train_autoencoder.py 2>&1 | filtra ;;
retrieval)
    $PY Retrievals_testset.py --data_path="$ME1_DATA" --model_name="$MODEL_NAME" --subj=1 "$@" 2>&1 | filtra ;;
recon)
    [ -n "$VD" ] || { echo "falta o Versatile Diffusion: mindeye1/download.py --stage vd"; exit 1; }
    $PY Reconstructions.py --data_path="$ME1_DATA" --model_name="$MODEL_NAME" \
        --autoencoder_name="$AE_NAME" --subj=1 --vd_cache_dir="$VD" "$@" 2>&1 | filtra ;;
metrics)
    $PY Reconstruction_Metrics.py --recon_path="${1:?caminho do .pt de reconstrucoes}" \
        --all_images_path=all_images.pt 2>&1 | filtra ;;
*) echo "etapa desconhecida: $etapa"; exit 1 ;;
esac
