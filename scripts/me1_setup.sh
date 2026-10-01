#!/bin/bash
# Prepara o MindEye1 em cima do ambiente e dos dados do MindEye2 (rode o setup_env.sh antes).
#
#   scripts/me1_setup.sh            instala os 3 pacotes que faltam e cria os links
#   scripts/me1_setup.sh --check    so mostra o que falta
#
# O MindEye1 foi escrito para torch 2.0.1 / diffusers 0.13 / accelerate 0.19; roda no
# ambiente do MindEye2 (torch 2.1, diffusers 0.23, accelerate 0.24) com os ajustes descritos
# no MINDEYE1.md. Faltam so:
#   info-nce-pytorch, pytorch-msssim   importados pelo utils.py do MindEye1
#   bitsandbytes 0.43.3                Adam de 8 bits, para o treino caber em 20 GB
#
# Os scripts do MindEye1 procuram tudo em mindeye1/train_logs/ (caminhos relativos a
# mindeye1/src). Este script liga ali:
#   models/sd_image_var_autoenc.pth, models/convnext_xlarge_alpha0.75_fullckpt.pth
#       -> os mesmos arquivos do MindEye2, em $MINDEYE_DATA (treino do low-level)
#   prior_257_final_subj01_bimixco_softclip_byol, autoencoder_subj01_4x_locont_no_reconst
#       -> os modelos publicados, em $ME1_DATA/mindeye_models (mindeye1/download.py --stage ckpts)
set -e
source "$(dirname "$0")/common.sh"
ME1="$REPO/mindeye1"
ME1_DATA="${ME1_DATA:-$HOME/mindeye1}"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1

falta=0
echo "== pacotes extras em $ENVP"
for mod in info_nce pytorch_msssim bitsandbytes; do
    if "$ENVP/bin/python" -c "import $mod" 2>/dev/null; then echo "  ok     $mod"
    else echo "  FALTA  $mod"; falta=1; fi
done
if [ $falta = 1 ] && [ $CHECK = 0 ]; then
    # --no-deps: nenhum dos tres precisa de nada alem do torch, que ja esta no ambiente,
    # e sem isso o pip tenta trocar a versao do torch
    "$ENVP/bin/pip" install --no-deps info-nce-pytorch==0.1.0 pytorch-msssim==1.0.0 bitsandbytes==0.43.3
fi

liga() {  # liga <destino> <origem>
    if [ -e "$1" ]; then echo "  ok     ${1#$REPO/}"
    elif [ ! -e "$2" ]; then echo "  FALTA  $2"
    elif [ $CHECK = 1 ]; then echo "  falta  ${1#$REPO/} (o link sera criado)"
    else mkdir -p "$(dirname "$1")"; ln -sfn "$2" "$1"; echo "  ligado ${1#$REPO/} -> $2"; fi
}
echo "== links em mindeye1/train_logs"
liga "$ME1/train_logs/models/sd_image_var_autoenc.pth" "$DATA/sd_image_var_autoenc.pth"
liga "$ME1/train_logs/models/convnext_xlarge_alpha0.75_fullckpt.pth" "$DATA/convnext_xlarge_alpha0.75_fullckpt.pth"
for m in prior_257_final_subj01_bimixco_softclip_byol autoencoder_subj01_4x_locont_no_reconst; do
    liga "$ME1/train_logs/$m" "$ME1_DATA/mindeye_models/$m"
done

echo "== dados em $ME1_DATA"
for f in webdataset_avg_split/test/test_subj01_0.tar webdataset_avg_split/train/train_subj01_0.tar; do
    [ -e "$ME1_DATA/$f" ] && echo "  ok     $f" || echo "  FALTA  $f  (mindeye1/download.py)"
done
VD=$(ls -d "$HF_HOME"/hub/models--shi-labs--versatile-diffusion/snapshots/*/ 2>/dev/null | head -1)
[ -n "$VD" ] && echo "  ok     Versatile Diffusion ($VD)" || echo "  FALTA  Versatile Diffusion  (mindeye1/download.py --stage vd)"
