#!/usr/bin/env bash
# =============================================================================
# setup_env.sh - cria o ambiente Python para rodar o MindEye2.
#
#   ./setup_env.sh                      # cria em ~/envs/fmri com conda
#   ./setup_env.sh --path /outro/lugar  # escolhe onde criar
#   ./setup_env.sh --venv               # usa python3.11 -m venv em vez de conda
#   ./setup_env.sh --with-extras        # inclui os pacotes do final_evaluations
#
# Baseado no setup.sh do repo original (legacy/setup.sh), com quatro ajustes necessarios em
# maquinas atuais (cada um esta comentado abaixo, na secao "correcoes").
# =============================================================================
set -euo pipefail

ENV_PATH="${HOME}/envs/fmri"
USE_VENV=0
WITH_EXTRAS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --path)        ENV_PATH="$2"; shift 2 ;;
        --venv)        USE_VENV=1; shift ;;
        --with-extras) WITH_EXTRAS=1; shift ;;
        -h|--help)     sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "argumento desconhecido: $1" >&2; exit 1 ;;
    esac
done

echo "==> criando ambiente em $ENV_PATH"

if [[ $USE_VENV -eq 1 ]]; then
    command -v python3.11 >/dev/null || { echo "python3.11 nao encontrado; use conda (sem --venv)"; exit 1; }
    python3.11 -m venv "$ENV_PATH"
else
    # localiza o conda.sh onde quer que o conda esteja instalado
    CONDA_BASE="$(conda info --base 2>/dev/null)" || { echo "conda nao encontrado; use --venv"; exit 1; }
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda create -y -p "$ENV_PATH" python=3.11
fi

PIP="$ENV_PATH/bin/pip"
"$PIP" install --upgrade pip

echo "==> instalando a stack principal"
# numpy fica em <2: o torch 2.1.0 foi compilado contra a serie 1.x
"$PIP" install "numpy<2" \
    torch==2.1.0 torchvision==0.16.0 xformers==0.0.22.post7 \
    matplotlib==3.8.2 tqdm scikit-image==0.22.0 pandas==2.2.0 einops ftfy regex \
    accelerate==0.24.1 webdataset==0.2.73 kornia==0.7.1 h5py==3.10.0 \
    open_clip_torch==2.24.0 transformers==4.37.2 torchmetrics==1.3.0.post0 \
    diffusers==0.23.0 omegaconf==2.3.0 pytorch-lightning==2.0.1 wandb \
    jupyter ipykernel

echo "==> instalando CLIP (OpenAI) e dalle2-pytorch"
# mindeye_ridge/models.py importa os dois no topo do modulo, entao sao obrigatorios
"$PIP" install git+https://github.com/openai/CLIP.git --no-deps
"$PIP" install dalle2-pytorch
# dalle2-pytorch tende a subir o torch; reafirma os pins
"$PIP" install torch==2.1.0 torchvision==0.16.0 "numpy<2"

if [[ $WITH_EXTRAS -eq 1 ]]; then
    echo "==> instalando extras (metricas do final_evaluations.ipynb)"
    # datasets fica pinado em 2.x: a serie 5.x exige huggingface_hub>=0.25, que
    # quebra o diffusers 0.23.0 (veja a secao de correcoes abaixo)
    "$PIP" install sentence-transformers==2.5.1 evaluate==0.4.1 nltk==3.8.1 \
        rouge_score==0.1.2 "datasets==2.16.1"
    # o setup.sh original pede umap==0.1.1, que nao existe mais no PyPI;
    # umap-learn fornece o mesmo modulo `umap` que o notebook importa
    "$PIP" install umap-learn
fi

echo "==> aplicando correcoes de compatibilidade"
# 1. diffusers 0.23.0 importa huggingface_hub.cached_download, removido na 0.26
"$PIP" install "huggingface_hub==0.20.3"
# 2. o CLIP da OpenAI faz "from pkg_resources import packaging";
#    pkg_resources foi removido do setuptools 81+
"$PIP" install "setuptools<81"
# 3. os checkpoints publicados foram salvos com deepspeed no pickle;
#    sem ele o torch.load falha com ModuleNotFoundError
"$PIP" install deepspeed==0.13.1

echo
echo "==> verificando a instalacao"
"$ENV_PATH/bin/python" - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import os, sys
import torch, numpy, h5py, webdataset, open_clip, transformers, diffusers, accelerate, kornia
import clip, dalle2_pytorch, deepspeed

src = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0] or ".")), "src")
if not os.path.isdir(src):
    src = os.path.join(os.getcwd(), "src")
if os.path.isdir(src):
    sys.path.insert(0, src)
    sys.path.insert(0, os.path.join(src, "generative_models"))
    os.chdir(src)
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # noqa
    from mindeye_ridge.models import BrainNetwork  # noqa
    from mindeye_ridge import frr, utils  # noqa
    print("modulos do repo (sgm, mindeye_ridge): OK")

print(f"torch {torch.__version__} | CUDA disponivel: {torch.cuda.is_available()}", end="")
print(f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
print(f"numpy {numpy.__version__} | diffusers {diffusers.__version__} | transformers {transformers.__version__}")
if not torch.cuda.is_available():
    print("AVISO: o torch nao enxerga a GPU. Confira o driver NVIDIA.")
PY

echo
echo "=========================================================="
echo " Ambiente pronto: $ENV_PATH"
echo " Para usar:  $ENV_PATH/bin/python ..."
echo "        ou:  source $ENV_PATH/bin/activate"
echo " Proximo passo:  ./download_data.py --help"
echo "=========================================================="
