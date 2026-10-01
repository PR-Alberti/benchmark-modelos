#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh - deixa uma maquina nova pronta para rodar o MindEye2.
#
#   ./bootstrap.sh --check        so diagnostica: o que ja existe, o que falta
#   ./bootstrap.sh                ambiente + dados de treino  (~30 GB)
#   ./bootstrap.sh --all          ambiente + todos os dados   (~62 GB)
#   ./bootstrap.sh --all --ckpts  + os checkpoints ja treinados (nao precisa treinar)
#
# Cada etapa e idempotente: pode interromper e rodar de novo que ele continua
# de onde parou. Detalhes de tudo que e instalado estao no SETUP.md.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

ENV_PATH="${ENV_PATH:-$HOME/envs/fmri}"
DATA_PATH="${DATA_PATH:-$HOME/mindeyev2}"
CKPT_RELEASE="${CKPT_RELEASE:-checkpoints-v1}"
# os checkpoints treinados ficam no release do repositorio de origem do MindEye2 ridge-only
REPO_URL="https://github.com/PR-Alberti/mindeye2-ridge"

STAGES=(finetune)
CHECK_ONLY=0
WANT_CKPTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)  CHECK_ONLY=1; shift ;;
        --all)    STAGES=(finetune recon enhanced evals paper); shift ;;
        --ckpts)  WANT_CKPTS=1; shift ;;
        --env-path)  ENV_PATH="$2"; shift 2 ;;
        --data-path) DATA_PATH="$2"; shift 2 ;;
        -h|--help)   sed -n '3,13p' "$0"; exit 0 ;;
        *) echo "argumento desconhecido: $1" >&2; exit 1 ;;
    esac
done

ok()   { printf '  \033[32m OK \033[0m %s\n' "$*"; }
falta(){ printf '  \033[33mFALTA\033[0m %s\n' "$*"; }
erro() { printf '  \033[31mERRO\033[0m %s\n' "$*"; }
titulo(){ printf '\n\033[1m%s\033[0m\n' "$*"; }

# --- 1. diagnostico -----------------------------------------------------------
titulo "1. Maquina"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    erro "nenhuma GPU NVIDIA utilizavel"
    if command -v lspci >/dev/null 2>&1; then
        if lspci 2>/dev/null | grep -qi "nvidia"; then
            echo "        a placa aparece no PCI, mas o driver nao responde -> problema de driver"
        else
            echo "        nenhuma placa NVIDIA no barramento PCI -> a placa nao esta"
            echo "        encaixada, esta sem cabo de forca, ou foi removida."
            echo "        Confira tambem se o monitor esta na saida da placa-mae."
        fi
    fi
    echo "        (o resto do bootstrap continua; so o treino e que nao roda sem GPU)"
fi

livre_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [[ ${livre_gb:-0} -ge 85 ]]; then ok "disco livre: ${livre_gb} GB"
else falta "disco livre: ${livre_gb} GB (recomendado 85 GB; veja a conta no SETUP.md)"; fi

if command -v conda >/dev/null 2>&1; then ok "conda: $(conda --version)"
elif command -v python3.11 >/dev/null 2>&1; then ok "python3.11 (sera usado com --venv)"
else erro "sem conda e sem python3.11 - instale um dos dois"; fi

# --- 2. ambiente --------------------------------------------------------------
titulo "2. Ambiente Python  ($ENV_PATH)"

env_pronto() { "$ENV_PATH/bin/python" -c "import torch, open_clip, clip, deepspeed" >/dev/null 2>&1; }

if env_pronto; then
    ok "ja existe e importa a stack principal"
    ok "$("$ENV_PATH/bin/python" -c 'import torch; print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()}")' 2>/dev/null)"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    falta "nao existe - rode sem --check para criar (~15 min)"
else
    echo "  criando (~15 min)..."
    args=(--with-extras --path "$ENV_PATH")
    command -v conda >/dev/null 2>&1 || args+=(--venv)
    if ./setup_env.sh "${args[@]}"; then ok "ambiente criado"; else erro "setup_env.sh falhou"; exit 1; fi
fi

# --- 3. dados -----------------------------------------------------------------
titulo "3. Dados externos  ($DATA_PATH)"

PY=$([[ -x "$ENV_PATH/bin/python" ]] && echo "$ENV_PATH/bin/python" || echo python3)

# o download so precisa do huggingface_hub; numa maquina limpa (sem o ambiente
# ainda) instala no site do usuario para o --check funcionar de primeira
if ! "$PY" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "  instalando huggingface_hub (necessario para o download)..."
    "$PY" -m pip install --user -q huggingface_hub >/dev/null 2>&1 \
        || { erro "nao consegui instalar huggingface_hub com $PY"; exit 1; }
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    "$PY" download_data.py --stage "${STAGES[@]}" --subj 1 --num-sessions 1 \
        --data-path "$DATA_PATH" --dry-run 2>&1 | sed 's/^/  /'
else
    if "$PY" download_data.py --stage "${STAGES[@]}" --subj 1 --num-sessions 1 \
           --data-path "$DATA_PATH" 2>&1 | sed 's/^/  /'; then
        ok "dados no lugar"
    else
        erro "download falhou - pode rodar de novo, ele continua de onde parou"; exit 1
    fi
fi

# --- 4. checkpoints ja treinados ---------------------------------------------
titulo "4. Modelos treinados e reconstrucoes"

baixa_ckpt() {   # nome_do_modelo
    local m="$1" destino="train_logs/$1/last.pth"
    [[ -f "$destino" ]] && { ok "$m: ja existe"; return 0; }
    if [[ $CHECK_ONLY -eq 1 ]]; then falta "$m: ausente (--ckpts baixa)"; return 0; fi

    mkdir -p "train_logs/$m"
    local base="$REPO_URL/releases/download/$CKPT_RELEASE"
    echo "  baixando $m ..."
    # arquivos acima de 2 GB vao partidos no release; junta de volta se houver partes
    if curl -fsSL -o "$destino.tmp" "$base/$m.last.pth" 2>/dev/null; then
        mv "$destino.tmp" "$destino"
    else
        rm -f "$destino.tmp" "$destino".part-*
        local i=0 sufixo
        for sufixo in 00 01 02 03 04; do
            curl -fsSL -o "$destino.part-$sufixo" "$base/$m.last.pth.part-$sufixo" 2>/dev/null || break
            i=$((i+1))
        done
        if [[ $i -eq 0 ]]; then
            falta "$m: nao publicado no release $CKPT_RELEASE"
            echo "        de para regerar treinando: scripts/run_ridgeonly_prior.sh"
            rm -rf "train_logs/$m"; return 0
        fi
        cat "$destino".part-* > "$destino" && rm -f "$destino".part-*
    fi
    ok "$m: $(du -h "$destino" | cut -f1)"
}

baixa_recons() {   # nome_do_modelo -- tensores de reconstrucao ja gerados
    local m="$1" destino="results/evals/$1"
    [[ -f "$destino/${m}_all_enhancedrecons.pt" ]] && { ok "$m: reconstrucoes ja existem"; return 0; }
    if [[ $CHECK_ONLY -eq 1 ]]; then falta "$m: reconstrucoes ausentes (--ckpts baixa)"; return 0; fi

    local tar="$REPO_URL/releases/download/$CKPT_RELEASE/${m}_recons.tar.gz"
    echo "  baixando reconstrucoes de $m ..."
    mkdir -p "$destino"
    if curl -fsSL "$tar" | tar xzf - -C "$destino"; then
        ok "$m: reconstrucoes em $destino"
    else
        falta "$m: reconstrucoes nao publicadas"
        echo "        de para regerar: scripts/run_recon.sh $m 1024 noblurry  (~3h)"
        rmdir "$destino" 2>/dev/null || true
    fi
}

if [[ $WANT_CKPTS -eq 1 || $CHECK_ONLY -eq 1 ]]; then
    baixa_ckpt subj01_ridgeonly_1sess_prior
    baixa_ckpt subj01_ridgeonly_1sess
    baixa_recons subj01_ridgeonly_1sess_prior
else
    echo "  (pulado; use --ckpts para baixar modelos treinados e reconstrucoes)"
fi

# --- 5. resumo ----------------------------------------------------------------
titulo "Pronto"

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "  Diagnostico apenas. Para preparar de verdade:  ./bootstrap.sh --all --ckpts"
    exit 0
fi

cat <<FIM
  ambiente : $ENV_PATH
  dados    : $DATA_PATH

  Treinar (ridge-only + prior, subj01, 1 sessao):
      scripts/run_ridgeonly_prior.sh

  Rodar um modelo ja treinado (reconstrucoes + metricas):
      scripts/run_recon.sh subj01_ridgeonly_1sess_prior 1024 noblurry
      scripts/run_evals.sh subj01_ridgeonly_1sess_prior enhanced

  Tudo explicado em SETUP.md.
FIM
