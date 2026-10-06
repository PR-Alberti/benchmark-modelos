#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh - deixa uma maquina nova pronta para treinar e avaliar os tres
# modelos do benchmark: MindEye2 (ridge-only), FRR e MindEye1.
#
#   setup/bootstrap.sh              tudo: ambiente, dados (40 sessoes), pesos, MindEye1 (~130 GB)
#   setup/bootstrap.sh --check      so diagnostica: o que ja existe, o que falta
#   setup/bootstrap.sh --ckpts      + os checkpoints ridge-only ja treinados (release do GitHub)
#   setup/bootstrap.sh --minimo     so o ridge-only de 1 sessao: ambiente + dados (~30 GB)
#
# Opcoes: --env-path DIR (padrao ~/envs/fmri), --data-path DIR (padrao ~/mindeyev2).
# Cada etapa e idempotente: pode interromper e rodar de novo que ele continua
# de onde parou. Detalhes de tudo que e instalado estao no docs/SETUP.md.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."   # raiz do repositorio

ENV_PATH="${ENV_PATH:-$HOME/envs/fmri}"
DATA_PATH="${DATA_PATH:-$HOME/mindeyev2}"
CKPT_RELEASE="${CKPT_RELEASE:-checkpoints-v1}"
# os checkpoints treinados ficam no release do repositorio de origem do MindEye2 ridge-only
REPO_URL="https://github.com/PR-Alberti/mindeye2-ridge"

# todas as etapas do download_data.py: treino de 1 e 40 sessoes, ramo blurry (ridge 4096 e
# baixo nivel do MindEye1), reconstrucao, refinamento, metricas e os modelos do artigo
STAGES=(finetune blurry recon enhanced evals paper paper40)
SESSOES=()          # vazio = todas as 40
MINIMO=0
CHECK_ONLY=0
WANT_CKPTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)  CHECK_ONLY=1; shift ;;
        --all)    shift ;;   # era a opcao para "tudo"; hoje tudo e o padrao
        --minimo) MINIMO=1; STAGES=(finetune); SESSOES=(--num-sessions 1); shift ;;
        --ckpts)  WANT_CKPTS=1; shift ;;
        --env-path)  ENV_PATH="$2"; shift 2 ;;
        --data-path) DATA_PATH="$2"; shift 2 ;;
        -h|--help)   sed -n '3,15p' "$0"; exit 0 ;;
        *) echo "argumento desconhecido: $1" >&2; exit 1 ;;
    esac
done
# os scripts de scripts/ leem estes dois (scripts/common.sh)
export MINDEYE_ENV="$ENV_PATH" MINDEYE_DATA="$DATA_PATH"

ok()   { printf '  \033[32m OK \033[0m %s\n' "$*"; }
falta(){ printf '  \033[33mFALTA\033[0m %s\n' "$*"; }
erro() { printf '  \033[31mERRO\033[0m %s\n' "$*"; }
titulo(){ printf '\n\033[1m%s\033[0m\n' "$*"; }
PROBLEMAS=0
problema() { erro "$*"; PROBLEMAS=$((PROBLEMAS+1)); }

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

# tudo: ~130 GB de dados e pesos + ~40 GB por treino do MindEye1 de 40 sessoes
precisa_gb=$([[ $MINIMO -eq 1 ]] && echo 40 || echo 200)
livre_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [[ ${livre_gb:-0} -ge $precisa_gb ]]; then ok "disco livre: ${livre_gb} GB"
else falta "disco livre: ${livre_gb} GB (recomendado ${precisa_gb} GB; veja a conta no docs/SETUP.md)"; fi

if command -v conda >/dev/null 2>&1; then ok "conda: $(conda --version)"
elif command -v python3.11 >/dev/null 2>&1; then ok "python3.11 (sera usado com --venv)"
else erro "sem conda e sem python3.11 - instale um dos dois"; fi

# --- 2. ambiente --------------------------------------------------------------
titulo "2. Ambiente Python  ($ENV_PATH)"

env_pronto() { "$ENV_PATH/bin/python" -c "import torch, open_clip, clip, deepspeed" >/dev/null 2>&1; }
env_me1()    { "$ENV_PATH/bin/python" -c "import info_nce, pytorch_msssim, bitsandbytes" >/dev/null 2>&1; }

if env_pronto; then
    ok "ja existe e importa a stack principal"
    ok "$("$ENV_PATH/bin/python" -c 'import torch; print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()}")' 2>/dev/null)"
    # ambientes criados antes de o MindEye1 entrar no setup_env.sh nao tem os 3 pacotes dele
    if env_me1; then ok "pacotes do MindEye1"
    elif [[ $CHECK_ONLY -eq 1 ]]; then falta "pacotes do MindEye1 (info-nce, msssim, bitsandbytes)"
    else
        "$ENV_PATH/bin/pip" install -q --no-deps info-nce-pytorch==0.1.0 pytorch-msssim==1.0.0 bitsandbytes==0.43.3 \
            && ok "pacotes do MindEye1 instalados" || problema "pip falhou nos pacotes do MindEye1"
    fi
elif [[ $CHECK_ONLY -eq 1 ]]; then
    falta "nao existe - rode sem --check para criar (~15 min)"
else
    echo "  criando (~15 min)..."
    args=(--with-extras --path "$ENV_PATH")
    command -v conda >/dev/null 2>&1 || args+=(--venv)
    if setup/setup_env.sh "${args[@]}"; then ok "ambiente criado"; else erro "setup_env.sh falhou"; exit 1; fi
fi

# --- 3. dados -----------------------------------------------------------------
titulo "3. Dados do MindEye2  ($DATA_PATH)"

PY=$([[ -x "$ENV_PATH/bin/python" ]] && echo "$ENV_PATH/bin/python" || echo python3)

# o download so precisa do huggingface_hub; numa maquina limpa (sem o ambiente
# ainda) instala no site do usuario para o --check funcionar de primeira
if ! "$PY" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "  instalando huggingface_hub (necessario para o download)..."
    "$PY" -m pip install --user -q huggingface_hub >/dev/null 2>&1 \
        || { erro "nao consegui instalar huggingface_hub com $PY"; exit 1; }
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    "$PY" setup/download_data.py --stage "${STAGES[@]}" --subj 1 "${SESSOES[@]}" \
        --data-path "$DATA_PATH" --dry-run 2>&1 | sed 's/^/  /'
else
    if "$PY" setup/download_data.py --stage "${STAGES[@]}" --subj 1 "${SESSOES[@]}" \
           --data-path "$DATA_PATH" 2>&1 | sed 's/^/  /'; then
        ok "dados no lugar"
    else
        erro "download falhou - pode rodar de novo, ele continua de onde parou"; exit 1
    fi
fi

# rotulos semanticos do dataset controlado (src/mindeye_ridge/semantica.py): as anotacoes COCO 2017
# (~250 MB) e a tabela de fracao da tela das 10.000 imagens do subj01 (~20 s). Sem isto, o
# treina.py baixa e calcula na primeira vez que um experimento usar classes.
TABELA="$DATA_PATH/semantica/subj01_fracao_tela.csv"
if [[ $MINIMO -eq 1 ]]; then
    echo "  rotulos semanticos: pulados com --minimo (o treina.py calcula se precisar)"
elif [[ -f "$TABELA" ]]; then
    ok "rotulos semanticos: $TABELA"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    falta "rotulos semanticos: tabela ausente (o bootstrap baixa as anotacoes COCO, ~250 MB)"
elif (cd src && "$ENV_PATH/bin/python" -W ignore -c \
        "from mindeye_ridge import semantica, paths; semantica.tabela_fracoes(paths.DATA, 1)") 2>&1 | sed 's/^/  /'; then
    ok "rotulos semanticos: $TABELA"
else
    problema "rotulos semanticos: falharam - rode de novo"
fi

# --- 4. pesos de terceiros ----------------------------------------------------
titulo "4. Pesos que os modelos baixariam na primeira execucao"

if [[ $MINIMO -eq 1 ]]; then
    echo "  (pulado com --minimo; o treino baixa o CLIP bigG sozinho na primeira vez)"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    "$PY" setup/prefetch_weights.py --check 2>&1 | grep -v -i "warn\|pynvml\|pkg_resources"
else
    "$PY" setup/prefetch_weights.py 2>&1 | grep -v -i "warn\|pynvml\|pkg_resources" \
        || problema "algum peso nao baixou - rode de novo"
fi

# --- 5. MindEye1 --------------------------------------------------------------
titulo "5. MindEye1 (links para os pesos do MindEye2)"

if [[ $MINIMO -eq 1 ]]; then
    echo "  (pulado com --minimo)"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    scripts/me1_setup.sh --check 2>&1 | sed 's/^/  /'
else
    scripts/me1_setup.sh 2>&1 | sed 's/^/  /' || problema "scripts/me1_setup.sh falhou"
fi

# --- 6. checkpoints ja treinados (opcional) -----------------------------------
titulo "6. Modelos ridge-only ja treinados (opcional)"

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
    echo "  (pulado; use --ckpts para nao precisar retreinar o ridge-only de 1 sessao)"
fi

# --- 7. verificacao -----------------------------------------------------------
titulo "7. Cada modelo carrega?"

# importa o codigo de cada modelo com o ambiente e confere os arquivos de que ele precisa,
# sem treinar nada; assim um problema aparece agora, e nao horas depois de um treino comecar
verifica() {   # rotulo, arquivos necessarios..., -- , comando python
    local rotulo="$1"; shift
    local faltam=()
    while [[ "$1" != "--" ]]; do [[ -e "$1" ]] || faltam+=("$(basename "$1")"); shift; done
    shift
    if [[ ${#faltam[@]} -gt 0 ]]; then
        if [[ $CHECK_ONLY -eq 1 ]]; then falta "$rotulo: faltam ${faltam[*]}"
        else problema "$rotulo: faltam ${faltam[*]}"; fi
        return
    fi
    if (cd "$1" && "$ENV_PATH/bin/python" -W ignore -c "$2") >/dev/null 2>&1; then ok "$rotulo"
    elif [[ $CHECK_ONLY -eq 1 ]]; then falta "$rotulo: o codigo nao importa (ambiente incompleto?)"
    else problema "$rotulo: o codigo nao importa - rode o import na mao para ver o erro"; fi
}

D="$DATA_PATH"
verifica "MindEye2 ridge-only (1 sessao)" "$D/betas_all_subj01_fp32_renorm.hdf5" \
    "$D/coco_images_224_float16.hdf5" "$D/wds/subj01/train/0.tar" \
    "$D/train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth" -- \
    src "import sys; sys.path.insert(0,'generative_models'); import mindeye_ridge.models, mindeye_ridge.utils; from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder"
if [[ $MINIMO -eq 0 ]]; then
    verifica "MindEye2 ridge-only (40 sessoes, 4096 + blurry)" "$D/wds/subj01/train/39.tar" \
        "$D/train_logs/final_multisubject_subj01/last.pth" "$D/convnext_xlarge_alpha0.75_fullckpt.pth" -- \
        src "import mindeye_ridge.models"
    verifica "MindEye2 reconstrucao e metricas" "$D/unclip6_epoch0_step110000.ckpt" \
        "$D/zavychromaxl_v30.safetensors" "$D/evals/all_images.pt" "$D/gnet_multisubject.pt" -- \
        src "import sentence_transformers, evaluate, nltk"
    verifica "FRR (1 e 40 sessoes)" "$D/wds/subj01/train/39.tar" "$D/wds/subj01/new_test/0.tar" -- \
        src "import mindeye_ridge.frr, mindeye_ridge.nsd_data, mindeye_ridge.clip_targets"
    verifica "Dataset controlado (src/treina.py)" "$D/semantica/subj01_fracao_tela.csv" \
        "$D/wds/subj01/train/39.tar" -- \
        src "import treina, experimentos, mindeye_ridge.dataset_controlado, mindeye_ridge.semantica"
    verifica "MindEye1 (1 e 40 sessoes)" "$D/sd_image_var_autoenc.pth" \
        "mindeye1/train_logs/models/convnext_xlarge_alpha0.75_fullckpt.pth" -- \
        mindeye1/src "import utils, models, nsd_benchmark, vd_compat, bitsandbytes"
fi

# --- 8. resumo ----------------------------------------------------------------
titulo "Resumo"

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "  Diagnostico apenas. Para preparar tudo:  setup/bootstrap.sh"
    exit 0
fi
if [[ $PROBLEMAS -gt 0 ]]; then
    erro "$PROBLEMAS problema(s) acima. Rode de novo: o que ja esta pronto e pulado."
    exit 1
fi

cat <<FIM
  ambiente : $ENV_PATH
  dados    : $DATA_PATH

  Pronto para treinar. Um exemplo de cada modelo (detalhes no README):

  MindEye2 ridge-only
      scripts/run_ridgeonly_prior.sh                                  # 1 sessao (~2 h)
      MODEL_NAME=subj01_ridgeonly_40sess_prior NUM_SESSIONS=40 NUM_EPOCHS=20 FROZEN=1 \\
          scripts/run_ridgeonly_prior.sh                              # 40 sessoes (~8 h)
  FRR
      scripts/run_frr.sh                                              # 1 sessao (~1 min)
      NUM_SESSIONS=40 scripts/run_frr.sh                              # 40 sessoes (~30 min)
  MindEye1
      scripts/run_me1_benchmark.sh                                    # 1 e 40 sessoes (~37 h)
      SESSOES=1 scripts/run_me1_benchmark.sh                          # so 1 sessao (~5 h)

  Treinos longos: rode com nohup ... & para sobreviver ao fechamento do terminal.
FIM
