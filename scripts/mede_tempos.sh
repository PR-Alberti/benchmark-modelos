#!/bin/bash
# Quanto custa cada etapa depois do treino, com testes curtos (cerca de 1 h de GPU, nenhum modelo refeito):
#
#   scripts/mede_tempos.sh --fonte results/evals/<modelo MindEye2 1024 com as 1.000 reconstrucoes>
#
# Grava results/tables/tempos_etapas.json e regenera o site (aba Tempos). Interrompido, retoma do que
# falta. Detalhes em src/etapas/mede_tempos.py.
set -e
set -o pipefail
ORIGEM_FONTE="$PWD"
source "$(dirname "$0")/common.sh"
args=(); while [ $# -gt 0 ]; do
    case "$1" in
        --fonte) case "$2" in /*) args+=(--fonte "$2") ;; *) args+=(--fonte "$ORIGEM_FONTE/$2") ;; esac; shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
"$ENVP/bin/python" -u etapas/mede_tempos.py "${args[@]}"
atualiza_site
