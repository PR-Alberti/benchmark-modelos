"""Caminhos do repositorio e dos dados, em um lugar so.

Tudo parte da localizacao deste arquivo, entao os scripts rodam de qualquer diretorio. Os
dados (o dataset do HuggingFace e o cache dos pesos) ficam em MINDEYE_DATA, com ~/mindeyev2
como padrao; o scripts/common.sh exporta a variavel para os pontos de entrada.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
DATA = Path(os.environ.get("MINDEYE_DATA", "~/mindeyev2")).expanduser()

TRAIN_LOGS = REPO / "train_logs"       # checkpoints e logs de treino (fora do git)
RESULTS = REPO / "results"
EVALS = RESULTS / "evals"              # tensores que cada modelo produz (fora do git)
TABLES = RESULTS / "tables"            # metricas finais (no git)
FIGS = RESULTS / "figs"
METRICS = RESULTS / "metrics"          # curvas de treino guardadas de corridas antigas
BENCHMARK = REPO / "benchmark"         # pagina gerada pelo make_benchmark.py


def add_vendored_to_path():
    """Poe src/ e src/generative_models/ no sys.path.

    O repositorio da Stability AI (src/generative_models) se importa como `sgm`, de dentro
    dele, e o resto do codigo como `generative_models.sgm`; os dois precisam funcionar.
    """
    for p in (SRC, SRC / "generative_models"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
