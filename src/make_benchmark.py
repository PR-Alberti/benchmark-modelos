#!/usr/bin/env python3
"""Monta o benchmark de todos os modelos: metricas, curvas de treino, galeria e o baseline linear (FRR).

Le o que o run_benchmark.sh produz (results/tables, results/evals, train_logs) e grava:

    results/benchmark/index.html         pagina autocontida (imagens embutidas, abre offline)
    results/benchmark/galeria_*.jpg      grades de exemplo usadas no BENCHMARK.md
    BENCHMARK.md                         as tabelas em markdown
    results/logs/benchmark_artifact.html a pagina sem <html>/<head>, para publicar (fora do git)

O codigo da pagina esta em src/report/ (dados, tabelas, textos, pagina, markdown e, em
assets/, o CSS e o JavaScript).

Modelo ainda sem resultado aparece como pendente, entao pode rodar a qualquer
momento do pipeline.

    python src/make_benchmark.py
    python src/make_benchmark.py --n 30 --seed 0
"""
import argparse

from mindeye_ridge import paths
from report.config import MODELOS, REPO, SAIDA
from report.data import coleta
from report.markdown import grade_jpg, markdown
from report.page import ESTILO, pagina, situacao


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=30, help="imagens na galeria (padrao: 30)")
    p.add_argument("--seed", type=int, default=0, help="semente do sorteio das imagens")
    p.add_argument("--lado", type=int, default=224, help="lado das miniaturas em px")
    p.add_argument("--qualidade", type=int, default=80, help="qualidade JPEG")
    args = p.parse_args()

    dados = coleta(args)
    SAIDA.mkdir(exist_ok=True)
    titulo, fontes, corpo = pagina(dados, args)
    estilo = f"<style>{ESTILO}</style>"
    completo = ('<!doctype html>\n<html lang="pt-BR">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                f'{titulo}\n{fontes}\n{estilo}\n</head>\n<body>{corpo}</body>\n</html>\n')
    (SAIDA / "index.html").write_text(completo)
    paths.LOGS.mkdir(exist_ok=True)
    (paths.LOGS / "benchmark_artifact.html").write_text(f"{titulo}\n{fontes}\n{estilo}\n{corpo}")
    grade_jpg(dados, "enh", SAIDA / "galeria_refinadas.jpg")
    grade_jpg(dados, "base", SAIDA / "galeria_unclip.jpg")
    (REPO / "BENCHMARK.md").write_text(markdown(dados, args))

    tam = (SAIDA / "index.html").stat().st_size / 2**20
    completos, _ = situacao(dados)
    print(f"gravado: {SAIDA / 'index.html'} ({tam:.1f} MB), {REPO / 'BENCHMARK.md'}")
    print(f"modelos com metricas completas: {completos} de {len(MODELOS)}")


if __name__ == "__main__":
    main()
