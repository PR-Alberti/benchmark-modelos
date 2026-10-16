#!/usr/bin/env python3
"""Gera o site do benchmark: results/site/index.html, com Inicio, Como usar e Resultados, e a
apresentacao do trabalho, results/site/apresentacao.html.

Descobre sozinho os modelos com resultado (report/catalogo.py), atualiza o registro de cada um, as
tiras da galeria, os dados do simulador de dataset e os exemplos do controle semantico, e monta as
duas paginas. Os scripts de treino e de avaliacao rodam isto ao terminar (scripts/common.sh,
atualiza_site); na mao:

    python src/make_site.py
    python src/make_site.py --sem-galeria          # nao abre os tensores de reconstrucao (mais rapido)
    python src/make_site.py --exemplos             # refaz os exemplos semanticos, baixando o que faltar
    python src/make_site.py --fragmento pag.html   # tambem a pagina sem <html>/<head>, para publicar

Se nada mudou nos resultados, as paginas ficam como estavam (nem a data muda).
"""
import argparse
import os
import re
import time
from pathlib import Path

from mindeye_ridge import paths
from report import apresentacao, catalogo, exemplos_site, galeria_site, semantica_site, site

CARIMBO = re.compile(r"Gerado em [^<]* a partir do commit <span class=\"mono\">[^<]*</span>")


def grava(destino, pagina, log, resumo=""):
    """Grava a pagina se ela mudou (fora o carimbo de data e commit)."""
    antes = destino.read_text(encoding="utf-8") if destino.exists() else None
    if antes is not None and CARIMBO.sub("", antes) == CARIMBO.sub("", pagina):
        log(f"site: {destino} sem mudanças{resumo}")
        return
    tmp = destino.with_suffix(".tmp")
    tmp.write_text(pagina, encoding="utf-8")
    os.replace(tmp, destino)
    log(f"site: {destino} ({destino.stat().st_size / 1e6:.1f} MB){resumo}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saida", default=str(paths.SITE), help="pasta do site (padrao: results/site)")
    p.add_argument("--sem-galeria", action="store_true", help="usa as tiras ja gravadas, sem abrir os tensores")
    p.add_argument("--exemplos", action="store_true",
                   help="refaz os exemplos semanticos da apresentacao, baixando as anotacoes COCO se faltarem")
    p.add_argument("--fragmento", default=None, help="grava tambem a pagina sem <html>/<head> neste arquivo")
    p.add_argument("--quieto", action="store_true", help="so os avisos")
    a = p.parse_args(argv)
    log = (lambda *x: None) if a.quieto else print
    t0 = time.time()

    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    modelos, planejados = catalogo.atualiza(saida / "modelos", resumos=catalogo.resumo_completo_factory())
    galeria = galeria_site.atualiza(modelos, saida, gerar=not a.sem_galeria, log=log)
    semantica = semantica_site.atualiza(saida)
    exemplos = exemplos_site.atualiza(saida, semantica, gerar=not a.sem_galeria, baixar=a.exemplos, log=log)
    cabeca, corpo = site.monta(modelos, planejados, galeria, semantica)
    grava(saida / "index.html", site.documento(cabeca, corpo), log,
          f" · {len(modelos)} modelos, {len(planejados)} planejados")
    grava(saida / "apresentacao.html",
          site.documento(*apresentacao.monta(modelos, planejados, galeria, semantica, exemplos)), log)
    log(f"site: {time.time() - t0:.0f} s")
    if a.fragmento:
        Path(a.fragmento).write_text(cabeca + "\n" + corpo, encoding="utf-8")


if __name__ == "__main__":
    main()
