#!/usr/bin/env python3
"""Gera o site do benchmark: results/site/index.html, com Inicio, Como usar e Resultados.

Descobre sozinho os modelos com resultado (report/catalogo.py), atualiza o registro de cada um, as
tiras da galeria e os dados do simulador de dataset, e monta a pagina. Os scripts de treino e de
avaliacao rodam isto ao terminar (scripts/common.sh, atualiza_site); na mao:

    python src/make_site.py
    python src/make_site.py --sem-galeria          # nao abre os tensores de reconstrucao (mais rapido)
    python src/make_site.py --fragmento pag.html   # tambem a pagina sem <html>/<head>, para publicar

Se nada mudou nos resultados, o index.html fica como estava (nem a data muda).
"""
import argparse
import os
import re
import time
from pathlib import Path

from mindeye_ridge import paths
from report import catalogo, galeria_site, semantica_site, site

CARIMBO = re.compile(r"Gerado em [^<]* a partir do commit <span class=\"mono\">[^<]*</span>")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saida", default=str(paths.SITE), help="pasta do site (padrao: results/site)")
    p.add_argument("--sem-galeria", action="store_true", help="usa as tiras ja gravadas, sem abrir os tensores")
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
    cabeca, corpo = site.monta(modelos, planejados, galeria, semantica)
    pagina = site.documento(cabeca, corpo)

    destino = saida / "index.html"
    antes = destino.read_text() if destino.exists() else None
    if antes is not None and CARIMBO.sub("", antes) == CARIMBO.sub("", pagina):
        log(f"site: {destino} sem mudanças · {len(modelos)} modelos · {time.time() - t0:.0f} s")
    else:
        tmp = destino.with_suffix(".tmp")
        tmp.write_text(pagina)
        os.replace(tmp, destino)
        log(f"site: {destino} ({destino.stat().st_size / 1e6:.1f} MB) · {len(modelos)} modelos, "
            f"{len(planejados)} planejados · {time.time() - t0:.0f} s")
    if a.fragmento:
        Path(a.fragmento).write_text(cabeca + "\n" + corpo)


if __name__ == "__main__":
    main()
