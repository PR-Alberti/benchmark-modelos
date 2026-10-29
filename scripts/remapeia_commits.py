"""Troca hashes de commit anteriores a reescrita do historico (29/10/2026) pelos novos.

O historico foi reescrito para tirar o co-autor Claude das mensagens: o codigo de cada commit e o mesmo,
mas todos os hashes mudaram. O treina.py grava o hash em train_logs/<m>/experimento.json e o site o copia,
entao uma maquina com train_logs de antes da reescrita precisa rodar isto antes de regenerar o site:

    python scripts/remapeia_commits.py train_logs            # corrige
    python scripts/remapeia_commits.py train_logs --confere  # so lista o que trocaria

Aceita pastas e arquivos. Troca qualquer hash (7 a 40 caracteres) que seja prefixo de um unico commit antigo
pelo prefixo de mesmo tamanho do novo, entao o sufixo "+alteracoes" e as abreviacoes ficam como estavam.
"""
import argparse
import re
from pathlib import Path

MAPA = Path(__file__).resolve().parents[1] / "docs" / "commits_reescritos_2026-10-29.txt"
HASH = re.compile(rb"(?<![0-9a-fA-F])[0-9a-f]{7,40}(?![0-9a-fA-F])")
TEXTO = {".json", ".md", ".html", ".txt", ".csv", ".log", ".py", ".sh"}


def le_mapa():
    mapa = dict(l.split() for l in MAPA.read_text().splitlines() if l.strip() and not l.startswith("#"))
    prefixos = {}
    for antigo, novo in mapa.items():
        for k in range(7, 41):
            # prefixo de dois commits antigos e ambiguo: fica None e nao e trocado
            prefixos[antigo[:k]] = None if antigo[:k] in prefixos else novo[:k]
    return prefixos


def arquivos(alvos):
    # o proprio mapa e o .git ficam de fora
    fora = lambda p: p.resolve() == MAPA or ".git" in p.parts
    for a in map(Path, alvos):
        if a.is_dir():
            yield from (p for p in sorted(a.rglob("*")) if p.is_file() and p.suffix in TEXTO and not fora(p))
        elif a.is_file() and not fora(a):
            yield a


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("alvos", nargs="+", help="pastas ou arquivos")
    ap.add_argument("--confere", action="store_true", help="so lista, nao altera nada")
    args = ap.parse_args()
    prefixos = le_mapa()
    total = 0
    for p in arquivos(args.alvos):
        dado = p.read_bytes()
        trocas = []

        def troca(m):
            novo = prefixos.get(m.group().decode())
            if novo is None:
                return m.group()
            trocas.append(f"{m.group().decode()} -> {novo}")
            return novo.encode()

        novo = HASH.sub(troca, dado)
        if trocas:
            total += len(trocas)
            print(f"{p}: {', '.join(sorted(set(trocas)))}")
            if not args.confere:
                p.write_bytes(novo)
    print(f"{total} hash(es) {'a trocar' if args.confere else 'trocado(s)'}")


if __name__ == "__main__":
    main()
