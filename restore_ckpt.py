#!/usr/bin/env python3
"""Reconstroi um checkpoint .pth que foi descompactado em pasta.

Um .pth do PyTorch e um zip. Alguns backups (o Google Drive, por exemplo)
descompactam esse zip e voce recebe de volta uma PASTA com data.pkl, data/,
version, byteorder e .data/ -- que o torch nao consegue carregar.

Re-zipar a pasta na mao NAO resolve: o torch exige que os dados de cada registro
comecem alinhados em 64 bytes, coisa que o zipfile do Python nao faz. O jeito
certo e ler as storages e deixar o proprio torch gravar de novo, que e o que
este script faz.

    ./restore_ckpt.py pasta_do_checkpoint/ -o train_logs/meu_modelo/last.pth

Se a pasta ja for um .pth valido, ele avisa e nao faz nada.
"""
import argparse
import pathlib
import pickle
import sys


def valido(caminho):
    """Um .pth so e legivel pelo torch se os registros estiverem alinhados a 64."""
    import struct
    import zipfile
    if not caminho.is_file():
        return False
    try:
        z = zipfile.ZipFile(caminho)
    except zipfile.BadZipFile:
        return False
    i = z.infolist()[0]
    with open(caminho, "rb") as f:
        f.seek(i.header_offset + 26)
        nlen, elen = struct.unpack("<HH", f.read(4))
    return (i.header_offset + 30 + nlen + elen) % 64 == 0


def reconstroi(pasta, destino):
    import torch

    raiz = pasta if (pasta / "data.pkl").exists() else next(
        (d for d in pasta.iterdir() if (d / "data.pkl").exists()), None)
    if raiz is None:
        sys.exit(f"nao achei data.pkl em {pasta} (nem um nivel abaixo)")

    cache = {}   # storages sao compartilhadas entre tensores: reusar pela chave

    class U(pickle.Unpickler):
        def persistent_load(self, sid):
            assert sid[0] == "storage", sid[0]
            tipo, chave, _local, _numel = sid[1:]
            if chave not in cache:
                dtype = torch.uint8 if tipo is torch.UntypedStorage else tipo.dtype
                buf = bytearray((raiz / "data" / str(chave)).read_bytes())
                st = torch.frombuffer(buf, dtype=torch.uint8).untyped_storage()
                cache[chave] = torch.storage.TypedStorage(
                    wrap_storage=st, dtype=dtype, _internal=True)
            return cache[chave]

    ck = U(open(raiz / "data.pkl", "rb"), encoding="utf-8").load()
    destino.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, destino)

    chaves = ", ".join(sorted(ck)) if isinstance(ck, dict) else type(ck).__name__
    print(f"gravado: {destino}  ({destino.stat().st_size / 1e9:.2f} GB)")
    print(f"  storages: {len(cache)} | chaves: {chaves}")
    if isinstance(ck, dict) and "epoch" in ck:
        print(f"  epoch: {ck['epoch']}")

    # so acredita depois de carregar de verdade
    torch.load(destino, map_location="cpu")
    print("  torch.load: OK")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("origem", type=pathlib.Path,
                   help="pasta descompactada (ou o .pth, para so verificar)")
    p.add_argument("-o", "--saida", type=pathlib.Path,
                   help="onde gravar o .pth (padrao: <origem>.pth)")
    args = p.parse_args()

    if args.origem.is_file():
        print("valido para o torch" if valido(args.origem)
              else "NAO e legivel pelo torch: registros desalinhados")
        return

    destino = args.saida or args.origem.with_suffix(".pth")
    reconstroi(args.origem, destino)


if __name__ == "__main__":
    main()
