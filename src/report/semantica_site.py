"""Dados do simulador de dataset do site: para cada imagem de treino do subj01, em que sessoes ela
foi exibida e quanto da tela cada supercategoria do COCO ocupa.

Formato (base64 de bytes, uma imagem apos a outra, em ordem crescente de id NSD):
    3 bytes   sessao de cada exibicao (0 a 39; 255 = sem exibicao)
    12 x 3    pixels de cada supercategoria na tela de 425 x 425, uint24 little-endian
A fracao da tabela (semantica.calcula_fracoes) e 100 * (pixels / 180.625); o simulador
(assets/simulador.js) refaz essa mesma conta em ponto flutuante, entao compara com os limiares
exatamente os numeros que o Python compara. Fica
guardado em results/site/dados/semantica.json, para o site sair igual numa maquina sem os dados.
"""
import base64
import json
import os
import struct
from pathlib import Path

from mindeye_ridge import paths

SUPERCATEGORIAS = ["accessory", "animal", "appliance", "electronic", "food", "furniture", "indoor",
                   "kitchen", "outdoor", "person", "sports", "vehicle"]
NOMES_PT = {"accessory": "acessório", "animal": "animal", "appliance": "eletrodoméstico",
            "electronic": "eletrônico", "food": "comida", "furniture": "móvel", "indoor": "objeto de casa",
            "kitchen": "cozinha", "outdoor": "rua e exterior", "person": "pessoa", "sports": "esporte",
            "vehicle": "veículo"}
SEM = 255
SEM_FRACAO = 0xFFFFFF     # imagem fora da tabela de fracoes: descartada pelo filtro semantico
TELA = 425 * 425


def _assinatura(p):
    try:
        s = Path(p).stat()
        return [s.st_size, s.st_mtime_ns]
    except OSError:
        return None


def codifica(imagem, sessao, fracoes):
    """imagem, sessao: uma entrada por exibicao; fracoes: DataFrame indexado pelo id NSD."""
    import numpy as np
    imagem, sessao = np.asarray(imagem), np.asarray(sessao)
    ids = np.unique(imagem)
    cols = [f"sup_{s}" for s in SUPERCATEGORIAS]
    tabela = fracoes.reindex(ids)[cols]
    buf = bytearray()
    for i in ids:
        s = sorted(int(x) for x in sessao[imagem == i])[:3]
        s += [SEM] * (3 - len(s))
        buf += bytes(s)
        linha = tabela.loc[i]
        pixels = [SEM_FRACAO] * 12 if linha.isna().any() else \
            [int(round(float(v) / 100 * TELA)) for v in linha]
        for p in pixels:
            buf += struct.pack("<I", p)[:3]
    return {"n": int(len(ids)), "ids": [int(ids[0]), int(ids[-1])], "sup": SUPERCATEGORIAS, "nomes": NOMES_PT,
            "tela": TELA, "dados": base64.b64encode(bytes(buf)).decode()}


def decodifica(d):
    """O inverso de codifica: (sessoes (n, 3), pixels (n, 12)), na ordem crescente de id NSD.

    Como no simulador, a fracao e 100 * pixels / tela; SEM_FRACAO marca imagem fora da tabela.
    """
    import numpy as np
    k = len(d["sup"])
    b = np.frombuffer(base64.b64decode(d["dados"]), dtype=np.uint8).reshape(d["n"], 3 + 3 * k).astype(np.int64)
    pixels = b[:, 3::3] | (b[:, 4::3] << 8) | (b[:, 5::3] << 16)
    return b[:, :3], pixels


def atualiza(pasta_site):
    """Os dados do simulador, recalculados se as fontes mudaram; None se nunca houve dados."""
    cache = Path(pasta_site) / "dados" / "semantica.json"
    tabela = paths.DATA / "semantica" / "subj01_fracao_tela.csv"
    tars = paths.DATA / "wds" / "subj01" / "train"
    sig = [_assinatura(tabela), _assinatura(tars / "0.tar"), _assinatura(tars / "39.tar")]
    try:
        antigo = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        antigo = None
    if antigo and antigo.get("sig") == sig:
        return antigo
    if any(s is None for s in sig):           # sem os dados nesta maquina: fica o que ja havia
        return antigo
    import pandas as pd
    from mindeye_ridge import nsd_data
    ex = nsd_data.exibicoes_treino(paths.DATA, 1, 40)
    fr = pd.read_csv(tabela, index_col="nsd_id", float_precision="round_trip")
    novo = dict(codifica(ex["imagem"], ex["sessao"], fr), sig=sig)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(novo, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, cache)
    return novo
