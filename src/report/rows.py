"""Linhas das tabelas: o que cada modelo mostra, e qual valor e o melhor do grupo."""

from .config import ROT_BWD, ROT_FWD


def melhores(linhas, chave, maior, tipo):
    """Indice da(s) linha(s) com o melhor valor de `chave` dentro de um grupo."""
    vals = [(i, l[tipo][chave]) for i, l in enumerate(linhas)
            if not l["pub"] and l[tipo] and chave in l[tipo]]
    if len(vals) < 2:
        return set()
    alvo = (max if maior else min)(v for _, v in vals)
    return {i for i, v in vals if abs(v - alvo) < 1e-12}


def linhas_de(dados, grupo):
    linhas = []
    for m in dados["modelos"]:
        if m["grupo"] != grupo:
            continue
        if m.get("frr"):
            r = m["frr_dados"]["retrieval"] if m["frr_dados"] else None
            tab = {"FwdRetrieval": r["fwd"]["media"], "BwdRetrieval": r["bwd"]["media"]} if r else None
            linhas.append({"rotulo": m["rotulo"], "sub": "só retrieval, sem reconstrução", "pub": False,
                           "enh": tab, "base": tab, "id": m["id"]})
            continue
        if m.get("me1"):
            linhas.append({"rotulo": m["rotulo"], "sub": "sem refinamento: a mesma reconstrução nas duas abas",
                           "pub": False, "enh": m["tabelas"]["enh"], "base": m["tabelas"]["base"],
                           "id": m["id"]})
            continue
        linhas.append({"rotulo": m["rotulo"], "sub": "nossa execução" if m.get("paper") else "",
                       "pub": False, "enh": m["tabelas"]["enh"], "base": m["tabelas"]["base"],
                       "id": m["id"]})
        if m.get("paper"):
            linhas.append({"rotulo": "reconstruções publicadas pelos autores", "sub": "",
                           "pub": True, "enh": m["publicada"]["enh"], "base": m["publicada"]["base"],
                           "id": m["id"]})
    return linhas


def tem_valor(l, colunas):
    """A linha tem ao menos uma das metricas das `colunas`? (o FRR nao tem as de reconstrucao)"""
    return any(l[t] and c in l[t] for t in ("enh", "base") for c, *_ in colunas)


def colunas_frr():
    """(rotulo, getter, formato) das colunas da tabela do embedding previsto."""
    return [
        (ROT_FWD, lambda d: d["retrieval"]["fwd"]["media"], "pct"),
        (ROT_BWD, lambda d: d["retrieval"]["bwd"]["media"], "pct"),
        (ROT_FWD + " centrado", lambda d: d["retrieval_centrado"]["fwd"]["media"], "pct"),
        (ROT_BWD + " centrado", lambda d: d["retrieval_centrado"]["bwd"]["media"], "pct"),
        ("Cosseno", lambda d: d["embedding"]["cosseno_medio"], "dec"),
        ("Pearson", lambda d: d["embedding"]["pearson_medio"], "dec"),
        ("Cosseno centrado", lambda d: d["embedding"]["cosseno_centrado_medio"], "dec"),
        ("R² da CV", lambda d: d["cv"]["cv_r2_escolhida"], "dec"),
    ]


def valor_frr(get, d):
    try:
        return get(d)
    except (KeyError, TypeError):
        return None
