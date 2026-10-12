"""Galeria do site: as mesmas imagens de teste para todos os modelos, em tiras WebP.

Cada tira (results/site/galeria/<id>__<tipo>.webp) junta, lado a lado, as N miniaturas de um tipo de
reconstrucao de um modelo; a pagina mostra cada miniatura recortando a tira. Uma tira so e refeita
quando o tensor de origem muda (tamanho e data), e numa maquina sem os tensores (results/evals fica
fora do git) as tiras e o indice.json ja gravados continuam valendo.
"""
import json
import os
from pathlib import Path

import numpy as np

from mindeye_ridge import paths

from .catalogo import PUBLICADA

N, LADO, SEMENTE, QUALIDADE = 24, 176, 0, 74
TIPOS = {"refinada": "all_enhancedrecons", "unclip": "all_recons", "borrada": "all_blurryrecons"}


def _assinatura(p):
    try:
        s = Path(p).stat()
        return [s.st_size, s.st_mtime_ns]
    except OSError:
        return None


def _carrega(p):
    import torch
    try:
        return torch.load(str(p), map_location="cpu", mmap=True)
    except RuntimeError:                  # formato antigo nao aceita mmap
        return torch.load(str(p), map_location="cpu")


def _tira(tensor, idx, destino, lado=LADO):
    from PIL import Image
    tira = Image.new("RGB", (lado * len(idx), lado))
    for j, i in enumerate(idx):
        a = (tensor[i].float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        im = Image.fromarray(a)
        if im.size != (lado, lado):
            im = im.resize((lado, lado), Image.LANCZOS)
        tira.paste(im, (j * lado, 0))
    tmp = destino.with_name(destino.stem + ".tmp.webp")
    tira.save(tmp, "WEBP", quality=QUALIDADE, method=6)
    os.replace(tmp, destino)


def _fontes(m):
    """{tipo: tensor de origem} e o arquivo de legendas previstas de um modelo."""
    mid = m["id"]
    if m.get("publicada"):
        base = mid[:-len(PUBLICADA)]
        return {t: paths.EVALS / base / f"{base}_published_{s}.pt" for t, s in TIPOS.items()
                if t != "borrada"}, None
    return ({t: paths.EVALS / mid / f"{mid}_{s}.pt" for t, s in TIPOS.items()},
            paths.EVALS / mid / f"{mid}_all_predcaptions.pt")


def atualiza(modelos, pasta_site, gerar=True, log=print):
    """Gera as tiras que faltam ou mudaram e devolve os dados da galeria para a pagina.

    gerar=False so le o que ja esta gravado (mais rapido; nao toca nos tensores).
    """
    pasta = Path(pasta_site) / "galeria"
    pasta.mkdir(parents=True, exist_ok=True)
    arq_indice = pasta / "indice.json"
    try:
        indice = json.loads(arq_indice.read_text())
    except (OSError, ValueError):
        indice = {}
    params = {"n": N, "lado": LADO, "semente": SEMENTE}
    if any(indice.get(k) != v for k, v in params.items()):
        indice = dict(params, modelos={}, legendas={})     # outra galeria: tudo de novo
    idx = indice.get("idx")
    if not idx:
        idx = sorted(np.random.default_rng(SEMENTE).choice(1000, N, replace=False).tolist())
    indice["idx"] = idx
    mudou = False

    if gerar:
        img = paths.DATA / "evals/all_images.pt"
        sig = _assinatura(img)
        vista = pasta / "vista.webp"
        if sig and (indice.get("vista", {}).get("sig") != sig or not vista.exists()):
            log("galeria: imagens vistas")
            _tira(_carrega(img), idx, vista)
            indice["vista"] = {"arquivo": vista.name, "sig": sig}
            mudou = True
        cap = paths.DATA / "evals/all_captions.pt"
        if not indice.get("coco") and cap.exists():
            legs = np.asarray(_carrega(cap)).astype(str)
            indice["coco"] = [str(legs[i]) for i in idx]
            mudou = True
        if not indice.get("nsd"):
            from .data import ids_nsd
            nsd = ids_nsd()
            if nsd:
                indice["nsd"] = [nsd[i] for i in idx]
                mudou = True

        for m in modelos:
            if m["familia"] == "frr":
                continue
            fontes, legendas = _fontes(m)
            atual = indice["modelos"].setdefault(m["id"], {})
            for tipo, src in fontes.items():
                sig = _assinatura(src)
                destino = pasta / f"{m['id']}__{tipo}.webp"
                if sig is None or (atual.get(tipo, {}).get("sig") == sig and destino.exists()):
                    continue
                log(f"galeria: {m['id']} ({tipo})")
                try:
                    _tira(_carrega(src), idx, destino)
                except Exception as e:                      # um tensor estragado nao derruba o site
                    log(f"aviso: galeria de {m['id']} ({tipo}): {e}")
                    continue
                atual[tipo] = {"arquivo": destino.name, "sig": sig}
                mudou = True
            if legendas is not None and legendas.exists() and not indice["legendas"].get(m["id"]):
                leg = np.asarray(_carrega(legendas)).astype(str)
                indice["legendas"][m["id"]] = [str(leg[i]) for i in idx]
                mudou = True
            if not atual:
                del indice["modelos"][m["id"]]

    if mudou or not arq_indice.exists():
        tmp = arq_indice.with_suffix(".tmp")
        tmp.write_text(json.dumps(indice, ensure_ascii=False, indent=1, sort_keys=True))
        os.replace(tmp, arq_indice)

    # para a pagina: so o que existe em disco
    vista = indice.get("vista", {}).get("arquivo")
    if not vista or not (pasta / vista).exists():
        return None
    modelos_pag = {}
    for mid, tipos in indice.get("modelos", {}).items():
        ok = {t: f"galeria/{v['arquivo']}" for t, v in tipos.items() if (pasta / v["arquivo"]).exists()}
        if ok:
            modelos_pag[mid] = ok
    return {"n": len(idx), "lado": LADO, "idx": idx, "nsd": indice.get("nsd"), "coco": indice.get("coco"),
            "vista": f"galeria/{vista}", "modelos": modelos_pag, "legendas": indice.get("legendas", {})}
