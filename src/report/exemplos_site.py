"""Exemplos do controle semantico para a apresentacao: imagens de treino do subj01 em que a
supercategoria k ocupa x% da tela, com a mascara de k por cima.

Para cada supercategoria, as imagens de treino se dividem em faixas de % da tela em torno das
PARADAS (bordas na media geometrica entre paradas vizinhas), e N_POR_FAIXA delas sao sorteadas em
cada faixa. A fracao mostrada e a do semantica.json do site (a mesma tabela que o treino usa); a
mascara e refeita aqui com semantica.mascaras_na_tela, sobre as anotacoes COCO.

Grava em results/site/exemplos/:
    <sup>.webp            as fotos de uma supercategoria, lado a lado (o recorte que o NSD mostrou)
    <sup>__mascara.png    na mesma ordem, a camada que escurece o que nao e <sup> e contorna o que e
    recorte.webp          uma imagem COCO inteira, para mostrar o recorte quadrado do NSD
    indice.json           o que a pagina le

As fotos vem do coco_images_224_float16.hdf5 (o estimulo como foi mostrado) ou, sem ele, da imagem
COCO baixada e recortada pelo cropBox. Nada disso muda com os resultados: so e refeito quando os
parametros abaixo ou as fracoes do semantica.json mudam, e numa maquina sem as anotacoes COCO fica
o que ja estava gravado.
"""
import hashlib
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mindeye_ridge import paths

from .semantica_site import SUPERCATEGORIAS, decodifica

PARADAS = (1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90)   # % da tela
PISO = 0.5                                                 # abaixo disso a classe nao conta como presente
N_POR_FAIXA, LADO, SEMENTE, QUALIDADE = 4, 224, 0, 70
ESCURO = 0.62            # opacidade do que nao e a classe, na camada da mascara
LARGURA_RECORTE = 480
URL_IMAGEM = "http://images.cocodataset.org/{split}/{coco:012d}.jpg"
VERSAO = 1


def faixas():
    """[(lo, hi)] de cada parada: [PISO, ...) ate [..., 100]."""
    p = np.asarray(PARADAS, float)
    bordas = np.r_[PISO, np.sqrt(p[:-1] * p[1:]), 100.0]
    return [(float(bordas[i]), float(bordas[i + 1])) for i in range(len(p))]


def _params(semantica):
    return {"paradas": list(PARADAS), "piso": PISO, "n": N_POR_FAIXA, "lado": LADO, "semente": SEMENTE,
            "versao": VERSAO, "fracoes": hashlib.sha1(semantica["dados"].encode()).hexdigest()[:16]}


def ids_treino(stim, subj=1):
    """As imagens de treino do sujeito (as dele fora das 1.000 compartilhadas), em ordem crescente:
    a mesma ordem do semantica.json."""
    s = stim[(stim[f"subject{subj}"] == 1) & ~stim["shared1000"].astype(bool)]
    return np.sort(s.index.values)


def escolhe(fracoes, semente=SEMENTE):
    """{sup: (contagem por faixa, [(linha, faixa)])}: as imagens sorteadas em cada faixa.

    fracoes: (n, 12) em % da tela. Cada supercategoria tem o seu gerador, entao mudar uma nao muda
    o sorteio das outras.
    """
    fx = faixas()
    out = {}
    for j, s in enumerate(SUPERCATEGORIAS):
        rng = np.random.default_rng([semente, j])
        f = fracoes[:, j]
        contagem, linhas = [], []
        for b, (lo, hi) in enumerate(fx):
            cand = np.flatnonzero((f >= lo) & ((f < hi) if b < len(fx) - 1 else (f <= hi)))
            contagem.append(int(len(cand)))
            if len(cand):
                pick = rng.choice(cand, min(N_POR_FAIXA, len(cand)), replace=False)
                linhas += [(int(i), b) for i in sorted(pick, key=lambda i: f[i])]
        out[s] = (contagem, linhas)
    return out


# --------------------------------------------------------------------------- imagens

def _baixa(url, tentativas=3):
    erro = None
    for _ in range(tentativas):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except OSError as e:
            erro = e
    raise erro


def _caixa(crop, W, H):
    """cropBox (topo, base, esquerda, direita) -> (x0, y0, lado) em pixels da imagem COCO."""
    topo, _, esq, dir_ = crop
    return esq * W, topo * H, W * (1 - esq - dir_)


def _coco_inteira(coco_id, split):
    from PIL import Image
    return Image.open(io.BytesIO(_baixa(URL_IMAGEM.format(split=split, coco=coco_id)))).convert("RGB")


def _recorta(im, crop, lado=LADO):
    from PIL import Image
    x0, y0, l = _caixa(crop, *im.size)
    return im.crop((round(x0), round(y0), round(x0 + l), round(y0 + l))).resize((lado, lado), Image.LANCZOS)


class Fotos:
    """O estimulo de cada imagem NSD, em LADO x LADO: do hdf5 do MindEye2 ou do COCO."""

    def __init__(self, stim, data=paths.DATA):
        import ast
        self.stim, self.parse = stim, ast.literal_eval
        h5 = Path(data) / "coco_images_224_float16.hdf5"
        self.h5 = None
        if h5.exists():
            import h5py
            self.h5 = h5py.File(h5, "r")["images"]

    def crop(self, nsd):
        return self.parse(self.stim.at[nsd, "cropBox"])

    def __call__(self, ids):
        from PIL import Image
        if self.h5 is not None:
            out = {}
            for i in ids:
                a = np.asarray(self.h5[int(i)], dtype=np.float32).transpose(1, 2, 0)
                im = Image.fromarray((a.clip(0, 1) * 255).round().astype(np.uint8))
                out[int(i)] = im if im.size == (LADO, LADO) else im.resize((LADO, LADO), Image.LANCZOS)
            return out

        def uma(i):
            s = self.stim.loc[int(i)]
            return int(i), _recorta(_coco_inteira(int(s["cocoId"]), s["cocoSplit"]), self.crop(int(i)))
        with ThreadPoolExecutor(8) as ex:
            return dict(ex.map(uma, ids))


def camada_mascara(mascara, lado=LADO, escuro=ESCURO):
    """RGBA lado x lado: escurece o que nao e a classe (com a borda suavizada) e contorna a classe."""
    from PIL import Image, ImageFilter
    m = Image.fromarray(np.asarray(mascara, dtype=np.uint8) * 255)
    cobre = np.asarray(m.resize((lado, lado), Image.BOX), dtype=np.float32) / 255
    dentro = cobre >= 0.5
    fora_perto = np.asarray(Image.fromarray(dentro.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0
    contorno = fora_perto & ~dentro
    alfa = np.round(8 * (1 - cobre)) / 8 * escuro          # 9 niveis: comprime bem
    rgba = np.zeros((lado, lado, 4), np.uint8)
    rgba[..., :3] = 12
    rgba[..., 3] = np.round(255 * alfa).astype(np.uint8)
    rgba[contorno] = (255, 255, 255, 235)
    return Image.fromarray(rgba, "RGBA")


def mascara_classe(coco, coco_id, crop, sup):
    """Uniao das mascaras das categorias da supercategoria `sup`, na tela 425 x 425."""
    from mindeye_ridge import semantica
    masc = semantica.mascaras_na_tela(coco, coco_id, crop)
    tudo = np.zeros((semantica.LADO_TELA, semantica.LADO_TELA), bool)
    for nome, m in masc.items():
        if coco.cat_sup[nome] == sup:
            tudo |= m
    return tudo


def _salva(im, destino, **kw):
    tmp = destino.with_name(destino.stem + ".tmp" + destino.suffix)
    im.save(tmp, **kw)
    os.replace(tmp, destino)


def _tira(imagens, lado=LADO, modo="RGB"):
    from PIL import Image
    tira = Image.new(modo, (lado * max(1, len(imagens)), lado))
    for k, im in enumerate(imagens):
        tira.paste(im, (k * lado, 0))
    return tira


# --------------------------------------------------------------------------- pagina

def para_pagina(indice, pasta):
    """O indice com caminhos relativos ao site, so com o que existe em disco; None se nada."""
    if not indice:
        return None
    classes = {}
    for s, c in indice.get("classes", {}).items():
        if (pasta / c["fotos"]).exists() and (pasta / c["mascaras"]).exists():
            classes[s] = {**c, "fotos": f"exemplos/{c['fotos']}", "mascaras": f"exemplos/{c['mascaras']}"}
    if not classes:
        return None
    rec = indice.get("recorte")
    if rec and not (pasta / rec["arquivo"]).exists():
        rec = None
    return {"lado": indice["lado"], "paradas": indice["paradas"], "faixas": indice["faixas"], "classes": classes,
            "recorte": {**rec, "arquivo": f"exemplos/{rec['arquivo']}"} if rec else None}


def atualiza(pasta_site, semantica, gerar=True, baixar=False, log=print):
    """Os exemplos para a pagina, refeitos se os parametros ou as fracoes mudaram.

    gerar=False so le o que ja esta gravado. Sem as anotacoes COCO e o nsd_stim_info em
    <dados>/semantica, so gera com baixar=True (baixa ~250 MB; make_site.py --exemplos).
    """
    pasta = Path(pasta_site) / "exemplos"
    arq = pasta / "indice.json"
    try:
        indice = json.loads(arq.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        indice = None
    if not semantica:
        return para_pagina(indice, pasta)
    params = _params(semantica)
    em_dia = indice and indice.get("params") == params and para_pagina(indice, pasta) and all(
        (pasta / c["fotos"]).exists() for c in indice["classes"].values())
    if em_dia or not gerar:
        return para_pagina(indice, pasta)

    from mindeye_ridge import semantica as sem
    d = sem.diretorio(paths.DATA)
    faltam = not (d / "nsd_stim_info_merged.csv").exists() or not all(
        (d / "annotations" / f"instances_{s}2017.json").exists() for s in sem.SPLITS_COCO)
    if faltam and not baixar:
        log("exemplos: faltam as anotações COCO em <dados>/semantica; ficam os exemplos gravados "
            "(make_site.py --exemplos baixa e gera)")
        return para_pagina(indice, pasta)
    try:
        novo = gera(pasta, semantica, params, log=log)
    except Exception as e:                       # os exemplos so enfeitam a apresentacao
        log(f"aviso: exemplos não gerados ({type(e).__name__}: {e})")
        return para_pagina(indice, pasta)
    return para_pagina(novo, pasta)


def gera(pasta, semantica, params, log=print):
    from mindeye_ridge import semantica as sem
    pasta.mkdir(parents=True, exist_ok=True)
    stim = sem.le_stim_info(paths.DATA)
    ids = ids_treino(stim)
    _, pixels = decodifica(semantica)
    if len(ids) != semantica["n"] or [int(ids[0]), int(ids[-1])] != list(semantica["ids"]):
        raise ValueError("as imagens de treino do nsd_stim_info não batem com o semantica.json")
    if list(semantica["sup"]) != SUPERCATEGORIAS:
        raise ValueError("supercategorias do semantica.json fora da ordem esperada")
    fracoes = 100 * (np.where(pixels == 0xFFFFFF, 0, pixels) / semantica["tela"])
    escolha = escolhe(fracoes)

    log("exemplos: anotações COCO")
    _, anot = sem.baixa_se_faltar(paths.DATA)
    coco = sem.Coco(anot)
    fotos = Fotos(stim)
    todas = sorted({int(ids[i]) for _, linhas in escolha.values() for i, _ in linhas})
    log(f"exemplos: {len(todas)} imagens ({'hdf5' if fotos.h5 is not None else 'COCO'})")
    imgs = fotos(todas)

    fx = faixas()
    classes = {}
    for s, (contagem, linhas) in escolha.items():
        exemplos, camadas = [], []
        for i, b in linhas:
            nsd = int(ids[i])
            coco_id = int(stim.at[nsd, "cocoId"])
            camadas.append(camada_mascara(mascara_classe(coco, coco_id, fotos.crop(nsd), s)))
            exemplos.append({"nsd": nsd, "coco": coco_id, "faixa": b, "pix": [int(p) for p in pixels[i]]})
        _salva(_tira([imgs[e["nsd"]] for e in exemplos]), pasta / f"{s}.webp", format="WEBP", quality=QUALIDADE,
               method=6)
        _salva(_tira(camadas, modo="RGBA"), pasta / f"{s}__mascara.png", format="PNG", optimize=True)
        classes[s] = {"fotos": f"{s}.webp", "mascaras": f"{s}__mascara.png", "contagem": contagem,
                      "exemplos": exemplos}
        log(f"exemplos: {s} ({len(exemplos)})")

    recorte = _recorte(pasta, classes, stim, fotos)
    indice = {"params": params, "lado": LADO, "paradas": list(PARADAS), "faixas": [list(f) for f in fx],
              "tela": semantica["tela"], "classes": classes, "recorte": recorte}
    tmp = pasta / "indice.tmp"
    tmp.write_text(json.dumps(indice, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, pasta / "indice.json")
    return indice


def _recorte(pasta, classes, stim, fotos, classe="animal"):
    """A imagem COCO inteira de um exemplo, com a caixa que o NSD recortou: o de animal entre 15% e
    40% da tela cujo recorte joga fora a maior parte da imagem."""
    from PIL import Image
    ex = classes.get(classe, {}).get("exemplos", [])
    fx = faixas()
    cand = [k for k, e in enumerate(ex) if 15 <= fx[e["faixa"]][0] and fx[e["faixa"]][1] <= 45]
    if not cand:
        return None

    def descarte(k):
        t, b, e, d = fotos.crop(ex[k]["nsd"])
        return t + b + e + d
    k = max(cand, key=descarte)
    nsd = ex[k]["nsd"]
    s = stim.loc[nsd]
    try:
        im = _coco_inteira(int(s["cocoId"]), s["cocoSplit"])
    except OSError:
        return None
    W, H = im.size
    x0, y0, l = _caixa(fotos.crop(nsd), W, H)
    escala = LARGURA_RECORTE / W
    im = im.resize((LARGURA_RECORTE, round(H * escala)), Image.LANCZOS)
    _salva(im, pasta / "recorte.webp", format="WEBP", quality=78, method=6)
    return {"arquivo": "recorte.webp", "largura": im.size[0], "altura": im.size[1], "classe": classe, "exemplo": k,
            "caixa": [x0 / W, y0 / H, l / W, l / H], "original": [W, H]}
