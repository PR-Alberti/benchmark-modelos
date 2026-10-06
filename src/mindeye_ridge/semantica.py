"""Rotulo semantico das imagens do NSD pela fracao da tela que cada categoria COCO ocupa.

O NSD nao mostra a imagem COCO inteira: corta um quadrado (cropBox) e o exibe em 425 x 425
pixels. Aqui cada anotacao COCO (poligono, ou RLE nas anotacoes iscrowd) e desenhada dentro
desse recorte, e a fracao de uma categoria e a da uniao das suas mascaras, em % da tela. A
exploracao que motivou as regras esta em notebooks/classificacao_semantica.ipynb.

    fr = tabela_fracoes(data_path, subj)                # cache em <dados>/semantica/
    rot = rotulo_exclusivo(fr, {"pessoa": ["sup_person"], "animal": ["sup_animal"]}, 10, 2)

A tabela tem uma linha por imagem do sujeito (indice NSD de 0 a 72.999, o mesmo do behav e do
coco_images_224_float16.hdf5) e as colunas cat_<categoria> (80), sup_<supercategoria> (12),
anotada (qualquer objeto) e areacoco_<supercategoria> (soma de `area` sobre a imagem COCO
inteira, a medida antiga, para comparar).
"""
import ast
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import nsd_data

LADO_TELA = 425                         # o estimulo do NSD tem 425 x 425 pixels na tela
URL_STIM = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
URL_COCO = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
SPLITS_COCO = ("train", "val")          # o NSD usa imagens dos dois


def diretorio(data_path):
    return Path(data_path) / "semantica"


def baixa_se_faltar(data_path):
    """nsd_stim_info_merged.csv (bucket publico do NSD) e instances_*2017.json do COCO (~250 MB)."""
    sem = diretorio(data_path)
    sem.mkdir(parents=True, exist_ok=True)
    stim = sem / "nsd_stim_info_merged.csv"
    if not stim.exists():
        print("baixando", URL_STIM, flush=True)
        urllib.request.urlretrieve(URL_STIM, stim)
    anotacoes = [sem / "annotations" / f"instances_{s}2017.json" for s in SPLITS_COCO]
    if not all(a.exists() for a in anotacoes):
        zip_path = sem / "annotations_trainval2017.zip"
        if not zip_path.exists():
            print("baixando", URL_COCO, "(~250 MB)", flush=True)
            urllib.request.urlretrieve(URL_COCO, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for a in anotacoes:
                z.extract(f"annotations/{a.name}", sem)
    return stim, anotacoes


def le_stim_info(data_path):
    """nsd_stim_info_merged.csv indexado pelo indice NSD (0 a 72.999)."""
    stim, _ = baixa_se_faltar(data_path)
    return pd.read_csv(stim).set_index("nsdId").drop(columns="Unnamed: 0")


class Coco:
    """Anotacoes de instancias do COCO 2017 (train + val), indexadas por imagem."""

    def __init__(self, caminhos):
        self.cats, self.tamanhos, self.anots = {}, {}, {}
        for c in caminhos:
            with open(c) as f:
                d = json.load(f)
            self.cats.update({k["id"]: (k["name"], k["supercategory"]) for k in d["categories"]})
            self.tamanhos.update({im["id"]: (im["width"], im["height"]) for im in d["images"]})
            for a in d["annotations"]:
                self.anots.setdefault(a["image_id"], []).append(a)
        self.cat_sup = dict(self.cats.values())                 # categoria -> supercategoria
        self.supercats = sorted(set(self.cat_sup.values()))


def rle_para_mascara(seg):
    """RLE nao comprimido do COCO (anotacoes iscrowd), que percorre a imagem por colunas."""
    h, w = seg["size"]
    plano = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in seg["counts"]:
        plano[pos:pos + c] = val
        pos, val = pos + c, val ^ 1
    return plano.reshape(w, h).T


def mascaras_na_tela(coco, coco_id, crop, lado=LADO_TELA):
    """{categoria: mascara booleana lado x lado} no recorte que o NSD mostrou.

    crop = (topo, base, esquerda, direita): fracoes da imagem COCO cortadas fora (cropBox).
    """
    topo, base, esq, dir_ = crop
    W, H = coco.tamanhos[coco_id]
    x0, y0 = esq * W, topo * H
    largura = W * (1 - esq - dir_)                 # o recorte e quadrado
    escala = lado / largura
    telas = {}
    for a in coco.anots.get(coco_id, []):
        nome = coco.cats[a["category_id"]][0]
        tela = telas.setdefault(nome, Image.new("L", (lado, lado), 0))
        seg = a["segmentation"]
        if isinstance(seg, list):
            d = ImageDraw.Draw(tela)
            for p in seg:
                if len(p) >= 6:
                    d.polygon([((x - x0) * escala, (y - y0) * escala) for x, y in zip(p[0::2], p[1::2])], fill=1)
        else:
            m = rle_para_mascara(seg)[round(y0):round(y0 + largura), round(x0):round(x0 + largura)]
            tela.paste(1, mask=Image.fromarray(m * 255).resize((lado, lado), Image.NEAREST))
    return {n: np.asarray(t, dtype=bool) for n, t in telas.items()}


def calcula_fracoes(ids, stim, coco, lado=LADO_TELA):
    """Tabela de fracoes (em % da tela) das imagens `ids` (indices NSD)."""
    linhas = []
    for nid in ids:
        coco_id = int(stim.at[nid, "cocoId"])
        W, H = coco.tamanhos[coco_id]
        masc = mascaras_na_tela(coco, coco_id, ast.literal_eval(stim.at[nid, "cropBox"]), lado)
        lin = {"nsd_id": int(nid), "coco_id": coco_id, "n_objetos": len(coco.anots.get(coco_id, []))}
        por_sup, tudo = {}, np.zeros((lado, lado), bool)
        for nome, m in masc.items():
            lin[f"cat_{nome}"] = 100 * m.mean()
            s = coco.cat_sup[nome]
            por_sup[s] = por_sup[s] | m if s in por_sup else m
            tudo |= m
        lin.update({f"sup_{s}": 100 * m.mean() for s, m in por_sup.items()})
        lin["anotada"] = 100 * tudo.mean()
        for a in coco.anots.get(coco_id, []):
            k = f"areacoco_{coco.cats[a['category_id']][1]}"
            lin[k] = lin.get(k, 0) + 100 * a["area"] / (W * H)
        linhas.append(lin)
    colunas = (["coco_id", "n_objetos", "anotada"] + [f"sup_{s}" for s in coco.supercats]
               + [f"cat_{c}" for c in coco.cat_sup] + [f"areacoco_{s}" for s in coco.supercats])
    return pd.DataFrame(linhas).set_index("nsd_id").reindex(columns=colunas).fillna(0)


def tabela_fracoes(data_path, subj):
    """Fracoes da tela de todas as imagens do sujeito (treino das 40 sessoes e teste), com cache.

    O cache fica em <dados>/semantica/subj0<N>_fracao_tela.csv. Calcular leva ~20 s, mais o
    download das anotacoes na primeira vez.
    """
    cache = diretorio(data_path) / f"subj0{subj}_fracao_tela.csv"
    treino = nsd_data.exibicoes_treino(data_path, subj, 40)["imagem"]
    ids_teste, _ = nsd_data.exibicoes_teste(data_path, subj)
    ids = np.union1d(np.unique(treino), ids_teste)
    if cache.exists():
        fr = pd.read_csv(cache, index_col="nsd_id")
        if not len(np.setdiff1d(ids, fr.index)):
            return fr.loc[ids]
    stim = le_stim_info(data_path)
    _, anotacoes = baixa_se_faltar(data_path)
    fr = calcula_fracoes(ids, stim, Coco(anotacoes))
    fr.to_csv(cache)
    return fr


def supercategorias(fr):
    """So as colunas sup_*, com o nome da supercategoria."""
    return fr[[c for c in fr.columns if c.startswith("sup_")]].rename(columns=lambda c: c[4:])


def fracao_classes(fr, classes):
    """% da tela de cada classe. Classe com varias colunas: a soma delas, limitada a 100
    (aproxima a uniao; a sobreposicao entre categorias diferentes e pequena)."""
    faltam = sorted({c for cols in classes.values() for c in cols} - set(fr.columns))
    if faltam:
        raise KeyError(f"colunas inexistentes na tabela de fracoes: {faltam} (use sup_<supercategoria> ou cat_<categoria>)")
    return pd.DataFrame({c: fr[list(cols)].sum(axis=1).clip(upper=100) for c, cols in classes.items()})


def rotulo_exclusivo(fr, classes, minimo=10, maximo_outros=2):
    """Classe de cada imagem, ou None (descartada).

    `classes`: {nome: [colunas de fr]}. A imagem e da classe c se c ocupa >= `minimo`% da tela
    e todas as outras classes ficam abaixo de `maximo_outros`%. Nenhuma ou mais de uma: None.
    O que nao e classe (raquete, prato) nao conta contra a pureza.
    """
    f = fracao_classes(fr, classes)
    rot = pd.Series(None, index=fr.index, dtype=object)
    for c in classes:
        outras = f.drop(columns=c)
        rot[(f[c] >= minimo) & (outras < maximo_outros).all(axis=1)] = c
    return _none(rot)


def rotulo_prioridade(fr, pessoa_min=5, animal_min=2, outros_validos=("vehicle", "furniture", "outdoor")):
    """A regra do process_data_ROI, medida na tela: pessoa se >= pessoa_min; senao animal se
    >= animal_min; senao "outro" se a supercategoria dominante entre as restantes estiver em
    `outros_validos`; senao None."""
    sup = supercategorias(fr)
    resto = sup.drop(columns=["person", "animal"])
    dom_resto = resto.idxmax(axis=1).where(resto.max(axis=1) > 0)
    rot = pd.Series(None, index=fr.index, dtype=object)
    rot[dom_resto.isin(outros_validos)] = "outro"
    rot[sup.animal >= animal_min] = "animal"
    rot[sup.person >= pessoa_min] = "pessoa"
    return _none(rot)


def _none(rot):
    """Descartadas como None (o pandas troca por NaN na atribuicao parcial)."""
    return rot.astype(object).where(rot.notna(), None)
