"""Testes do semantica.py com anotacoes sinteticas (nao precisam do NSD nem do COCO).

    cd tests && python -m unittest test_semantica -v
"""
import itertools
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mindeye_ridge import semantica


class CocoFalso:
    """O minimo do semantica.Coco: uma imagem 200 x 100 com as anotacoes dadas."""

    def __init__(self, anots, W=200, H=100):
        self.cats = {1: ("person", "person"), 2: ("dog", "animal"), 3: ("cat", "animal")}
        self.cat_sup = dict(self.cats.values())
        self.supercats = sorted(set(self.cat_sup.values()))
        self.tamanhos = {7: (W, H)}
        self.anots = {7: anots}


def retangulo(cat, x0, y0, x1, y1):
    return {"category_id": cat, "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
            "area": (x1 - x0) * (y1 - y0), "iscrowd": 0}


SEM_CORTE = (0, 0, 0, 0)
RECORTE_CENTRAL = (0, 0, 0.25, 0.25)      # 200 x 100 -> o quadrado central 100 x 100


class Mascaras(unittest.TestCase):
    def test_retangulo_no_recorte_ocupa_a_fracao_certa(self):
        coco = CocoFalso([retangulo(1, 50, 0, 100, 100)])          # metade esquerda do recorte central
        m = semantica.mascaras_na_tela(coco, 7, RECORTE_CENTRAL, lado=100)["person"]
        self.assertAlmostEqual(m.mean(), 0.5, delta=0.02)
        self.assertTrue(m[:, :45].all() and not m[:, 55:].any())

    def test_objeto_fora_do_recorte_nao_conta(self):
        coco = CocoFalso([retangulo(1, 0, 0, 40, 100)])            # so na faixa cortada a esquerda
        m = semantica.mascaras_na_tela(coco, 7, RECORTE_CENTRAL, lado=100)["person"]
        self.assertEqual(m.sum(), 0)

    def test_sobreposicao_conta_uma_vez(self):
        coco = CocoFalso([retangulo(1, 50, 0, 100, 100), retangulo(1, 50, 0, 100, 100)])
        m = semantica.mascaras_na_tela(coco, 7, RECORTE_CENTRAL, lado=100)["person"]
        self.assertAlmostEqual(m.mean(), 0.5, delta=0.02)

    def test_rle_percorre_por_colunas(self):
        # 2 x 3 (h x w), por colunas: col0 = [0, 0], col1 = [1, 1], col2 = [0, 1]
        seg = {"size": [2, 3], "counts": [2, 2, 1, 1]}
        np.testing.assert_array_equal(semantica.rle_para_mascara(seg), [[0, 1, 0], [0, 1, 1]])

    def test_rle_e_poligono_dao_a_mesma_fracao(self):
        H, W = 100, 200
        m = np.zeros((H, W), np.uint8); m[:, 50:100] = 1
        plano = m.T.ravel()                                        # por colunas, comecando em 0
        counts = [len(list(g)) for _, g in itertools.groupby(plano)]
        if plano[0] == 1:
            counts = [0] + counts
        crowd = {"category_id": 1, "segmentation": {"size": [H, W], "counts": counts}, "area": 5000, "iscrowd": 1}
        a = semantica.mascaras_na_tela(CocoFalso([crowd]), 7, RECORTE_CENTRAL, lado=100)["person"].mean()
        b = semantica.mascaras_na_tela(CocoFalso([retangulo(1, 50, 0, 100, 100)]), 7, RECORTE_CENTRAL, lado=100)["person"].mean()
        self.assertAlmostEqual(a, b, delta=0.02)


class Fracoes(unittest.TestCase):
    def test_tabela_tem_categoria_supercategoria_e_area_coco(self):
        coco = CocoFalso([retangulo(1, 50, 0, 100, 100), retangulo(2, 100, 0, 125, 100), retangulo(3, 100, 0, 125, 50)])
        stim = pd.DataFrame({"cocoId": [7], "cropBox": [str(RECORTE_CENTRAL)]}, index=[3])
        fr = semantica.calcula_fracoes([3], stim, coco, lado=100)
        lin = fr.loc[3]
        self.assertAlmostEqual(lin.cat_person, 50, delta=2)
        self.assertAlmostEqual(lin.cat_dog, 25, delta=2)
        self.assertAlmostEqual(lin.sup_animal, 25, delta=2)            # gato dentro do cachorro: uniao
        self.assertAlmostEqual(lin.anotada, 75, delta=2)
        self.assertAlmostEqual(lin.areacoco_person, 25)                # 5000 / 20000 da imagem COCO
        self.assertAlmostEqual(lin.areacoco_animal, 12.5 + 6.25)       # a area antiga soma


def tabela(**colunas):
    return pd.DataFrame(colunas, index=np.arange(len(next(iter(colunas.values())))))


class Regras(unittest.TestCase):
    CLASSES = {"pessoa": ["sup_person"], "animal": ["sup_animal"]}

    def test_exclusiva_exige_minimo_e_pureza(self):
        fr = tabela(sup_person=[30, 30, 1, 30, 0], sup_animal=[0, 5, 30, 1, 0])
        rot = semantica.rotulo_exclusivo(fr, self.CLASSES, minimo=10, maximo_outros=2)
        self.assertEqual(rot.tolist(), ["pessoa", None, "animal", "pessoa", None])

    def test_classe_de_varias_colunas_soma(self):
        fr = tabela(sup_person=[0], sup_animal=[0], cat_dog=[6], cat_cat=[6])
        rot = semantica.rotulo_exclusivo(fr, {"pessoa": ["sup_person"], "pet": ["cat_dog", "cat_cat"]}, minimo=10)
        self.assertEqual(rot.tolist(), ["pet"])

    def test_coluna_inexistente_avisa(self):
        with self.assertRaises(KeyError):
            semantica.rotulo_exclusivo(tabela(sup_person=[1]), {"x": ["sup_naoexiste"]})

    def test_prioridade_como_no_process_data_roi(self):
        fr = tabela(sup_person=[6, 1, 1, 1], sup_animal=[50, 3, 0, 0], sup_vehicle=[0, 0, 20, 0], sup_food=[0, 0, 5, 20])
        self.assertEqual(semantica.rotulo_prioridade(fr).tolist(), ["pessoa", "animal", "outro", None])


class Download(unittest.TestCase):
    def test_baixa_extrai_e_apaga_o_zip(self):
        def falso(url, destino):
            if destino.name.endswith(".zip"):
                with zipfile.ZipFile(destino, "w") as z:
                    for sp in semantica.SPLITS_COCO:
                        z.writestr(f"annotations/instances_{sp}2017.json", "{}")
                    z.writestr("annotations/captions_train2017.json", "{}")
            else:
                Path(destino).write_text("nsdId\n")
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(semantica.urllib.request, "urlretrieve", side_effect=falso) as baixa:
            stim, anotacoes = semantica.baixa_se_faltar(d)
            self.assertTrue(stim.exists() and all(a.exists() for a in anotacoes))
            sem = Path(d) / "semantica"
            self.assertFalse((sem / "annotations_trainval2017.zip").exists())
            self.assertFalse((sem / "annotations" / "captions_train2017.json").exists())
            semantica.baixa_se_faltar(d)                      # ja esta tudo: nao baixa de novo
            self.assertEqual(baixa.call_count, 2)


if __name__ == "__main__":
    unittest.main()
