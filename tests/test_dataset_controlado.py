"""Testes do dataset_controlado.py: com exibicoes sinteticas e, se os dados existirem, com o subj01.

    cd tests && python -m unittest test_dataset_controlado -v
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mindeye_ridge import dataset_controlado as dc
from mindeye_ridge import nsd_data, paths


def exibicoes_falsas(n_imagens=60, sessoes=6, seed=0):
    """Cada imagem com 3 exibicoes em sessoes crescentes; algumas so aparecem a partir da sessao 3."""
    rng = np.random.default_rng(seed)
    imagem, sessao = [], []
    for i in range(n_imagens):
        s = np.sort(rng.choice(sessoes, 3, replace=False))
        imagem += [100 + i] * 3
        sessao += s.tolist()
    ordem = np.lexsort((imagem, sessao))
    imagem, sessao = np.asarray(imagem)[ordem], np.asarray(sessao)[ordem]
    return {"imagem": imagem, "beta": np.arange(len(imagem)), "sessao": sessao}


def fracoes_falsas(ids):
    """Metade pessoa (30%), um terco animal (30%), o resto sem nada."""
    k = np.arange(len(ids))
    return pd.DataFrame({"sup_person": np.where(k % 6 < 3, 30.0, 0.0),
                         "sup_animal": np.where(k % 6 >= 4, 30.0, 0.0),
                         "sup_vehicle": 0.0, "sup_food": 0.0, "sup_furniture": 0.0}, index=ids)


EX = exibicoes_falsas()
FR = fracoes_falsas(np.unique(EX["imagem"]))
CLASSES = {"pessoa": ["sup_person"], "animal": ["sup_animal"]}


def monta(**kw):
    return dc.monta(dc.ConfigDataset(**kw), data_path=None, exibicoes=EX, fracoes=FR)


class Selecao(unittest.TestCase):
    def test_padrao_e_o_pool_inteiro_na_mesma_ordem(self):
        sub = monta(sessoes=6)
        for k in ("imagem", "beta", "sessao"):
            np.testing.assert_array_equal(sub[k], EX[k])

    def test_sessoes_recortam_o_pool(self):
        sub = monta(sessoes=2)
        self.assertTrue((sub["sessao"] < 2).all())
        self.assertEqual(len(sub["imagem"]), (EX["sessao"] < 2).sum())

    def test_n_imagens_e_repeticoes(self):
        sub = monta(sessoes=6, n_imagens=20, repeticoes=2, semente=3)
        imagens, reps = np.unique(sub["imagem"], return_counts=True)
        self.assertEqual(len(imagens), 20)
        self.assertTrue((reps == 2).all())
        self.assertEqual(sub["resumo"], {"exibicoes": 40, "imagens_unicas": 20, "repeticoes": {2: 20}, "sessoes_usadas": len(np.unique(sub["sessao"]))})

    def test_repeticoes_exige_exibicoes_suficientes_no_pool(self):
        sub = monta(sessoes=2, repeticoes=2)
        pool = pd.Series(EX["imagem"][EX["sessao"] < 2]).value_counts()
        self.assertEqual(set(sub["imagem"]), set(pool.index[pool >= 2]))

    def test_mesma_semente_mesmo_subconjunto_outra_semente_outro(self):
        a, b, c = (monta(sessoes=6, n_imagens=15, repeticoes=1, semente=s) for s in (1, 1, 2))
        np.testing.assert_array_equal(a["beta"], b["beta"])
        self.assertFalse(np.array_equal(a["beta"], c["beta"]))

    def test_classes_filtram_e_balanceado_sorteia_por_classe(self):
        sub = monta(sessoes=6, classes=CLASSES, n_imagens=8, balanceado=True)
        self.assertEqual(sub["resumo"]["imagens_por_classe"], {"animal": 8, "pessoa": 8})
        rot_img = pd.Series(sub["rotulo"]).groupby(sub["imagem"]).first()
        self.assertTrue((rot_img == FR.loc[rot_img.index].pipe(lambda f: np.where(f.sup_person > 0, "pessoa", "animal"))).all())

    def test_sem_n_imagens_pega_todas_da_classe(self):
        sub = monta(sessoes=6, classes=CLASSES)
        self.assertEqual(sub["resumo"]["imagens_por_classe"], {"animal": 20, "pessoa": 30})

    def test_prioridade_com_lista_de_classes(self):
        sub = monta(sessoes=6, regra="prioridade", classes=["animal"])
        self.assertEqual(sub["resumo"]["imagens_por_classe"], {"animal": 20})


class Erros(unittest.TestCase):
    def test_pedir_mais_do_que_ha(self):
        with self.assertRaises(ValueError):
            monta(sessoes=6, n_imagens=61)
        with self.assertRaisesRegex(ValueError, "por classe"):
            monta(sessoes=6, classes=CLASSES, n_imagens=21, balanceado=True)

    def test_campos_invalidos(self):
        for kw in ({"sessoes": 0}, {"repeticoes": 4}, {"regra": "outra"}, {"balanceado": True, "n_imagens": 5},
                   {"regra": "prioridade", "classes": {"x": ["sup_person"]}}):
            with self.assertRaises(ValueError, msg=str(kw)):
                dc.ConfigDataset(**kw)

    def test_chave_desconhecida_no_dicionario(self):
        with self.assertRaisesRegex(ValueError, "repeticao"):
            dc.ConfigDataset.de_dict({"sessoes": 10, "repeticao": 1})


class Manifesto(unittest.TestCase):
    def test_salva_e_carrega(self):
        sub = monta(sessoes=6, classes=CLASSES, n_imagens=5, balanceado=True, repeticoes=1)
        with tempfile.TemporaryDirectory() as d:
            dc.salva(sub, f"{d}/ds.json")
            volta = dc.carrega(f"{d}/ds.json")
        for k in ("imagem", "beta", "sessao"):
            np.testing.assert_array_equal(volta[k], sub[k])
        self.assertEqual(list(volta["rotulo"]), list(sub["rotulo"]))
        self.assertEqual(volta["config"], sub["config"])

    def test_assinatura_depende_so_da_config(self):
        self.assertEqual(dc.ConfigDataset(n_imagens=5).assinatura(), dc.ConfigDataset(n_imagens=5).assinatura())
        self.assertNotEqual(dc.ConfigDataset(n_imagens=5).assinatura(), dc.ConfigDataset(n_imagens=6).assinatura())


DADOS = os.path.exists(f"{paths.DATA}/wds/subj01/train/39.tar")


@unittest.skipUnless(DADOS, "sem os dados do subj01")
class Subj01(unittest.TestCase):
    def test_padrao_reproduz_as_n_primeiras_sessoes(self):
        for n in (1, 10):
            sub = dc.monta(dc.ConfigDataset(sessoes=n), paths.DATA)
            ref = nsd_data.exibicoes_treino(paths.DATA, 1, n)
            for k in ("imagem", "beta", "sessao"):
                np.testing.assert_array_equal(sub[k], ref[k])

    def test_exibicoes_confere_sessoes_do_manifesto(self):
        sub = dc.monta(dc.ConfigDataset(sessoes=5, n_imagens=50, repeticoes=1), paths.DATA)
        with tempfile.TemporaryDirectory() as d:
            dc.salva(sub, f"{d}/ds.json")
            ex = dc.exibicoes(paths.DATA, 1, 5, f"{d}/ds.json")
            np.testing.assert_array_equal(ex["beta"], sub["beta"])
            with self.assertRaises(ValueError):
                dc.exibicoes(paths.DATA, 1, 2, f"{d}/ds.json")
        _, ids_teste = None, nsd_data.exibicoes_teste(paths.DATA, 1)[0]
        self.assertFalse(set(sub["imagem"]) & set(ids_teste))


if __name__ == "__main__":
    unittest.main()
