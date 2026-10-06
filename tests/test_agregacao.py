"""Testes da agregacao das repeticoes (mindeye_ridge.agregacao) e de como o MindEye1 a aplica.

    cd tests && python -m unittest test_agregacao -v
"""
import os
import sys
import unittest

import numpy as np
import torch

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))
sys.path.insert(0, os.path.join(RAIZ, "mindeye1", "src"))
from mindeye_ridge import agregacao
from mindeye_ridge import dataset_controlado as dc
import nsd_benchmark

# 3 amostras: com 3, 2 e 1 repeticoes reais, completadas ate 3 como no MindEye1 ([a, b, a], [a, a, a])
REPS = [torch.tensor([[1., 1.], [2., 2.], [6., 6.]]),
        torch.tensor([[10., 0.], [20., 0.], [10., 0.]]),
        torch.tensor([[5., 5.], [5., 5.], [5., 5.]])]
VOX3, N_REP = torch.stack(REPS), torch.tensor([3, 2, 1])


class Seleciona(unittest.TestCase):
    def test_media_so_das_repeticoes_reais(self):
        m = agregacao.seleciona(VOX3, N_REP, "media")
        torch.testing.assert_close(m, torch.tensor([[3., 3.], [15., 0.], [5., 5.]]))

    def test_exibicoes_devolve_a_propria(self):
        torch.testing.assert_close(agregacao.seleciona(VOX3, N_REP, "exibicoes"), VOX3[:, 0])

    def test_sorteio_uniforme_entre_as_reais(self):
        g = torch.Generator().manual_seed(0)
        vistos = torch.stack([agregacao.seleciona(VOX3, N_REP, "sorteio", g) for _ in range(3000)])
        frac_a = (vistos[:, 1, 0] == 10).float().mean().item()       # 2a amostra: a ou b, 50% cada
        self.assertAlmostEqual(frac_a, 0.5, delta=0.04)
        self.assertTrue(set(vistos[:, 0, 0].tolist()) <= {1., 2., 6.})
        self.assertTrue((vistos[:, 2] == 5).all())

    def test_combinacao_fica_entre_as_repeticoes_reais(self):
        g = torch.Generator().manual_seed(1)
        for _ in range(300):
            c = agregacao.seleciona(VOX3, N_REP, "combinacao", g)
            self.assertTrue(1 <= c[0, 0] <= 6 and 10 <= c[1, 0] <= 20)
            torch.testing.assert_close(c[2], torch.tensor([5., 5.]))   # uma repeticao so: ela mesma

    def test_reprodutivel_e_mantem_o_dtype(self):
        a = agregacao.seleciona(VOX3.half(), N_REP, "combinacao", torch.Generator().manual_seed(5))
        b = agregacao.seleciona(VOX3.half(), N_REP, "combinacao", torch.Generator().manual_seed(5))
        self.assertEqual(a.dtype, torch.float16)
        torch.testing.assert_close(a, b)

    def test_modo_invalido(self):
        with self.assertRaises(ValueError):
            agregacao.seleciona(VOX3, N_REP, "mediana")
        with self.assertRaisesRegex(ValueError, "agregacao"):
            dc.ConfigDataset(agregacao="mediana")

    def test_grupos_e_tres(self):
        ids, g = agregacao.grupos([7, 3, 7, 7], [10, 11, 12, 13])
        self.assertEqual(ids.tolist(), [3, 7])
        self.assertEqual([x.tolist() for x in g], [[11], [10, 12, 13]])
        self.assertEqual(agregacao.tres([4, 5]).tolist(), [4, 5, 4])


def dados(modo, exibicoes=False):
    """Treino falso no formato do nsd_benchmark.carrega: 4 imagens, voxels = id da imagem."""
    if exibicoes:
        return {"voxel": torch.arange(6.)[:, None, None].expand(-1, 3, 2), "imagem": torch.arange(4.)[:, None] * 100,
                "indice_imagem": torch.tensor([0, 0, 1, 2, 3, 3]), "coco": torch.tensor([0, 0, 1, 2, 3, 3]),
                "n_rep": torch.ones(6, dtype=torch.long), "agregacao": modo}
    v = torch.tensor([[1., 2., 3.], [4., 5., 4.], [7., 7., 7.], [8., 9., 10.]])[..., None].expand(-1, -1, 2)
    return {"voxel": v, "imagem": torch.arange(4.)[:, None] * 100, "coco": torch.arange(4),
            "n_rep": torch.tensor([3, 2, 1, 3]), "agregacao": modo}


class LotesMindEye1(unittest.TestCase):
    def lotes(self, d, embaralha=True):
        return list(nsd_benchmark.Lotes(d, 2, embaralha=embaralha, device="cpu", seed=0))

    def test_sem_agregacao_e_o_original(self):
        for voxel, img, coco in self.lotes(dados(None)):
            self.assertFalse(torch.equal(voxel[:, 0], voxel[:, 1]) and torch.equal(voxel[:, 0], voxel[:, 2]))

    def test_media_chega_igual_nas_tres_posicoes(self):
        medias = {0: 2., 1: 4.5, 2: 7., 3: 9.}
        for voxel, img, coco in self.lotes(dados("media")):
            for v, c in zip(voxel, coco.tolist()):
                self.assertTrue(torch.all(v == medias[c]))

    def test_exibicoes_indexa_a_imagem_de_cada_amostra(self):
        n = 0
        for voxel, img, coco in self.lotes(dados("exibicoes", exibicoes=True)):
            torch.testing.assert_close(img[:, 0], coco.float() * 100)
            n += len(coco)
        self.assertEqual(n, 6)

    def test_validacao_nao_agrega(self):
        for voxel, _, _ in self.lotes(dados("media"), embaralha=False):
            self.assertFalse(torch.all(voxel[:, 0] == voxel[:, 1]))


if __name__ == "__main__":
    unittest.main()
