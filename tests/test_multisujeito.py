"""Testes do FRR multi-sujeito (src/mindeye_ridge/multisujeito.py) em dados sinteticos, na CPU.

    cd tests && python -m unittest test_multisujeito -v
"""
import os
import tempfile
import unittest

import numpy as np
import torch

from mindeye_ridge import frr
from mindeye_ridge import multisujeito as ms

DEV = "cpu"


def sintetico(n_sujeitos=3, voxels=(40, 50, 30), imagens=150, reps=2, D=120, k=5, ruido=0.3, seed=0):
    """Sujeitos que veem imagens diferentes; voxels e alvo saem do mesmo latente de k dimensoes:
    x = z A_s + e, y = z W* + e. Devolve (sujeitos, W*, gerador do latente)."""
    rng = np.random.RandomState(seed)
    W = rng.randn(k, D)
    sujeitos = []
    for s in range(n_sujeitos):
        Z = rng.randn(imagens, k)
        Y = (Z @ W + ruido * rng.randn(imagens, D)).astype(np.float32)
        A = rng.randn(k, voxels[s])
        linhas = np.repeat(np.arange(imagens), reps)
        X = (Z[linhas] @ A + ruido * rng.randn(len(linhas), voxels[s])).astype(np.float32)
        sujeitos.append(ms.Sujeito(f"s{s}", lambda X=X: torch.from_numpy(X), linhas, Y))
    return sujeitos, W


class Validacao(unittest.TestCase):
    def test_uma_imagem_nunca_fica_dos_dois_lados(self):
        linhas = np.repeat(np.arange(50), 3)
        val = ms.mascara_validacao(linhas, 0.2, 0)
        self.assertEqual(len(np.unique(linhas[val])), 10)
        self.assertFalse(set(linhas[val]) & set(linhas[~val]))


class Pca(unittest.TestCase):
    def test_igual_ao_svd_exato(self):
        (s,), _ = sintetico(1, (30,), imagens=80, D=60)
        b = ms.base_espectral(s.carrega(), s.linhas, np.zeros(len(s.linhas), bool))
        Q, k_ef = ms.pca_inicial([b], [s.Y], [1.0], k=4, D=60, chunk=16, device=DEV, seed=0)
        self.assertEqual(Q.shape, (4, 60))
        np.testing.assert_allclose(Q @ Q.T, np.eye(4), atol=1e-4)
        y = s.Y[b["img_tr"]]
        ybar = (b["cont_tr"][:, None] * y).sum(0) / b["n_tr"]
        M = np.sqrt(b["cont_tr"])[:, None] * (y - ybar)
        Vt = np.linalg.svd(M, full_matrices=False)[2][:4]
        np.testing.assert_allclose(Q.T @ Q, Vt.T @ Vt, atol=1e-3)      # mesmo subespaco

    def test_poucas_imagens_limitam_k(self):
        (s,), _ = sintetico(1, (30,), imagens=12, D=60)
        b = ms.base_espectral(s.carrega(), s.linhas, np.zeros(len(s.linhas), bool))
        Q, k_ef = ms.pca_inicial([b], [s.Y], [1.0], k=50, D=60, chunk=16, device=DEV, seed=0)
        self.assertLessEqual(k_ef, 11)                                 # 12 imagens centradas
        self.assertTrue(np.isfinite(Q).all())


class PassoW(unittest.TestCase):
    """Com fracao 1 (sem regularizacao), o passo W e o minimo quadrado ponderado das exibicoes empilhadas."""

    def setUp(self):
        self.suj, _ = sintetico(2, (40, 30), imagens=60, D=50)
        self.bases, self.R = [], []
        rng = np.random.RandomState(1)
        for s in self.suj:
            val = ms.mascara_validacao(s.linhas, 0.2, 0)
            b = ms.base_espectral(s.carrega(), s.linhas, val, guarda_V=True)
            R = torch.from_numpy(rng.randn(b["voxels"], 6).astype(np.float32))
            self.bases.append((b, val, R))

    def test_igual_ao_ajuste_direto(self):
        bases = [b for b, _, _ in self.bases]
        Bs = [b["V"].T @ R for b, _, R in self.bases]                  # R = V B (V completa: n > p)
        w, wv = ms._pesos(bases, "sujeito")
        W = np.zeros((6, 50), np.float32)
        _, perda, _ = ms.passo_W(bases, [s.Y for s in self.suj], Bs, w, wv, [1.0], 50, 16, DEV, W)
        Zs, Ys, ps, Zv, Yv, pv = [], [], [], [], [], []
        for (b, val, R), s, ws, wvs in zip(self.bases, self.suj, w, wv):
            X = s.carrega()
            ybar = (b["cont_tr"][:, None] * s.Y[b["img_tr"]]).sum(0) / b["n_tr"]
            Zs.append(((X[~val] - b["media"]) @ R).numpy()); Ys.append(s.Y[s.linhas[~val]] - ybar)
            ps.append(np.full((~val).sum(), ws))
            Zv.append(((X[val] - b["media"]) @ R).numpy()); Yv.append(s.Y[s.linhas[val]] - ybar)
            pv.append(np.full(val.sum(), wvs))
        Z, Yt, p = np.concatenate(Zs), np.concatenate(Ys), np.sqrt(np.concatenate(ps))
        W_dir = np.linalg.lstsq(p[:, None] * Z, p[:, None] * Yt, rcond=None)[0]
        np.testing.assert_allclose(W, W_dir, rtol=1e-3, atol=1e-3)
        Zv, Yv, pv = np.concatenate(Zv), np.concatenate(Yv), np.concatenate(pv)
        perda_dir = (pv[:, None] * (Zv @ W_dir - Yv) ** 2).sum() / (pv[:, None] * Yv ** 2).sum()
        self.assertAlmostEqual(perda, perda_dir, places=4)


class Escala(unittest.TestCase):
    def test_preserva_o_produto_e_ortonormaliza(self):
        rng = np.random.RandomState(0)
        W = rng.randn(6, 40).astype(np.float32)
        B = torch.from_numpy(rng.randn(10, 6).astype(np.float32))
        L, P = ms.escala(torch.from_numpy(W @ W.T).double())
        Q = np.zeros((P.shape[0], 40), np.float32)
        ms.projeta(W, P, Q, 16, DEV)
        np.testing.assert_allclose(Q @ Q.T, np.eye(6), atol=1e-4)
        np.testing.assert_allclose((B @ L).numpy() @ Q, B.numpy() @ W, rtol=1e-3, atol=1e-3)


class PreTreino(unittest.TestCase):
    def ajusta(self, d, sujeitos, **kw):
        return ms.FRRMultiSujeito(k=5, iteracoes=kw.pop("iteracoes", 4), chunk=32, device=DEV, **kw).ajusta(sujeitos, d)

    def test_pre_treino_ajuda_o_sujeito_com_pouco_dado(self):
        """O sujeito novo (10 imagens, 200 voxels ruidosos) com a leitura dos outros vence o FRR so com
        ele no teste. No sintetico o alvo ja e de posto 5, entao o ganho e pequeno; o teste confere o
        mecanismo (a leitura transferida ajuda), nao o tamanho."""
        outros, W = sintetico(3, (40, 50, 30), imagens=150, seed=0)
        rng = np.random.RandomState(7)
        A = rng.randn(5, 200)
        def novo(n):
            Z = rng.randn(n, 5)
            return (Z @ A + rng.randn(n, 200)).astype(np.float32), (Z @ W + 0.3 * rng.randn(n, W.shape[1])).astype(np.float32)
        Xn, Yn = novo(10)
        Xt, Yt = novo(200)
        with tempfile.TemporaryDirectory() as d:
            m = self.ajusta(d, outros)
            self.assertEqual(m.Q.shape, (5, W.shape[1]))
            idx = np.arange(10)
            T, ybar = ms.alvo_reduzido(Yn, idx, np.ones(10), m.Q, 32, DEV)
            fino = frr.FracRidge(n_folds=3, device=DEV, verbose=False).fit(torch.from_numpy(Xn), T, idx)
            prev = ms.expande(fino.predict(torch.from_numpy(Xt), saida=torch.float32), ybar, m.Q, 32, DEV,
                              torch.float32).numpy()
        so = frr.FracRidge(n_folds=3, device=DEV, verbose=False).fit(torch.from_numpy(Xn), Yn, idx)
        prev_so = so.predict(torch.from_numpy(Xt), saida=torch.float32).numpy()
        r2 = lambda p: 1 - ((p - Yt) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum()
        self.assertGreater(r2(prev), r2(prev_so) + 0.02)

    def test_retoma_e_recusa_outra_configuracao(self):
        sujeitos, _ = sintetico(2, (40, 30), imagens=60, seed=3)
        with tempfile.TemporaryDirectory() as d:
            m1 = self.ajusta(d, sujeitos, iteracoes=2)
            Q1 = np.array(m1.Q)
            m2 = self.ajusta(d, sujeitos, iteracoes=2)                   # tudo em disco: nada recalculado
            np.testing.assert_array_equal(Q1, np.array(m2.Q))
            self.assertEqual(len(m2.historico), len(m1.historico))
            with self.assertRaisesRegex(ValueError, "outro pre-treino"):
                self.ajusta(d, sujeitos, iteracoes=2, pesos="exibicao")

    def test_historico_e_melhor_iteracao(self):
        sujeitos, _ = sintetico(3, (40, 50, 30), imagens=100, seed=5)
        with tempfile.TemporaryDirectory() as d:
            m = self.ajusta(d, sujeitos, iteracoes=3)
        perdas = [h["perda_validacao"] for h in m.historico]
        self.assertEqual(m.melhor["perda_validacao"], min(perdas))
        self.assertLess(m.melhor["perda_validacao"], 0.5)


if __name__ == "__main__":
    unittest.main()
