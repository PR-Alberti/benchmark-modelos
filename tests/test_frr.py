"""Testes do frr.py: contra o fracridge de referencia, contra a ridge de forma fechada e nas bordas.

    cd tests && python -m unittest test_frr -v

Os testes de referencia so rodam se `fracridge` estiver instalado (pip install fracridge).
Tudo em float64 na CPU, para comparar com tolerancias apertadas; o de GPU, se houver, em fp32.
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mindeye_ridge import frr

try:
    import fracridge as fracridge_ref
except ImportError:
    fracridge_ref = None

CPU, F64 = "cpu", torch.float64


def dados(n, p, d, seed=0, ruido=1.0):
    """X, Y centrados (como a FRR os usa) de um modelo linear com ruido."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    B = rng.randn(p, d) / np.sqrt(p)
    Y = X @ B + ruido * rng.randn(n, d)
    return X - X.mean(0), Y - Y.mean(0)


def ajuste(X, groups=None, fracs=None, **kw):
    """_Ajuste sobre todas as exibicoes, em float64 na CPU."""
    n = len(X)
    groups = np.arange(n) if groups is None else groups
    fracs = frr.FRACS_DOERIG if fracs is None else fracs
    return frr._Ajuste(torch.from_numpy(X), groups, np.arange(n), fracs, kw.get("rtol", 1e-10),
                       kw.get("bias_step", frr.BIAS_STEP), CPU, F64)


class BaseEspectral(unittest.TestCase):
    def confere(self, n, p):
        X, _ = dados(n, p, 1)
        V, lam = frr.spectral_basis(torch.from_numpy(X), 1e-10)
        s = np.linalg.svd(X, compute_uv=False)
        r = len(lam)
        np.testing.assert_allclose(lam.numpy(), s[:r] ** 2, rtol=1e-8)
        np.testing.assert_allclose((V.T @ V).numpy(), np.eye(r), atol=1e-9)    # colunas ortonormais
        np.testing.assert_allclose((V * lam).numpy() @ V.T.numpy(), X.T @ X, atol=1e-7)

    def test_menos_exibicoes_que_voxels(self):
        self.confere(30, 80)

    def test_mais_exibicoes_que_voxels(self):
        self.confere(120, 40)

    def test_descarta_so_o_posto_que_falta(self):
        # centrada, n < p tem posto n - 1: exatamente uma direcao a menos
        X, _ = dados(30, 80, 1)
        _, lam = frr.spectral_basis(torch.from_numpy(X), 1e-10)
        self.assertEqual(len(lam), 29)
        _, lam = frr.spectral_basis(torch.from_numpy(dados(120, 40, 1)[0]), 1e-10)
        self.assertEqual(len(lam), 40)


@unittest.skipIf(fracridge_ref is None, "fracridge nao instalado")
class ContraReferencia(unittest.TestCase):
    """O pacote de Rokem e Kay e o controle: mesmos alphas e mesmos coeficientes."""

    def compara(self, n, p, d=6, fracs=None):
        fracs = frr.FRACS_DOERIG if fracs is None else fracs
        X, Y = dados(n, p, d, seed=n + p)
        coef, alphas_ref = fracridge_ref.fracridge(X, Y, fracs, jit=False)
        aj = ajuste(X, fracs=fracs)
        m, _ = aj.momentos(torch.from_numpy(Y))
        alphas = aj.alphas(m).numpy()
        np.testing.assert_allclose(alphas, alphas_ref, rtol=1e-6, atol=1e-9)
        # coeficientes: previsao em dados novos, fracao a fracao
        Xn, _ = dados(15, p, d, seed=99)
        Z = aj.projeta(torch.from_numpy(Xn))
        for i in range(len(fracs)):
            prev = (Z @ aj.pesos(m, torch.from_numpy(alphas[i]))).numpy()
            np.testing.assert_allclose(prev, Xn @ coef[:, i, :], rtol=1e-6, atol=1e-8)

    def test_menos_exibicoes_que_voxels(self):
        self.compara(40, 120)

    def test_mais_exibicoes_que_voxels(self):
        self.compara(150, 30)

    def test_fracoes_do_pacote(self):
        self.compara(40, 120, fracs=np.arange(0.1, 1.1, 0.1))


class ContraRidgeDireta(unittest.TestCase):
    """Com o alpha achado, a previsao tem de ser a da ridge resolvida pela equacao normal."""

    def compara(self, n, p):
        X, Y = dados(n, p, 4, seed=7)
        fracs = np.array([0.2, 0.5, 0.9])
        aj = ajuste(X, fracs=fracs)
        m, _ = aj.momentos(torch.from_numpy(Y))
        alphas = aj.alphas(m).numpy()
        Xn, _ = dados(10, p, 4, seed=8)
        Z = aj.projeta(torch.from_numpy(Xn))
        for i in range(len(fracs)):
            for j in range(Y.shape[1]):
                beta = np.linalg.solve(X.T @ X + alphas[i, j] * np.eye(p), X.T @ Y[:, j])
                prev = (Z @ aj.pesos(m, torch.from_numpy(alphas[i]))).numpy()[:, j]
                np.testing.assert_allclose(prev, Xn @ beta, rtol=1e-6, atol=1e-8)

    def test_menos_exibicoes_que_voxels(self):
        self.compara(30, 90)

    def test_mais_exibicoes_que_voxels(self):
        self.compara(120, 25)

    def test_fracao_alcancada(self):
        """|beta(alpha)| / |beta_OLS| ~ fracao pedida; a grade fina aproxima melhor que a de 0,2 decada."""
        X, Y = dados(40, 100, 5, seed=3)
        fracs = np.array([0.1, 0.3, 0.6, 0.9])
        beta_ols = np.linalg.pinv(X) @ Y
        erros = {}
        for passo in (0.2, 0.01):
            aj = ajuste(X, fracs=fracs, bias_step=passo)
            m, _ = aj.momentos(torch.from_numpy(Y))
            alphas = aj.alphas(m).numpy()
            e = 0.0
            for i, f in enumerate(fracs):
                for j in range(Y.shape[1]):
                    beta = np.linalg.solve(X.T @ X + alphas[i, j] * np.eye(100), X.T @ Y[:, j])
                    e = max(e, abs(np.linalg.norm(beta) / np.linalg.norm(beta_ols[:, j]) - f))
            erros[passo] = e
        self.assertLess(erros[0.2], 0.03)
        self.assertLess(erros[0.01], 1e-3)

    def test_fracao_um_e_ols_de_norma_minima(self):
        X, Y = dados(30, 90, 3, seed=5)
        aj = ajuste(X, fracs=np.array([1.0]))
        m, _ = aj.momentos(torch.from_numpy(Y))
        np.testing.assert_array_equal(aj.alphas(m).numpy(), 0.0)
        Xn, _ = dados(8, 90, 3, seed=6)
        prev = (aj.projeta(torch.from_numpy(Xn)) @ aj.pesos(m, torch.zeros(3, dtype=F64))).numpy()
        np.testing.assert_allclose(prev, Xn @ (np.linalg.pinv(X) @ Y), rtol=1e-6, atol=1e-8)


class GradeDeFracoes(unittest.TestCase):
    def test_estendida_contem_a_de_doerig_em_ordem_crescente(self):
        e, d = frr.FRACS_ESTENDIDA, frr.FRACS_DOERIG
        self.assertTrue(np.all(np.diff(e) > 0))
        self.assertTrue(np.allclose(e[-len(d):], d))
        self.assertLess(e[0], d[0])

    def test_fracao_abaixo_da_que_a_grade_alcanca_e_contada(self):
        X, Y = dados(40, 100, 3, seed=11)
        aj = ajuste(X, fracs=np.array([1e-12, 0.5]))
        m, _ = aj.momentos(torch.from_numpy(Y))
        alphas, n_baixo = aj.alphas(m, return_unreachable=True)
        self.assertEqual(n_baixo, 3)                       # a de 1e-12, nos 3 alvos
        self.assertTrue(torch.all(alphas[0] == alphas[0][0]))   # todas caem no maior alpha da grade
        self.assertGreater(float(alphas[0][0]), float(alphas[1][0]))

    def test_fracoes_alcancaveis_nao_sao_contadas(self):
        X, Y = dados(40, 100, 3, seed=12)
        aj = ajuste(X, fracs=np.array([0.2, 0.5, 0.9]))
        m, _ = aj.momentos(torch.from_numpy(Y))
        self.assertEqual(aj.alphas(m, return_unreachable=True)[1], 0)


class AlvosPorImagem(unittest.TestCase):
    """Varias exibicoes da mesma imagem compartilham uma linha de Y, sem copiar."""

    def test_momentos_iguais_aos_do_alvo_replicado(self):
        rng = np.random.RandomState(0)
        n_img, p, d = 20, 60, 5
        groups = rng.randint(0, n_img, size=70)          # repeticoes desiguais, como no NSD
        Yu = rng.randn(n_img, d)
        X = rng.randn(len(groups), p)
        aj = frr._Ajuste(torch.from_numpy(X), groups, np.arange(len(groups)), np.array([0.5]),
                         1e-10, frr.BIAS_STEP, CPU, F64)
        m, ybar = aj.momentos(torch.from_numpy(Yu[aj.linhas]))
        # o mesmo calculo, com Y replicado por exibicao
        Yt = Yu[groups]
        Xc = X - X.mean(0)
        np.testing.assert_allclose(ybar.numpy(), Yt.mean(0), atol=1e-12)
        np.testing.assert_allclose(m.numpy(), aj.V.numpy().T @ Xc.T @ (Yt - Yt.mean(0)), atol=1e-9)


class Dobras(unittest.TestCase):
    def test_nunca_separa_a_mesma_imagem(self):
        groups = np.random.RandomState(1).randint(0, 40, size=200)
        dobras = frr.group_folds(groups, 5, seed=3)
        self.assertEqual(sorted(np.concatenate(dobras)), list(range(200)))      # particao completa
        donas = [set(groups[d]) for d in dobras]
        for a in range(5):
            for b in range(a + 1, 5):
                self.assertFalse(donas[a] & donas[b])
        self.assertLessEqual(max(map(len, donas)) - min(map(len, donas)), 1)     # imagens balanceadas

    def test_semente_fixa_o_sorteio(self):
        groups = np.arange(100) // 3
        a, b = frr.group_folds(groups, 5, 7), frr.group_folds(groups, 5, 7)
        c = frr.group_folds(groups, 5, 8)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a, c)))


class AjusteCompleto(unittest.TestCase):
    def modelo(self, **kw):
        kw = {"n_folds": 4, "chunk": 7, "device": CPU, "dtype": F64, "verbose": False, "seed": 0, **kw}
        return frr.FracRidge(**kw)

    def dados_ruidosos(self):
        """Alvos de ruido crescente: os mais ruidosos devem pedir mais regularizacao."""
        rng = np.random.RandomState(0)
        n, p, d = 160, 120, 12
        X = rng.randn(n, p)
        B = rng.randn(p, d) / np.sqrt(p)
        ruido = np.repeat([0.05, 3.0], d // 2)
        Y = X @ B + ruido * rng.randn(n, d)
        return X, Y, ruido

    def test_ruido_pede_mais_regularizacao(self):
        X, Y, ruido = self.dados_ruidosos()
        m = self.modelo().fit(X, Y, np.arange(len(X)))
        limpos, ruidosos = m.best_[ruido < 1].float().mean(), m.best_[ruido > 1].float().mean()
        self.assertGreater(float(limpos), float(ruidosos))

    def test_fracao_global_e_uma_so(self):
        X, Y, _ = self.dados_ruidosos()
        m = self.modelo(per_target=False).fit(X, Y, np.arange(len(X)))
        self.assertEqual(len(set(m.best_.tolist())), 1)

    def test_preve_melhor_que_ols_com_poucas_exibicoes(self):
        """Com n < p o OLS interpola o ruido; a FRR por CV tem de errar menos contra o modelo verdadeiro."""
        X, Y, _ = self.dados_ruidosos()
        Xt, Yt = X[:100], Y[:100]
        Xn = np.random.RandomState(5).randn(300, X.shape[1])
        m = self.modelo().fit(Xt, Yt, np.arange(100))
        prev = m.predict(Xn, saida=torch.float64).numpy()
        ols = np.linalg.pinv(Xt - Xt.mean(0)) @ (Yt - Yt.mean(0))
        prev_ols = (Xn - Xt.mean(0)) @ ols + Yt.mean(0)
        # o modelo gerador de dados_ruidosos(), sem ruido (mesma sequencia da semente 0)
        rng = np.random.RandomState(0)
        rng.randn(160, 120)
        verdade = Xn @ (rng.randn(120, 12) / np.sqrt(120))
        eqm = lambda a: float(((a - verdade) ** 2).mean())
        self.assertLess(eqm(prev), eqm(prev_ols))

    def test_retoma_dobras_e_recusa_outra_config(self):
        X, Y, _ = self.dados_ruidosos()
        with tempfile.TemporaryDirectory() as tmp:
            a = self.modelo().fit(X, Y, np.arange(len(X)), workdir=tmp)
            self.assertEqual(len([f for f in os.listdir(tmp) if f.startswith("cv_dobra")]), 4)
            b = self.modelo().fit(X, Y, np.arange(len(X)), workdir=tmp)        # le o que ja existe
            self.assertTrue(torch.equal(a.best_, b.best_))
            with self.assertRaises(ValueError):
                self.modelo(n_folds=5).fit(X, Y, np.arange(len(X)), workdir=tmp)

    def test_dobras_de_outros_dados_do_mesmo_formato_sao_recusadas(self):
        X, Y, _ = self.dados_ruidosos()
        with tempfile.TemporaryDirectory() as tmp:
            self.modelo().fit(X, Y, np.arange(len(X)), workdir=tmp)
            X2 = X + np.random.RandomState(9).randn(*X.shape) * 1e-3      # mesmo formato, outros dados
            with self.assertRaises(ValueError):
                self.modelo().fit(X2, Y, np.arange(len(X)), workdir=tmp)
            outros_grupos = np.arange(len(X))[::-1].copy()                  # mesmo X, outro agrupamento
            with self.assertRaises(ValueError):
                self.modelo().fit(X, Y, outros_grupos, workdir=tmp)

    def test_dobra_antiga_sem_impressao_digital_ainda_e_aceita(self):
        X, Y, _ = self.dados_ruidosos()
        with tempfile.TemporaryDirectory() as tmp:
            a = self.modelo().fit(X, Y, np.arange(len(X)), workdir=tmp)
            for f in os.listdir(tmp):                                      # simula o formato anterior
                caminho = os.path.join(tmp, f)
                r = torch.load(caminho)
                del r["meta"]["dados"]
                torch.save(r, caminho)
            b = self.modelo().fit(X, Y, np.arange(len(X)), workdir=tmp)
            self.assertTrue(torch.equal(a.best_, b.best_))

    def test_bloco_nao_muda_o_resultado(self):
        X, Y, _ = self.dados_ruidosos()
        a = self.modelo(chunk=5).fit(X, Y, np.arange(len(X)))
        b = self.modelo(chunk=12).fit(X, Y, np.arange(len(X)))
        self.assertTrue(torch.equal(a.best_, b.best_))
        Xn = np.random.RandomState(2).randn(9, X.shape[1])
        np.testing.assert_allclose(a.predict(Xn, torch.float64).numpy(), b.predict(Xn, torch.float64).numpy(),
                                   rtol=1e-9, atol=1e-10)

    def test_alvos_por_imagem_igual_a_alvos_replicados(self):
        rng = np.random.RandomState(4)
        n_img, p, d = 30, 50, 6
        groups = rng.randint(0, n_img, size=120)
        Yu = rng.randn(n_img, d)
        X = rng.randn(120, p)
        a = self.modelo().fit(X, Yu, groups)
        # replicando o alvo por exibicao (cada exibicao vira sua propria "imagem")
        # as dobras mudam, entao compara so a previsao do ajuste final com uma fracao fixa
        Xn = rng.randn(7, p)
        aj = a._final
        m, ybar = aj.momentos(torch.from_numpy(Yu[aj.linhas]))
        alpha = aj.alphas(m)[3]
        prev = (ybar + aj.projeta(torch.from_numpy(Xn)) @ aj.pesos(m, alpha)).numpy()
        Yt = Yu[groups]
        Xc = X - X.mean(0)
        for j in range(d):
            beta = np.linalg.solve(Xc.T @ Xc + alpha[j].item() * np.eye(p), Xc.T @ (Yt[:, j] - Yt[:, j].mean()))
            np.testing.assert_allclose(prev[:, j], (Xn - X.mean(0)) @ beta + Yt[:, j].mean(), rtol=1e-6, atol=1e-8)

    def test_resumo(self):
        X, Y, _ = self.dados_ruidosos()
        r = self.modelo().fit(X, Y, np.arange(len(X))).resumo()
        self.assertEqual(sum(r["histograma_fracoes"]), Y.shape[1])
        self.assertGreaterEqual(r["cv_r2_escolhida"], max(r["cv_r2_por_fracao"]) - 1e-12)   # por alvo nunca perde da global


@unittest.skipUnless(torch.cuda.is_available(), "sem GPU")
class EmGpu(unittest.TestCase):
    def test_fp32_na_gpu_acompanha_o_fp64_na_cpu(self):
        rng = np.random.RandomState(1)
        n, p, d = 90, 200, 20
        X = rng.randn(n, p)
        Y = X @ (rng.randn(p, d) / np.sqrt(p)) + rng.randn(n, d)
        Xn = rng.randn(12, p)
        kw = {"n_folds": 3, "chunk": 9, "verbose": False}
        a = frr.FracRidge(device=CPU, dtype=F64, **kw).fit(X, Y, np.arange(n))
        b = frr.FracRidge(device="cuda", dtype=torch.float32, **kw).fit(X, Y, np.arange(n))
        agree = float((a.best_ == b.best_).float().mean())
        self.assertGreater(agree, 0.9)       # escolhas iguais, salvo empates no limite da precisao
        pa, pb = a.predict(Xn, torch.float64).numpy(), b.predict(Xn, torch.float64).numpy()
        self.assertLess(np.abs(pa - pb).max() / np.abs(pa).max(), 0.05)


if __name__ == "__main__":
    unittest.main()
