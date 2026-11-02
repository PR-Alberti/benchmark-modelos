"""Testes do protocolo do efeito do tempo na sessao: selecao dos trials, retrieval e agregacao.

    cd tests && PYTHONPATH=../src python -m unittest test_tiredness -v
"""
import unittest
from math import comb

import numpy as np
import pandas as pd

from mindeye_ridge import tiredness


def exibicoes(*linhas):
    """Trials (sessao, run, imagem), com a linha em betas na ordem dada."""
    ex = pd.DataFrame(linhas, columns=["sessao", "run", "imagem"])
    ex["trial"], ex["beta"], ex["compartilhada"] = 1, range(len(ex)), False
    return tiredness.anota_repeticoes(ex)


def manifesto(sessoes, imagem, beta, sessao):
    return {"config": {"sessoes": sessoes}, "imagem": np.array(imagem), "beta": np.array(beta),
            "sessao": np.array(sessao)}


class TestSelecao(unittest.TestCase):
    def setUp(self):
        self.ex = exibicoes((1, 1, 10), (1, 2, 11), (2, 1, 10), (2, 1, 12), (2, 12, 12), (3, 1, 11))
        self.treino = manifesto(1, [10, 11], [0, 1], [0, 0])

    def test_so_as_sessoes_depois_do_treino(self):
        teste = tiredness.seleciona_teste(self.ex, self.treino)
        self.assertEqual(sorted(teste["sessao"].unique()), [2, 3])

    def test_marca_imagem_vista_no_treino_mesmo_em_outra_sessao(self):
        teste = tiredness.seleciona_teste(self.ex, self.treino)
        self.assertEqual(teste["no_treino"].tolist(), [True, False, False, True])

    def test_trial_do_treino_no_teste_e_erro(self):
        with self.assertRaisesRegex(ValueError, "esta no treino"):
            tiredness.seleciona_teste(self.ex, manifesto(1, [10], [2], [0]))

    def test_manifesto_incoerente_e_erro(self):
        with self.assertRaisesRegex(ValueError, "declara"):
            tiredness.seleciona_teste(self.ex, manifesto(1, [10], [0], [1]))

    def test_sem_sessoes_de_teste_e_erro(self):
        with self.assertRaisesRegex(ValueError, "nao ha sessoes"):
            tiredness.seleciona_teste(self.ex, manifesto(3, [10], [0], [0]))


class TestRepeticoes(unittest.TestCase):
    def test_inedito_vale_por_sessao(self):
        ex = exibicoes((1, 1, 5), (1, 2, 5), (2, 1, 5), (2, 1, 6))
        self.assertEqual(ex["inedito"].tolist(), [True, False, True, True])
        self.assertEqual(ex["exibicoes_antes"].tolist(), [0, 1, 2, 0])

    def test_ordena_pela_apresentacao(self):
        ex = pd.DataFrame({"sessao": [1, 1], "run": [1, 1], "imagem": [5, 5], "trial": [2, 1], "beta": [9, 3]})
        self.assertEqual(tiredness.anota_repeticoes(ex)["inedito"].tolist(), [True, False])

    def test_linha_em_betas_repetida_e_erro(self):
        ex = pd.DataFrame({"sessao": [1, 1], "run": [1, 1], "imagem": [5, 6], "trial": [1, 2], "beta": [3, 3]})
        with self.assertRaisesRegex(ValueError, "mesma linha"):
            tiredness.anota_repeticoes(ex)


class TestAcerto(unittest.TestCase):
    def test_primeira_do_pool_sempre_acerta(self):
        self.assertAlmostEqual(tiredness.acerto_top1([0], 1000)[0], 1.0)

    def test_igual_a_conta_exata(self):
        for m in (1, 7, 15):
            exato = comb(19 - m, 4) / comb(19, 4)
            self.assertAlmostEqual(tiredness.acerto_top1([m], 20, 5)[0], exato, places=10)

    def test_igual_ao_sorteio(self):
        rng = np.random.default_rng(0)
        n_pool, m, n = 40, 6, 20000
        # posicoes 0..m-1 sao as melhores que a certa; uma distratora entre elas derruba o acerto
        hits = sum(rng.choice(n_pool - 1, 9, replace=False).min() >= m for _ in range(n))
        self.assertAlmostEqual(tiredness.acerto_top1([m], n_pool, 10)[0], hits / n, delta=0.01)

    def test_mais_melhores_do_que_cabe_e_zero(self):
        self.assertEqual(tiredness.acerto_top1([16], 20, 5)[0], 0.0)
        self.assertGreater(tiredness.acerto_top1([15], 20, 5)[0], 0.0)

    def test_pool_menor_que_as_candidatas_e_erro(self):
        with self.assertRaisesRegex(ValueError, "pool"):
            tiredness.acerto_top1([0], 299)


def trials_exemplo():
    t = pd.DataFrame({"sessao": [2, 2, 2, 2, 3, 3], "run": [1, 1, 12, 5, 2, 11],
                      "no_treino": [False, True, False, False, False, False],
                      "inedito": [True, True, False, True, True, True],
                      "bwd_top1": [0.8, np.nan, 0.4, 0.6, 0.5, 0.3], "cosseno": [.6, np.nan, .5, .55, .7, .6]})
    return t


class TestAgregacao(unittest.TestCase):
    def test_por_run_conta_e_exclui(self):
        r = tiredness.por_run(trials_exemplo()).set_index(["sessao", "run"])
        self.assertEqual(len(r), 5)
        self.assertEqual(r.loc[(2, 1), "n_trials"], 2)
        self.assertEqual(r.loc[(2, 1), "n_excluidos_treino"], 1)
        self.assertEqual(r.loc[(2, 1), "n_avaliados"], 1)
        self.assertAlmostEqual(r.loc[(2, 1), "bwd_top1"], 0.8)
        self.assertEqual(r.loc[(2, 5), "grupo"], "meio")
        self.assertEqual(r.loc[(2, 12), "grupo"], "fim")

    def test_por_run_ineditos(self):
        r = tiredness.por_run(trials_exemplo()).set_index(["sessao", "run"])
        self.assertEqual(r.loc[(2, 12), "n_ineditos"], 0)
        self.assertTrue(np.isnan(r.loc[(2, 12), "bwd_top1_ineditos"]))

    def test_run_so_com_excluidos_fica_vazio(self):
        t = trials_exemplo()
        r = tiredness.por_run(t[t["no_treino"]]).iloc[0]
        self.assertEqual(r["n_avaliados"], 0)
        self.assertTrue(np.isnan(r["bwd_top1"]))

    def test_contraste_dentro_da_sessao(self):
        t = trials_exemplo()
        c = tiredness.contraste(t[~t["no_treino"]], "bwd_top1")
        self.assertAlmostEqual(c["diferenca"], ((0.8 - 0.4) + (0.5 - 0.3)) / 2)
        self.assertEqual(c["sessoes"], 2)
        self.assertEqual((c["n_inicio"], c["n_fim"]), (2, 2))
        self.assertLessEqual(c["ic95"][0], c["diferenca"])
        self.assertGreaterEqual(c["ic95"][1], c["diferenca"])

    def test_contraste_ignora_sessao_sem_os_dois_lados(self):
        t = pd.DataFrame({"sessao": [1, 1, 2], "run": [1, 12, 1], "bwd_top1": [0.9, 0.5, 0.1]})
        c = tiredness.contraste(t, "bwd_top1")
        self.assertEqual(c["sessoes"], 1)
        self.assertAlmostEqual(c["diferenca"], 0.4)

    def test_contraste_sem_sessao_pareada_e_erro(self):
        t = pd.DataFrame({"sessao": [1, 2], "run": [1, 12], "bwd_top1": [0.9, 0.5]})
        with self.assertRaisesRegex(ValueError, "nenhuma sessao"):
            tiredness.contraste(t, "bwd_top1")


if __name__ == "__main__":
    unittest.main()
