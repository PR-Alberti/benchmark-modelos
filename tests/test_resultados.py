"""Testes do resultados.py: le o retrieval de onde cada modelo grava.

    cd tests && python -m unittest test_resultados -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mindeye_ridge import paths, resultados


class Versionados(unittest.TestCase):
    """Os resultados do benchmark que estao no git."""

    def test_frr(self):
        r = resultados.retrieval("subj01_frr_1sess")
        esperado = json.loads((paths.TABLES / "subj01_frr_1sess_frr.json").read_text())["retrieval"]
        self.assertEqual((r["fwd"], r["bwd"]), (esperado["fwd"]["media"], esperado["bwd"]["media"]))
        self.assertIn("FRR", r["fonte"])

    def test_metricas_das_reconstrucoes(self):
        with mock.patch.object(resultados.paths, "EVALS", Path("/nao/existe")):
            r = resultados.retrieval("subj01_me1_1sess")
        self.assertAlmostEqual(r["fwd"], 0.306, places=3)
        self.assertIn("final_evaluations", r["fonte"])

    def test_sem_resultado(self):
        self.assertIsNone(resultados.retrieval("modelo_que_nao_existe"))


class Treino(unittest.TestCase):
    def test_ultima_epoca_do_metrics_csv(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "m").mkdir()
            (Path(d) / "m" / "metrics.csv").write_text(
                "epoch,test/test_fwd_pct_correct,test/test_bwd_pct_correct\n0,0.1,0.05\n1,0.3,0.2\n")
            with mock.patch.object(resultados.paths, "TRAIN_LOGS", Path(d)), \
                    mock.patch.object(resultados.paths, "EVALS", Path(d) / "evals"):
                r = resultados.retrieval("m")
        self.assertEqual((r["fwd"], r["bwd"]), (0.3, 0.2))
        self.assertIn("2 epocas", r["fonte"])


if __name__ == "__main__":
    unittest.main()
