"""Caminhos do repositorio: o layout esperado e a variavel MINDEYE_DATA."""
import importlib
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mindeye_ridge import paths


class Paths(unittest.TestCase):
    def test_raiz_tem_o_layout_esperado(self):
        for d in ("src", "scripts", "setup", "tests", "results", "notebooks", "docs"):
            self.assertTrue((paths.REPO / d).is_dir(), d)
        self.assertEqual(paths.SRC, paths.REPO / "src")
        self.assertTrue((paths.SRC / "mindeye_ridge" / "paths.py").exists())

    def test_saidas_ficam_em_results(self):
        self.assertEqual(paths.EVALS, paths.REPO / "results" / "evals")
        self.assertEqual(paths.TABLES, paths.REPO / "results" / "tables")
        self.assertTrue(paths.TABLES.is_dir())

    def test_dados_vem_da_variavel_de_ambiente(self):
        antes = os.environ.get("MINDEYE_DATA")
        try:
            os.environ["MINDEYE_DATA"] = "/outro/lugar"
            importlib.reload(paths)
            self.assertEqual(paths.DATA, Path("/outro/lugar"))
            del os.environ["MINDEYE_DATA"]
            importlib.reload(paths)
            self.assertEqual(paths.DATA, Path("~/mindeyev2").expanduser())
        finally:
            if antes is not None:
                os.environ["MINDEYE_DATA"] = antes
            importlib.reload(paths)

    def test_vendored_entra_no_path_uma_vez(self):
        paths.add_vendored_to_path()
        paths.add_vendored_to_path()
        self.assertEqual(sys.path.count(str(paths.SRC / "generative_models")), 1)
        self.assertTrue((paths.SRC / "generative_models" / "sgm").is_dir())


if __name__ == "__main__":
    unittest.main()
