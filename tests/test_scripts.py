"""Testes das protecoes dos scripts de treino com DATASET (dataset controlado), sem treinar nada.

    cd tests && python -m unittest test_scripts -v

Os casos que devem falhar falham antes de chamar o Python; os demais so exercitam as funcoes do
scripts/common.sh num bash a parte.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def bash(codigo, env=None, cwd=None):
    """Roda `codigo` num bash com o common.sh carregado; devolve (codigo de saida, stdout + stderr)."""
    base = {k: v for k, v in os.environ.items() if k not in ("DATASET", "MODEL_NAME", "NUM_SESSIONS", "RESUME")}
    r = subprocess.run(["bash", "-c", f'source "{SCRIPTS}/common.sh"\n{codigo}'], env={**base, **(env or {})},
                       cwd=cwd or REPO, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def script(nome, *args, env=None):
    base = {k: v for k, v in os.environ.items() if k not in ("DATASET", "MODEL_NAME", "NUM_SESSIONS", "RESUME")}
    r = subprocess.run(["bash", str(SCRIPTS / nome), *args], env={**base, **(env or {})},
                       capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


class PreparaDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manifesto = Path(self.tmp.name) / "ds.json"
        self.manifesto.write_text("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def test_sem_dataset_nao_faz_nada(self):
        cod, out = bash('prepara_dataset; echo "[$DATASET]"')
        self.assertEqual(cod, 0)
        self.assertIn("[]", out)

    def test_dataset_exige_model_name(self):
        cod, out = bash("prepara_dataset; echo passou", env={"DATASET": str(self.manifesto)})
        self.assertNotEqual(cod, 0)
        self.assertIn("MODEL_NAME", out)
        self.assertNotIn("passou", out)

    def test_dataset_inexistente(self):
        cod, out = bash("prepara_dataset", env={"DATASET": "/nao/existe.json", "MODEL_NAME": "x"})
        self.assertNotEqual(cod, 0)
        self.assertIn("nao existe", out)

    def test_caminho_relativo_vira_absoluto(self):
        cod, out = bash('prepara_dataset; echo "[$DATASET]"', env={"DATASET": "ds.json", "MODEL_NAME": "x"},
                        cwd=self.tmp.name)
        self.assertEqual(cod, 0, out)
        self.assertIn(f"[{self.manifesto}]", out)


class ConfereDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "modelo"
        self.a, self.b = Path(self.tmp.name) / "a.json", Path(self.tmp.name) / "b.json"
        self.a.write_text('{"a": 1}')
        self.b.write_text('{"b": 2}')

    def tearDown(self):
        self.tmp.cleanup()

    def confere(self, retoma, dataset=None):
        env = {"DATASET": str(dataset)} if dataset else {}
        return bash(f'confere_dataset "{self.dir}" {retoma}', env=env)

    def test_retomar_com_o_mesmo_manifesto_segue(self):
        self.assertEqual(self.confere(0, self.a)[0], 0)
        self.assertEqual(self.confere(1, self.a)[0], 0)

    def test_retomar_com_outro_manifesto_ou_sem_nenhum_para(self):
        self.confere(0, self.a)
        cod, out = self.confere(1, self.b)
        self.assertNotEqual(cod, 0)
        self.assertIn("outro dataset", out)
        self.assertNotEqual(self.confere(1)[0], 0)

    def test_treinar_do_zero_troca_o_registro(self):
        self.confere(0, self.a)
        self.assertEqual(self.confere(0, self.b)[0], 0)
        self.assertEqual(self.confere(1, self.b)[0], 0)

    def test_modelo_antigo_sem_registro_conta_como_completo(self):
        self.dir.mkdir()
        (self.dir / "last.pth").write_bytes(b"")
        self.assertNotEqual(self.confere(1, self.a)[0], 0)
        self.assertEqual(self.confere(1)[0], 0)
        self.assertEqual((self.dir / "dataset.sha1").read_text().strip(), "completo")


class Scripts(unittest.TestCase):
    """Os tres scripts de treino recusam DATASET sem MODEL_NAME antes de rodar qualquer coisa."""

    def test_dataset_sem_model_name(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            for nome, args in (("run_frr.sh", ()), ("run_ridgeonly_prior.sh", ()), ("me1_run.sh", ("train",))):
                with self.subTest(nome):
                    cod, out = script(nome, *args, env={"DATASET": f.name, "NUM_SESSIONS": "40"})
                    self.assertNotEqual(cod, 0)
                    self.assertIn("defina MODEL_NAME", out)

    def test_mindeye1_dataset_exige_num_sessions(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            cod, out = script("me1_run.sh", "train", env={"DATASET": f.name, "MODEL_NAME": "x"})
        self.assertNotEqual(cod, 0)
        self.assertIn("NUM_SESSIONS", out)

    def test_mindeye1_low_level_do_modelo_controlado(self):
        """Com DATASET o low-level e <MODEL_NAME>_lowlevel, nao o do benchmark (bash -x mostra a escolha)."""
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            base = {k: v for k, v in os.environ.items() if k not in ("DATASET", "AE_NAME")}
            for env, esperado in (({"DATASET": f.name, "MODEL_NAME": "exp"}, "AE_NAME=exp_lowlevel"),
                                  ({}, "AE_NAME=subj01_me1_lowlevel_40sess")):
                r = subprocess.run(["bash", "-x", str(SCRIPTS / "me1_run.sh"), "etapa_inexistente"],
                                   env={**base, "NUM_SESSIONS": "40", **env}, capture_output=True, text=True)
                self.assertIn(esperado, r.stderr)

    def test_benchmark_ignora_dataset(self):
        for nome in ("run_benchmark.sh", "run_me1_benchmark.sh"):
            texto = (SCRIPTS / nome).read_text()
            fonte = texto.index('source "$(dirname "$0")/common.sh"')
            self.assertGreater(texto.index("unset DATASET"), fonte, nome)


if __name__ == "__main__":
    unittest.main()
