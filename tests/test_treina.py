"""Testes do treina.py e do experimentos.py: traducao dos hiperparametros e validacao, sem treinar.

    cd tests && python -m unittest test_treina -v
"""
import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import treina
from experimentos import EXPERIMENTOS
from mindeye_ridge import dataset_controlado as dc
from mindeye_ridge import paths


class Hiper(unittest.TestCase):
    def test_frr_vira_extra(self):
        h = treina.MODELOS["frr"].completa({"folds": 3, "fracs": [0.1, 0.5], "global_fraction": True})
        env, args = treina.MODELOS["frr"].traduz(h)
        self.assertEqual(args, [])
        self.assertEqual(env["EXTRA"], "--folds=3 --grid=doerig --fracs 0.1 0.5 --seed=42 --chunk=8192 --global_fraction")

    def test_mindeye2_vira_variaveis_do_script(self):
        h = treina.MODELOS["mindeye2"].completa({"num_epochs": 20, "prior": False, "frozen_fp16": True})
        env, args = treina.MODELOS["mindeye2"].traduz(h)
        self.assertEqual(args, [])
        self.assertEqual(env, {"HIDDEN_DIM": "1024", "NUM_EPOCHS": "20", "MAX_LR": "0.0003", "PRIOR": "0",
                               "BLURRY": "0", "SEED": "42", "FROZEN": "1"})

    def test_mindeye1_vira_opcoes_do_treino(self):
        h = treina.MODELOS["mindeye1"].completa({"num_epochs": 10, "batch_size": 8})
        env, args = treina.MODELOS["mindeye1"].traduz(h)
        self.assertEqual(env, {"PAPER": "0", "SAVE_EVERY": "1"})
        self.assertEqual(args, ["--num_epochs=10", "--max_lr=0.0003", "--seed=42", "--mixup_pct=0.33", "--batch_size=8"])

    def test_hiper_invalido(self):
        with self.assertRaisesRegex(ValueError, "desconhecidos"):
            treina.MODELOS["frr"].completa({"num_epochs": 3})
        with self.assertRaisesRegex(ValueError, "num_epochs"):
            treina.MODELOS["mindeye2"].completa({"num_epochs": 1})
        with self.assertRaisesRegex(ValueError, "4096"):
            treina.MODELOS["mindeye2"].completa({"blurry": True})
        with self.assertRaisesRegex(ValueError, "num_epochs"):
            treina.MODELOS["mindeye1"].completa({"num_epochs": 1})

    def test_mindeye1_precisa_de_um_lote_de_imagens(self):
        m = treina.MODELOS["mindeye1"]
        m.valida_dataset(m.completa({}), {"imagens_unicas": 16})
        with self.assertRaisesRegex(ValueError, "lote de 16"):
            m.valida_dataset(m.completa({}), {"imagens_unicas": 15})
        with self.assertRaisesRegex(ValueError, "lote de 32"):
            m.valida_dataset(m.completa({"paper": True}), {"imagens_unicas": 20})


MAQUINA = {"HOME", "PWD", "BASH_SOURCE", "MINDEYE_ENV", "MINDEYE_DATA", "CUDA_VISIBLE_DEVICES",
           "PYTORCH_CUDA_ALLOC_CONF", "ME1_DATA"}      # configuracao da maquina, nao do experimento


def variaveis_lidas(*arquivos):
    """Variaveis que um script le do ambiente: as com valor padrao (${X:-...}, ${X:+...}) e as
    usadas sem nunca serem atribuidas no script nem no common.sh."""
    texto = "\n".join(Path(a).read_text() for a in arquivos)
    texto = "\n".join(l for l in texto.splitlines() if not l.lstrip().startswith("#"))
    usadas = set(re.findall(r"\$\{?([A-Z_][A-Z0-9_]*)", texto))
    com_padrao = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*):?[-+=]", texto))
    atribuidas = set(re.findall(r"(?:^|[\s;(])(?:export |local )?([A-Z_][A-Z0-9_]*)=", texto, re.M))
    return (com_padrao | (usadas - atribuidas)) - MAQUINA


class Ambiente(unittest.TestCase):
    def test_treina_controla_toda_variavel_que_os_scripts_leem(self):
        comum = paths.REPO / "scripts" / "common.sh"
        for nome, modelo in treina.MODELOS.items():
            with self.subTest(nome):
                lidas = variaveis_lidas(paths.REPO / "scripts" / modelo.script[0], comum)
                self.assertEqual(lidas, set(modelo.variaveis))

    def test_variavel_do_shell_nao_vaza_para_o_treino(self):
        exp = {"modelo": "mindeye2", "hiper": {"num_epochs": 5},
               "dataset": {"sessoes": 3, "n_imagens": 40, "repeticoes": 1}}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(treina.paths, "TRAIN_LOGS", Path(d)), \
                mock.patch.object(treina.dc, "monta", return_value=MANIFESTO_FALSO), \
                mock.patch.object(treina, "_commit", return_value="abc"), \
                mock.patch.dict(os.environ, {"BATCH_SIZE": "8", "RESUME": "1", "EXTRA": "--x", "OUTRA": "fica"}), \
                mock.patch("subprocess.run") as run, redirect_stdout(io.StringIO()) as out:
            treina.roda("exp", exp)
        env = run.call_args.kwargs["env"]
        self.assertNotIn("BATCH_SIZE", env)
        self.assertNotIn("RESUME", env)
        self.assertEqual(env["OUTRA"], "fica")                          # o resto do ambiente segue
        self.assertIn("ignorando do ambiente", out.getvalue())
        self.assertIn("BATCH_SIZE=8", out.getvalue())

    def test_frr_nao_sorteia_por_epoca(self):
        for modo in ("sorteio", "combinacao"):
            with self.assertRaisesRegex(ValueError, "nao se aplica"):
                treina.roda("x", {"modelo": "frr", "dataset": {"agregacao": modo}}, dry_run=True)

    def test_so_subj01(self):
        with self.assertRaisesRegex(ValueError, "subj01"):
            treina.roda("x", {"modelo": "frr", "dataset": {"subj": 2}}, dry_run=True)


MANIFESTO_FALSO = {"imagem": np.arange(3), "beta": np.arange(3), "sessao": np.zeros(3, int),
                   "rotulo": np.array([None] * 3, dtype=object), "config": {}, "resumo": {"exibicoes": 3}}


class Experimentos(unittest.TestCase):
    def test_todos_os_experimentos_sao_validos(self):
        for nome, exp in EXPERIMENTOS.items():
            with self.subTest(nome):
                self.assertIn(exp["modelo"], treina.MODELOS)
                treina.MODELOS[exp["modelo"]].completa(exp.get("hiper"))
                dc.ConfigDataset.de_dict(exp.get("dataset"))

    def test_experimento_mal_formado(self):
        for exp, msg in (({"modelo": "svm"}, "modelo"), ({"modelo": "frr", "hiperparametros": {}}, "chaves"),
                         ({"modelo": "frr", "dataset": {"sessao": 2}}, "sessao")):
            with self.assertRaisesRegex(ValueError, msg):
                treina.roda("x", exp, dry_run=True)
        with self.assertRaisesRegex(ValueError, "nome"):
            treina.roda("a b", {"modelo": "frr"}, dry_run=True)


DADOS = os.path.exists(f"{paths.DATA}/wds/subj01/train/39.tar")


@unittest.skipUnless(DADOS, "sem os dados do subj01")
class Roda(unittest.TestCase):
    EXP = {"modelo": "mindeye2", "hiper": {"num_epochs": 5},
           "dataset": {"sessoes": 3, "n_imagens": 40, "repeticoes": 1, "semente": 1}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(treina.paths, "TRAIN_LOGS", Path(self.tmp.name))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def roda(self, exp, **kw):
        with redirect_stdout(io.StringIO()) as out, mock.patch("subprocess.run") as run, \
                mock.patch.object(treina, "_commit", return_value="abc"):
            treina.roda("exp", exp, **kw)
        return out.getvalue(), run

    def test_monta_dataset_registra_e_chama_o_script(self):
        _, run = self.roda(self.EXP)
        cmd, env = run.call_args_list[-1].args[0], run.call_args_list[-1].kwargs["env"]
        self.assertTrue(cmd[1].endswith("scripts/run_ridgeonly_prior.sh"))
        self.assertEqual((env["MODEL_NAME"], env["NUM_SESSIONS"], env["NUM_EPOCHS"]), ("exp", "3", "5"))
        man = dc.carrega(env["DATASET"])
        self.assertEqual(man["resumo"]["imagens_unicas"], 40)
        with open(Path(self.tmp.name) / "exp" / "experimento.json") as f:
            self.assertEqual(json.load(f)["hiper"]["num_epochs"], 5)

    def test_mesmo_nome_com_outra_configuracao_para(self):
        self.roda(self.EXP)
        self.roda(self.EXP)                                         # igual: segue
        with self.assertRaises(SystemExit):
            self.roda({**self.EXP, "dataset": {**self.EXP["dataset"], "semente": 2}})
        with self.assertRaises(SystemExit):
            self.roda({**self.EXP, "hiper": {"num_epochs": 6}})

    def test_dry_run_nao_grava_nem_treina(self):
        out, run = self.roda(self.EXP, dry_run=True)
        self.assertFalse((Path(self.tmp.name) / "exp").exists())
        self.assertIn("run_ridgeonly_prior.sh", out)
        self.assertFalse(any(c.args and c.args[0][0] == "bash" for c in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
