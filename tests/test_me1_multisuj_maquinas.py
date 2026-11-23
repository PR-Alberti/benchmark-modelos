"""Protocolos multi-sujeito do MindEye1 (docs/PROCEDIMENTO_ME1_MULTISUJ.md): a conta de sujeitos, sessoes,
batch e passos por maquina, e o script de cada maquina. Roda sem GPU e sem dados."""
import ast
import shutil
import subprocess
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "mindeye1" / "src" / "Train_MindEye_multisubject.py"
SCRIPT = REPO / "mindeye1" / "src" / "run_me1_ms_maquina.sh"


def _carrega_plan():
    """So plan_subjects e NSESSIONS_ALLSUBJ, sem importar torch/webdataset."""
    arvore = ast.parse(TRAIN.read_text(encoding="utf-8"))
    nos = [n for n in arvore.body
           if (isinstance(n, ast.FunctionDef) and n.name == "plan_subjects")
           or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "NSESSIONS_ALLSUBJ" for t in n.targets))]
    ns = {"np": np}
    exec(compile(ast.Module(body=nos, type_ignores=[]), str(TRAIN), "exec"), ns)
    return ns["plan_subjects"]


def _args(**kw):
    base = dict(subj=1, multi_subject=False, pretrain_subjs=None, pretrain_sessions=None, num_sessions=40, batch_size=24)
    base.update(kw)
    return types.SimpleNamespace(**base)


# maquina: (sujeitos, sessoes, batch global, batch por sujeito, passos por epoca)
MAQUINAS = {
    1: ([2, 5], 10, 24, 12, 750 * 10 // 24),
    2: ([2, 5], 20, 24, 12, 750 * 20 // 24),
    3: ([2, 5, 7, 3], 5, 24, 6, 750 * 5 // 24),
    4: ([2, 5, 7, 3], 10, 24, 6, 750 * 10 // 24),
    5: ([2, 3, 4, 5, 6, 7, 8], 1, 21, 3, 750 * 1 // 21),
    6: ([2, 3, 4, 5, 6, 7, 8], 5, 21, 3, 750 * 5 // 21),
}


class TestPlano(unittest.TestCase):
    plan = staticmethod(_carrega_plan())

    def test_padrao_igual_ao_mindeye2(self):
        subj, nsess, bs, it = self.plan(_args(multi_subject=True, batch_size=21))
        self.assertEqual(subj, [2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(nsess, {2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30})
        self.assertEqual((bs, it), (3, 750 * 40 // 21))

    def test_maquinas(self):
        for m, (subjs, n, bglob, bsub, iters) in MAQUINAS.items():
            with self.subTest(maquina=m):
                subj, nsess, bs, it = self.plan(_args(multi_subject=True, pretrain_subjs=subjs,
                                                      pretrain_sessions=n, batch_size=bglob))
                self.assertEqual(subj, subjs)
                self.assertEqual(set(nsess.values()), {n})
                self.assertEqual((bs, it), (bsub, iters))

    def test_ajuste_no_subj01(self):
        for m in (1, 5, 10, 20, 40):
            subj, nsess, bs, it = self.plan(_args(num_sessions=m))
            self.assertEqual((subj, nsess, bs, it), ([1], {1: m}, 24, 750 * m // 24))

    def test_recusas(self):
        with self.assertRaises(ValueError):   # sujeito de teste no pre-treino
            self.plan(_args(multi_subject=True, pretrain_subjs=[1, 2]))
        with self.assertRaises(ValueError):   # subj04 tem 30 sessoes
            self.plan(_args(multi_subject=True, pretrain_subjs=[4], pretrain_sessions=40))
        with self.assertRaises(ValueError):   # opcao de pre-treino no ajuste
            self.plan(_args(pretrain_sessions=5))


@unittest.skipUnless(shutil.which("bash"), "sem bash")
class TestScript(unittest.TestCase):
    def roda(self, *a):
        return subprocess.run(["bash", str(SCRIPT), *a], capture_output=True, text=True,
                              env={"PATH": "/usr/bin:/bin", "HOME": "/tmp/me1ms_teste", "OUT": "/tmp/me1ms_teste"})

    def test_lista(self):
        r = self.roda("--lista")
        self.assertEqual(r.returncode, 0, r.stderr)
        for nome in ("2subj_10sess", "2subj_20sess", "4subj_5sess", "4subj_10sess", "7subj_1sess", "7subj_5sess"):
            self.assertIn(nome, r.stdout)

    def test_maquina_invalida(self):
        self.assertNotEqual(self.roda("7").returncode, 0)


if __name__ == "__main__":
    unittest.main()
