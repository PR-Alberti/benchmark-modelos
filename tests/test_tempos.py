"""Os tempos por etapa: a regra de 3 do src/etapas/mede_tempos.py e a aba Tempos do site."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from etapas import mede_tempos
from mindeye_ridge import paths
from report import site, tempos

MEDIDAS = {"etapas": {
    "reconstrucao": {p: {"s": s, "origem": "estimado"} for p, s in (
        ("mindeye2_1024", 8000.0), ("mindeye2_4096_blurry", 9000.0), ("frr", 7000.0),
        ("mindeye1", 3000.0), ("mindeye1_semlowlevel", 3100.0))},
    "refino": {p: {"s": 2800.0, "origem": "estimado"} for p in ("mindeye2_1024", "mindeye2_4096_blurry", "frr")},
    "retrieval": {p: {"s": s, "origem": "medido"} for p, s in (("mindeye2_1024", 200.0), ("mindeye2_4096_blurry", 300.0), ("frr", 50.0))},
    "metricas": {"bigg/unclip": {"s": 370.0, "origem": "medido"}, "bigg/refinada": {"s": 360.0, "origem": "medido"},
                 "mindeye1/unclip": {"s": 180.0, "origem": "medido"}}}}


class RegraDeTres(unittest.TestCase):
    def test_so_le_a_barra_do_total_certo(self):
        texto = "  3%|x| 1/30 [00:08<03:00,  8.0s/it]\r 50%|y| 19/38 [00:01<00:01, 20it/s]\r  7%|x| 2/30 [00:16<03:00]\n" \
                "100%|x| 30/30 [01:04:10<00:00]\n"
        self.assertEqual(mede_tempos.progresso(texto, 30), [(1, 8), (2, 16), (30, 3850)])

    def test_a_carga_conta_uma_vez_e_a_primeira_imagem_fica_fora_do_ritmo(self):
        # 20 s de carga, 30 s na 1a imagem e 8 s em cada uma das outras 29
        r = mede_tempos.estima(20 + 30 + 29 * 8, [(1, 30), (15, 142), (30, 262)])
        self.assertEqual(r["s_por_imagem"], 8.0)
        self.assertEqual(r["carga_s"], 20.0)
        self.assertEqual(r["s"], 20 + 30 + 999 * 8)
        self.assertEqual(r["origem"], "estimado")

    def test_sem_barra_nao_inventa_numero(self):
        with self.assertRaises(ValueError):
            mede_tempos.estima(100, [])

    def test_limpa_apaga_so_os_nomes_de_teste(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for pasta in ("tl/tempo_teste_a", "tl/subj01_me1_1sess", "ev/tempo_teste_b", "me1/models/tempo_teste_c"):
                (d / pasta).mkdir(parents=True)
            (d / "tb").mkdir()
            (d / "tb/tempo_teste_a_all_recons.csv").touch()
            (d / "tb/subj01_all_recons.csv").touch()
            with mock.patch.object(paths, "TRAIN_LOGS", d / "tl"), mock.patch.object(paths, "EVALS", d / "ev"), \
                    mock.patch.object(paths, "TABLES", d / "tb"), mock.patch.object(mede_tempos, "ME1_LOGS", d / "me1"):
                mede_tempos.limpa()
            self.assertEqual(sorted(p.name for p in (d / "tl").iterdir()), ["subj01_me1_1sess"])
            self.assertEqual(sorted(p.name for p in (d / "tb").iterdir()), ["subj01_all_recons.csv"])
            self.assertFalse((d / "me1/models/tempo_teste_c").exists())


class Aba(unittest.TestCase):
    def test_perfil(self):
        self.assertEqual(tempos.perfil({"id": "x_frr", "familia": "frr"}), "frr")
        self.assertEqual(tempos.perfil({"id": "subj01_me1_5sess", "familia": "mindeye1"}), "mindeye1")
        self.assertEqual(tempos.perfil({"id": "subj01_me1_5sess_semlowlevel", "familia": "mindeye1"}), "mindeye1_semlowlevel")
        self.assertEqual(tempos.perfil({"id": "a", "familia": "mindeye2", "config": {"hiper": {"hidden_dim": 4096}}}), "mindeye2_4096_blurry")
        self.assertEqual(tempos.perfil({"id": "a", "familia": "mindeye2", "config": {"hiper": {"hidden_dim": 1024}}}), "mindeye2_1024")

    def test_mindeye2_soma_as_tres_etapas(self):
        t = tempos.etapas({"id": "a", "familia": "mindeye2", "tempo_s": 1000.0}, MEDIDAS)
        self.assertEqual(t["avaliacao"], {"s": 930.0, "origem": "medido"})
        self.assertEqual(t["reconstrucao"]["origem"], "estimado")
        self.assertEqual(t["total_s"], 1000 + 8000 + 2800 + 930)

    def test_mindeye1_nao_tem_refino(self):
        t = tempos.etapas({"id": "subj01_me1_1sess", "familia": "mindeye1", "tempo_s": 100.0}, MEDIDAS)
        self.assertIsNone(t["refino"])
        self.assertEqual(t["avaliacao"]["s"], 180.0)
        self.assertEqual(t["total_s"], 100 + 3000 + 180)

    def test_modelo_do_artigo_fica_sem_treino_nem_total(self):
        t = tempos.etapas({"id": "final_subj01", "familia": "artigo", "tempo_s": None, "config": {"hiper": {"hidden_dim": 4096}}}, MEDIDAS)
        self.assertIsNone(t["treino"])
        self.assertIsNone(t["total_s"])
        self.assertEqual(t["reconstrucao"]["s"], 9000.0)

    def test_publicada_so_tem_as_tabelas(self):
        t = tempos.etapas({"id": "final__publicada", "familia": "artigo", "publicada": True}, MEDIDAS)
        self.assertIsNone(t["reconstrucao"])
        self.assertIsNone(t["refino"])
        self.assertEqual(t["avaliacao"]["s"], 730.0)

    def test_sem_medidas_mostra_so_o_treino(self):
        t = tempos.etapas({"id": "a", "familia": "frr", "tempo_s": 10.0, "frr": {"custo": {"segundos_pretreino": 5.0}}}, None)
        self.assertEqual(t["treino"]["s"], 10.0)
        self.assertEqual(t["pretreino_s"], 5.0)
        self.assertIsNone(t["reconstrucao"])
        self.assertIsNone(t["total_s"])

    def test_a_aba_existe_na_pagina_e_no_script(self):
        js = (Path(site.__file__).parent / "assets" / "site.js").read_text(encoding="utf-8")
        self.assertIn("tempos", site.ABAS)
        self.assertIn("['tempos', 'Tempos']", js)
        self.assertIn("tempos: renderTempos", js)


if __name__ == "__main__":
    unittest.main()
