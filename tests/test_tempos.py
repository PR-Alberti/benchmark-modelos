"""Os tempos por etapa: a regra de 3 do src/etapas/mede_tempos.py, a gravacao ao rodar e a aba Tempos do site."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from etapas import mede_tempos
from mindeye_ridge import paths
from report import catalogo, site, tempos

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


def _ini(e):
    return json.dumps({"etapa": e, "evento": "inicio", "em": 0})


def _fim(e, s, rc=0):
    return json.dumps({"etapa": e, "evento": "fim", "em": s, "s": s, "rc": rc})


class Registro(unittest.TestCase):
    """O tempos.jsonl: o que entra como o tempo de cada etapa."""

    def le(self, *linhas, familia=None):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tempos.jsonl"
            p.write_text("\n".join(linhas) + "\n", encoding="utf-8")
            return tempos.le_registro(p, familia)

    def test_sem_arquivo(self):
        self.assertIsNone(tempos.le_registro(Path("/nao/existe/tempos.jsonl")))

    def test_corrida_limpa(self):
        self.assertEqual(self.le(_ini("reconstrucao"), _fim("reconstrucao", 100), _ini("metricas_base"), _fim("metricas_base", 7)),
                         {"reconstrucao": 100, "metricas_base": 7})

    def test_reconstrucao_retomada_soma_as_tentativas(self):
        self.assertEqual(self.le(_ini("reconstrucao"), _fim("reconstrucao", 100, rc=1), _ini("reconstrucao"), _fim("reconstrucao", 50)),
                         {"reconstrucao": 150})

    def test_o_mindeye1_recomeca_do_zero_e_so_conta_a_que_terminou(self):
        self.assertEqual(self.le(_ini("reconstrucao"), _fim("reconstrucao", 100, rc=1), _ini("reconstrucao"), _fim("reconstrucao", 50),
                                 familia="mindeye1"), {"reconstrucao": 50})

    def test_metrica_que_falhou_nao_conta(self):
        self.assertEqual(self.le(_ini("metricas_base"), _fim("metricas_base", 100, rc=1), _ini("metricas_base"), _fim("metricas_base", 50)),
                         {"metricas_base": 50})

    def test_refazer_uma_etapa_substitui_a_medida(self):
        self.assertEqual(self.le(_ini("metricas_base"), _fim("metricas_base", 100), _ini("metricas_base"), _fim("metricas_base", 50)),
                         {"metricas_base": 50})
        self.assertEqual(self.le(_ini("refino"), _fim("refino", 100), _ini("refino"), _fim("refino", 50)), {"refino": 50})

    def test_reset_no_meio_de_etapa_retomada_nao_vira_tempo(self):
        # a primeira tentativa nao tem fim: o tempo dela se perdeu, e os 50 s nao sao o custo da etapa
        self.assertEqual(self.le(_ini("reconstrucao"), _ini("reconstrucao"), _fim("reconstrucao", 50)), {"reconstrucao": None})

    def test_reset_em_etapa_que_recomeca_nao_atrapalha(self):
        self.assertEqual(self.le(_ini("metricas_base"), _ini("metricas_base"), _fim("metricas_base", 50)), {"metricas_base": 50})

    def test_rodando_agora_nao_tem_tempo_e_nao_apaga_a_medida_anterior(self):
        self.assertEqual(self.le(_ini("refino")), {"refino": None})
        self.assertEqual(self.le(_ini("refino"), _fim("refino", 100), _ini("refino")), {"refino": 100})

    def test_linha_cortada_e_ignorada(self):
        self.assertEqual(self.le(_ini("refino"), _fim("refino", 100), '{"etapa": "ref'), {"refino": 100})


class NoCatalogo(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.raiz = Path(t.name)
        self.d = {"tables": self.raiz / "tb", "evals": self.raiz / "ev", "train_logs": self.raiz / "tl",
                  "me1": self.raiz / "me1", "logs": self.raiz / "lg", "metrics": self.raiz / "mt"}
        for p in self.d.values():
            p.mkdir()

    def grava(self, m, *linhas):
        (self.d["train_logs"] / m).mkdir(exist_ok=True)
        (self.d["train_logs"] / m / "tempos.jsonl").write_text("\n".join(linhas) + "\n", encoding="utf-8")

    def curva(self, m):
        (self.d["train_logs"] / m / "metrics.csv").write_text(
            "epoch,time/epoch_s,test/test_fwd_pct_correct,test/test_bwd_pct_correct\n0,10,0.1,0.1\n1,20,0.2,0.2\n", encoding="utf-8")

    def test_vai_para_o_registro_e_o_site_usa(self):
        self.grava("novo", _ini("reconstrucao"), _fim("reconstrucao", 1234))
        r = catalogo.extrai("novo", d=self.d, benchmark={}, experimentos={})
        self.assertEqual(r["tempos_medidos"], {"reconstrucao": 1234})
        t = tempos.etapas(r, MEDIDAS)
        self.assertEqual(t["reconstrucao"], {"s": 1234, "origem": "medido"})
        self.assertEqual(t["refino"]["origem"], "estimado")

    def test_so_o_tempo_nao_faz_de_uma_pasta_um_modelo(self):
        self.grava("so_tempo", _ini("metricas_base"), _fim("metricas_base", 1))
        self.assertNotIn("so_tempo", catalogo.ids_com_resultado(self.d))

    def test_mindeye1_treinado_fora_do_benchmark_tem_treino(self):
        (self.d["me1"] / "novo_me1").mkdir()
        self.grava("novo_me1", _ini("treino_lowlevel"), _fim("treino_lowlevel", 70))
        self.curva("novo_me1")
        r = catalogo.extrai("novo_me1", d=self.d, benchmark={}, experimentos={})
        self.assertEqual(r["familia"], "mindeye1")
        self.assertEqual(r["tempo_s"], 10 + 20 + 70)

    def test_mindeye1_com_lowlevel_sem_tempo_conhecido_fica_sem_treino(self):
        (self.d["me1"] / "novo_me1").mkdir()
        self.grava("novo_me1", _ini("treino_lowlevel"))
        self.curva("novo_me1")
        self.assertIsNone(catalogo.extrai("novo_me1", d=self.d, benchmark={}, experimentos={})["tempo_s"])

    def test_outra_maquina_nao_apaga_o_que_o_registro_tem(self):
        antigo = {"id": "a", "tempos_medidos": {"reconstrucao": 100, "refino": 50}}
        self.assertEqual(catalogo.mescla(antigo, {"id": "a", "tempos_medidos": None})["tempos_medidos"], antigo["tempos_medidos"])
        novo = catalogo.mescla(antigo, {"id": "a", "tempos_medidos": {"refino": 60, "reconstrucao": None, "metricas_base": 5}})
        self.assertEqual(novo["tempos_medidos"], {"reconstrucao": 100, "refino": 60, "metricas_base": 5})


class NaAba(unittest.TestCase):
    def test_a_propria_corrida_vale_mais_que_a_estimativa(self):
        m = {"id": "a", "familia": "mindeye2", "tempo_s": 1000.0,
             "tempos_medidos": {"reconstrucao": 11, "refino": 22, "retrieval": 33, "metricas_base": 44, "metricas_enhanced": 55}}
        t = tempos.etapas(m, MEDIDAS)
        self.assertEqual((t["reconstrucao"]["origem"], t["refino"]["origem"]), ("medido", "medido"))
        self.assertEqual(t["avaliacao"]["s"], 33 + 44 + 55)
        self.assertEqual(t["total_s"], 1000 + 11 + 22 + 132)

    def test_o_que_o_modelo_nao_gravou_vem_do_perfil(self):
        t = tempos.etapas({"id": "a", "familia": "mindeye2", "tempos_medidos": {"reconstrucao": None, "metricas_base": 44}}, MEDIDAS)
        self.assertEqual(t["reconstrucao"], {"s": 8000.0, "origem": "estimado"})
        self.assertEqual(t["avaliacao"]["s"], 44 + 360 + 200)

    def test_uma_medida_de_zero_segundo_e_medida(self):
        t = tempos.etapas({"id": "a", "familia": "mindeye2", "tempos_medidos": {"reconstrucao": 0}}, MEDIDAS)
        self.assertEqual(t["reconstrucao"], {"s": 0, "origem": "medido"})

    def test_frr_usa_o_retrieval_gravado_no_treino(self):
        m = {"id": "a", "familia": "frr", "frr": {"custo": {"segundos_retrieval": 9.0}}}
        self.assertEqual(tempos.etapas(m, MEDIDAS)["avaliacao"]["s"], 9 + 370 + 360)

    def test_mindeye1_usa_as_proprias_metricas(self):
        m = {"id": "subj01_me1_5sess", "familia": "mindeye1", "tempos_medidos": {"metricas_base": 44, "reconstrucao": 11}}
        t = tempos.etapas(m, MEDIDAS)
        self.assertEqual(t["avaliacao"]["s"], 44)
        self.assertEqual(t["reconstrucao"]["s"], 11)

    def test_chega_na_pagina(self):
        # dados_pagina escolhe os campos do registro: um campo fora da lista some sem erro
        registro = {"id": "a", "rotulo": "a", "familia": "mindeye2", "grupo": "experimento", "tempo_s": 5.0,
                    "tempos_medidos": {"reconstrucao": 11}}
        with mock.patch.object(tempos, "carrega", return_value=MEDIDAS):
            m = site.dados_pagina([registro], [], {}, {})["modelos"][0]
        self.assertEqual(m["tempos"]["reconstrucao"], {"s": 11, "origem": "medido"})

    def test_publicada_ignora_o_registro(self):
        t = tempos.etapas({"id": "x__publicada", "familia": "artigo", "publicada": True, "tempos_medidos": {"reconstrucao": 11}}, MEDIDAS)
        self.assertIsNone(t["reconstrucao"])


class NosScripts(unittest.TestCase):
    """Os scripts de verdade numa copia do repositorio, com um python falso: nenhuma etapa roda de fato."""

    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.raiz = Path(t.name) / "repo"
        for pasta in ("src", "mindeye1/src", "results/evals/m", "results/tables", "train_logs"):
            (self.raiz / pasta).mkdir(parents=True)
        shutil.copytree(Path(paths.REPO) / "scripts", self.raiz / "scripts")
        env = self.raiz / "env/bin"
        env.mkdir(parents=True)
        (env / "python").write_text('#!/bin/bash\necho falso\nexit "${FALSO_RC:-0}"\n', encoding="utf-8")
        (env / "python").chmod(0o755)
        self.dados = self.raiz / "dados"
        (self.dados / ".cache/hub/models--shi-labs--versatile-diffusion/snapshots/x").mkdir(parents=True)

    def roda(self, *cmd, ok=True, **env):
        r = subprocess.run(["bash", str(self.raiz / "scripts" / cmd[0]), *cmd[1:]], capture_output=True, text=True,
                           env={**os.environ, "MINDEYE_ENV": str(self.raiz / "env"), "MINDEYE_DATA": str(self.dados), **env})
        self.assertEqual(r.returncode == 0, ok, r.stdout + r.stderr)
        return r

    def medido(self, m, familia=None):
        return tempos.le_registro(self.raiz / "train_logs" / m / "tempos.jsonl", familia)

    def test_mindeye2_reconstrucao_e_refino(self):
        self.roda("run_recon.sh", "m", "1024", "noblurry")
        self.assertEqual(set(self.medido("m")), {"reconstrucao", "refino"})

    def test_o_que_ja_estava_feito_nao_grava_nada(self):
        (self.raiz / "results/evals/m/m_all_enhancedrecons.pt").touch()
        self.roda("run_recon.sh", "m", "1024", "noblurry")
        self.assertEqual(set(self.medido("m")), {"reconstrucao"})

    def test_falha_passa_adiante_e_fica_registrada(self):
        self.roda("run_recon.sh", "m", "1024", "noblurry", ok=False, FALSO_RC="3")
        linhas = [json.loads(x) for x in (self.raiz / "train_logs/m/tempos.jsonl").read_text().splitlines()]
        self.assertEqual([x["evento"] for x in linhas], ["inicio", "fim"])
        self.assertEqual(linhas[1]["rc"], 3)
        self.assertIsNone(self.medido("m"))

    def test_frr_reconstrucao_refino_e_metricas(self):
        ev = self.raiz / "results/evals/m/m"
        for suf in ("all_clipvoxels", "all_recons", "all_enhancedrecons"):
            Path(f"{ev}_{suf}.pt").touch()
        (self.raiz / "results/evals/m/m_all_recons.pt").unlink()
        (self.raiz / "results/evals/m/m_all_enhancedrecons.pt").unlink()
        falso = self.raiz / "env/bin/python"       # desta vez as etapas gravam a saida, como as de verdade
        falso.write_text('#!/bin/bash\nfor a in "$@"; do case "$a" in\n'
                         f'  etapas/recon_embeddings.py) touch {ev}_all_recons.pt ;;\n'
                         f'  etapas/enhanced_recon_inference.py) touch {ev}_all_enhancedrecons.pt ;;\n'
                         'esac; done\n', encoding="utf-8")
        self.roda("run_recon_embeddings.sh", "m", "refino")
        self.assertEqual(set(self.medido("m")), {"reconstrucao", "refino", "metricas_base", "metricas_enhanced"})

    def test_metricas(self):
        (self.raiz / "results/evals/m/m_all_recons.pt").touch()
        self.roda("run_evals.sh", "m", "base")
        self.assertEqual(set(self.medido("m")), {"metricas_base"})

    def test_mindeye1_lowlevel_e_reconstrucao(self):
        self.roda("me1_run.sh", "lowlevel", NUM_SESSIONS="5", MODEL_NAME="m1", AE_NAME="ae")
        self.roda("me1_run.sh", "recon", NUM_SESSIONS="5", MODEL_NAME="m1", AE_NAME="ae")
        self.assertEqual(set(self.medido("m1", "mindeye1")), {"treino_lowlevel", "reconstrucao"})

    def test_mindeye1_reconstrucao_conta_para_quem_recebe_as_imagens(self):
        # run_me1_sessoes.sh: o _semlowlevel sai do checkpoint do outro; nao pode sobrescrever o tempo dele
        self.roda("me1_run.sh", "recon", "--autoencoder_name=None", "--img2img_strength=1", "--out_name=m1_semlowlevel",
                  NUM_SESSIONS="5", MODEL_NAME="m1")
        self.assertEqual(set(self.medido("m1_semlowlevel", "mindeye1")), {"reconstrucao"})
        self.assertIsNone(self.medido("m1", "mindeye1"))

    def test_todo_retrieval_do_mindeye2_passa_pelo_cronometro(self):
        for script in ("run_benchmark.sh", "run_reexecutar.sh", "run_sessoes_intermediarias.sh"):
            texto = (self.raiz / "scripts" / script).read_text(encoding="utf-8")
            linha = next(x for x in texto.splitlines() if "etapas/verify_retrieval.py --model_name" in x)
            self.assertIn('cronometra "$m" retrieval', linha, script)


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
