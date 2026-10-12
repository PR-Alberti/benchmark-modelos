"""O site (src/make_site.py): descoberta dos modelos, registros, pagina e simulador de dataset."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import treina                                                     # noqa: E402
from mindeye_ridge import agregacao, dataset_controlado as dc, paths   # noqa: E402
from report import catalogo, site                                 # noqa: E402

NODE = shutil.which("node")
ASSETS = Path(paths.SRC) / "report" / "assets"
DADOS = (paths.DATA / "wds/subj01/train/39.tar").exists() and \
    (paths.DATA / "semantica/subj01_fracao_tela.csv").exists()

FRR = {"retrieval": {"fwd": {"media": 0.5, "ic95": [0.49, 0.51]}, "bwd": {"media": 0.1, "ic95": [0.09, 0.11]}},
       "retrieval_centrado": {"fwd": {"media": 0.55}, "bwd": {"media": 0.3}},
       "config": {"sessoes": 1, "exibicoes_treino": 688, "imagens_treino": 536}, "parametros_lineares": 10,
       "custo": {"segundos_ajuste": 12.0}, "hubness": {},
       "cv": {"fracs": [0.05, 0.1], "cv_r2_por_fracao": [0.01, -0.01], "cv_r2_escolhida": 0.01, "histograma_fracoes": [9, 1]},
       "embedding": {"cosseno_medio": 0.5, "pearson_medio": 0.5, "cosseno_centrado_medio": 0.1}}
TABELA = {"PixCorr": 0.2, "SSIM": 0.3, "AlexNet(2)": 0.8, "FwdRetrieval": 0.9, "BwdRetrieval": 0.85}
CABECALHO_CURVA = "epoch,time/epoch_s,test/test_fwd_pct_correct,test/test_bwd_pct_correct,train/loss"


def tabela(p, valores):
    p.write_text("Metric\tValue\n" + "".join(f"{k}\t{v}\n" for k, v in valores.items()))


def curva(p, n, parcial=False):
    linhas = [CABECALHO_CURVA] + [f"{e},10,{0.1 * (e + 1)},{0.05 * (e + 1)},{5 - e}" for e in range(n)]
    if parcial:
        linhas.append(f"{n},10,0.9")                   # o treino no meio da gravacao
    p.write_text("\n".join(linhas) + "\n")


class Arvore(unittest.TestCase):
    """Uma arvore de resultados falsa, no formato dos scripts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = Path(self.tmp.name)
        self.d = {"tables": r / "tables", "evals": r / "evals", "train_logs": r / "train_logs",
                  "me1": r / "mindeye1/train_logs", "logs": r / "logs", "metrics": r / "metrics"}
        for p in self.d.values():
            p.mkdir(parents=True)
        self.registros = r / "site" / "modelos"
        t, tl, ev = self.d["tables"], self.d["train_logs"], self.d["evals"]
        # FRR treinado pelo treina.py, com rotulo
        (t / "novo_frr_frr.json").write_text(json.dumps(FRR))
        (tl / "novo_frr").mkdir()
        (tl / "novo_frr" / "experimento.json").write_text(json.dumps(
            {"nome": "novo_frr", "modelo": "frr", "rotulo": "FRR de teste", "hiper": {}, "dataset": {"sessoes": 1},
             "resumo_dataset": {"exibicoes": 688, "imagens_unicas": 536}, "commit": "abc"}))
        # MindEye2 avaliado, com curva
        tabela(t / "novo_me2_all_recons.csv", TABELA)
        tabela(t / "novo_me2_all_enhancedrecons.csv", {**TABELA, "PixCorr": 0.25})
        (tl / "novo_me2").mkdir()
        curva(tl / "novo_me2" / "metrics.csv", 4)
        (tl / "novo_me2" / "experimento.json").write_text(json.dumps(
            {"modelo": "mindeye2", "hiper": {"num_epochs": 4}, "dataset": {"sessoes": 40, "n_imagens": 100}}))
        # so o retrieval, sem configuracao registrada
        (ev / "so_ret").mkdir()
        (ev / "so_ret" / "so_ret_retrieval.json").write_text(json.dumps(
            {"fwd": {"media": 0.7, "ic95": [0.69, 0.71]}, "bwd": {"media": 0.6, "ic95": [0.59, 0.61]}}))
        # em treino: curva com a ultima linha pela metade
        (tl / "em_treino").mkdir()
        curva(tl / "em_treino" / "metrics.csv", 3, parcial=True)
        (tl / "em_treino" / "experimento.json").write_text(json.dumps(
            {"modelo": "mindeye2", "hiper": {}, "dataset": {"sessoes": 1}}))

    def tearDown(self):
        self.tmp.cleanup()

    def atualiza(self):
        return catalogo.atualiza(self.registros, d=self.d)


class Catalogo(Arvore):
    def test_descobre_cada_tipo_de_resultado(self):
        modelos, planejados = self.atualiza()
        m = {x["id"]: x for x in modelos}
        self.assertEqual(set(m), {"novo_frr", "novo_me2", "so_ret", "em_treino"})
        self.assertEqual((m["novo_frr"]["familia"], m["novo_frr"]["status"], m["novo_frr"]["rotulo"]),
                         ("frr", "completo", "FRR de teste"))
        self.assertEqual(m["novo_frr"]["retrieval"]["fonte"], "frr")
        self.assertEqual(m["novo_frr"]["config"]["hiper"], treina.MODELOS["frr"].padrao)   # completado
        self.assertEqual(m["novo_me2"]["status"], "completo")
        self.assertEqual(m["novo_me2"]["valores"]["refinada"]["PixCorr"], 0.25)
        self.assertEqual(m["novo_me2"]["valores"]["unclip"]["PixCorr"], 0.2)
        self.assertEqual(m["novo_me2"]["curva"]["epocas"], [1, 2, 3, 4])
        self.assertIn("100 img", m["novo_me2"]["rotulo"])                                 # rotulo montado
        self.assertEqual((m["so_ret"]["status"], m["so_ret"]["config"], m["so_ret"]["grupo"]),
                         ("retrieval", None, "experimento"))
        self.assertEqual(m["em_treino"]["status_txt"], "em treino · época 3")             # linha parcial fora
        self.assertEqual(m["em_treino"]["retrieval"]["fonte"], "treino")
        planejados = {p["id"] for p in planejados}
        self.assertIn("subj01_frr_1sess", planejados)        # do benchmark, sem resultado nesta arvore
        self.assertFalse(planejados & set(m))

    def test_registro_guarda_o_que_sumiu_desta_maquina(self):
        self.atualiza()
        (self.d["train_logs"] / "novo_me2" / "metrics.csv").unlink()            # curva so na outra maquina
        (self.d["train_logs"] / "so_ret").mkdir()
        curva(self.d["train_logs"] / "so_ret" / "metrics.csv", 2)
        (self.d["evals"] / "so_ret" / "so_ret_retrieval.json").unlink()        # avaliacao so na outra maquina
        m = {x["id"]: x for x in self.atualiza()[0]}
        self.assertEqual(m["novo_me2"]["curva"]["epocas"], [1, 2, 3, 4])
        self.assertEqual(m["so_ret"]["retrieval"]["fonte"], "verify_retrieval")  # a curva nao substitui
        self.assertEqual(m["so_ret"]["valores"]["refinada"]["FwdRetrieval"], 0.7)

    def test_registro_apagado_tira_o_modelo(self):
        self.atualiza()
        shutil.rmtree(self.d["evals"] / "so_ret")
        (self.registros / "so_ret.json").unlink()
        self.assertNotIn("so_ret", {x["id"] for x in self.atualiza()[0]})

    def test_rotulo_automatico(self):
        def rot(cfg):
            return catalogo.rotulo_automatico(catalogo.config_efetiva(cfg)[0])
        self.assertEqual(rot({"modelo": "frr", "dataset": {"sessoes": 1}}), "FRR · 1 sessão")
        self.assertEqual(rot({"modelo": "mindeye2", "hiper": {"prior": False, "seed": 7}, "dataset": {}}),
                         "Ridge 1024 sem prior · 40 sessões · semente 7")
        self.assertEqual(rot({"modelo": "mindeye1", "dataset": {"n_imagens": 1000, "repeticoes": 3, "agregacao": "media"}}),
                         "MindEye1 · 1.000 img × 3 · média das repetições · 40 sessões")
        self.assertEqual(rot({"modelo": "frr", "dataset": {"classes": {"animal": ["sup_animal"]}, "n_imagens": 300,
                                                             "balanceado": True}}),
                         "FRR · animal · 300 img/classe · 40 sessões")


class Descricoes(unittest.TestCase):
    """As tabelas de Como usar saem do codigo; toda chave nova precisa de uma descricao."""

    def test_todo_hiperparametro_tem_descricao(self):
        for nome, modelo in treina.MODELOS.items():
            self.assertEqual(set(site.HIPER_DESC[nome]), set(modelo.padrao), nome)

    def test_todo_campo_do_dataset_tem_descricao(self):
        campos = set(dc.ConfigDataset().como_dict())
        self.assertEqual(set(site.DATASET_DESC), campos)
        self.assertEqual(set(site.AGREGACAO_DESC), {None, *agregacao.MODOS})

    def test_nome_da_metrica_liga_a_definicao_aos_resultados(self):
        slugs = [m["slug"] for m in site.METRICAS]
        self.assertEqual(len(set(slugs)), len(slugs))                # um endereco por metrica
        self.assertTrue(all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", x) for x in slugs), slugs)
        tabela = site.html_metricas()
        js = (ASSETS / "site.js").read_text()
        for x in slugs:
            self.assertIn(f'<tr id="def-{x}">', tabela)              # destino dos nomes nos resultados
            self.assertIn(f'href="#res-{x}"', tabela)                # e a ida para os resultados
        self.assertIn("'#def-' + mt.slug", js)
        self.assertIn("id: 'res-' + mt.slug", js)

    def test_presets_e_pares_apontam_para_modelos_do_benchmark(self):
        conhecidos = set(catalogo.ROTULOS)
        for _, _, _, ids in site.PRESETS:
            self.assertLessEqual(set(ids), conhecidos)
        for _, a, b in site.PARES:
            self.assertLessEqual({a, b}, conhecidos)


class Pagina(Arvore):
    def test_pagina_tem_os_modelos_e_nenhum_marcador_sobrando(self):
        modelos, planejados = self.atualiza()
        cabeca, corpo = site.monta(modelos, planejados, None, None)
        html = site.documento(cabeca, corpo)
        self.assertNotIn("{{", html)
        self.assertIn("<title>benchmark-modelos</title>", cabeca)
        bloco = re.search(r'<script type="application/json" id="dados">(.*?)</script>', html, re.S)[1]
        dados = json.loads(bloco)
        self.assertEqual({m["id"] for m in dados["modelos"]}, {m["id"] for m in modelos})
        for pag in ("inicio", "uso", "resultados"):
            self.assertIn(f'data-pag="{pag}"', html)
        self.assertNotIn("<html", corpo)                       # o corpo serve para publicar sem o esqueleto

    def test_make_site_nao_reescreve_sem_mudanca(self):
        import make_site
        saida = Path(self.tmp.name) / "site"
        alvos = {"TABLES": self.d["tables"], "EVALS": self.d["evals"], "TRAIN_LOGS": self.d["train_logs"],
                 "LOGS": self.d["logs"], "METRICS": self.d["metrics"], "DATA": Path(self.tmp.name) / "sem_dados",
                 "REPO": Path(self.tmp.name)}
        with mock.patch.multiple(paths, **alvos):
            make_site.main(["--saida", str(saida), "--quieto", "--sem-galeria"])
            antes = (saida / "index.html").stat().st_mtime_ns
            make_site.main(["--saida", str(saida), "--quieto", "--sem-galeria"])
            self.assertEqual((saida / "index.html").stat().st_mtime_ns, antes)
            (self.d["tables"] / "outro_frr.json").write_text(json.dumps(FRR))   # um modelo novo
            make_site.main(["--saida", str(saida), "--quieto", "--sem-galeria"])
        self.assertIn("outro", (saida / "index.html").read_text())
        self.assertTrue((saida / "modelos" / "outro.json").exists())


@unittest.skipUnless(NODE, "sem node")
class JavaScript(unittest.TestCase):
    def test_sintaxe(self):
        for arq in ("site.js", "simulador.js"):
            r = subprocess.run([NODE, "--check", str(ASSETS / arq)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    @unittest.skipUnless(DADOS, "sem os dados do subj01")
    def test_simulador_conta_como_o_dataset_controlado(self):
        from mindeye_ridge import semantica
        from report import semantica_site
        with tempfile.TemporaryDirectory() as tmp:
            d = semantica_site.atualiza(Path(tmp))
            cfgs = [{"sessoes": 1}, {"sessoes": 20, "repeticoes": 2},
                    {"classes": {"pessoa": ["sup_person"], "animal": ["sup_animal"], "comida": ["sup_food"]}},
                    {"regra": "prioridade", "classes": ["animal", "outro"], "sessoes": 7, "repeticoes": 1},
                    {"classes": {"pessoa": ["sup_person"], "animal": ["sup_animal"]}, "balanceado": True,
                     "n_imagens": 300, "repeticoes": 3}]
            Path(tmp, "cfgs.json").write_text(json.dumps(cfgs))
            js = subprocess.run([NODE, "-e", (
                f"const s = require({json.dumps(str(ASSETS / 'simulador.js'))});"
                f"const d = s.decodifica(JSON.parse(require('fs').readFileSync({json.dumps(tmp + '/dados/semantica.json')})));"
                f"const c = JSON.parse(require('fs').readFileSync({json.dumps(tmp + '/cfgs.json')}));"
                "console.log(JSON.stringify(c.map(x => s.simula(d, x))));")], capture_output=True, text=True)
            self.assertEqual(js.returncode, 0, js.stderr)
            fr = semantica.tabela_fracoes(paths.DATA, 1)
            for cfg, b in zip(cfgs, json.loads(js.stdout)):
                a = dc.monta(dc.ConfigDataset.de_dict(cfg), paths.DATA, fracoes=fr)["resumo"]
                with self.subTest(cfg=cfg):
                    self.assertEqual((a["imagens_unicas"], a["exibicoes"]), (b["imagens_unicas"], b["exibicoes"]))
                    self.assertEqual({int(k): v for k, v in a["repeticoes"].items()},
                                     {int(k): v for k, v in b["repeticoes"].items()})
                    self.assertEqual(a.get("imagens_por_classe") or {},
                                     {k: v for k, v in (b["imagens_por_classe"] or {}).items() if v})


if __name__ == "__main__":
    unittest.main()
