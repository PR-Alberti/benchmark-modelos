"""O site (src/make_site.py): descoberta dos modelos, registros, pagina e simulador de dataset."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import treina                                                     # noqa: E402
from mindeye_ridge import agregacao, dataset_controlado as dc, paths   # noqa: E402
from report import apresentacao, catalogo, exemplos_site, galeria_site, semantica_site, site   # noqa: E402

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

    def test_fonte_pior_nao_substitui_a_melhor(self):
        """Numa maquina sem train_logs/ e sem results/evals/: a tabela nao passa por cima do verify_retrieval, e o
        dicionario nao passa por cima do experimento.json."""
        antigo = {"retrieval": {"bwd": {"media": .9, "ic95": [.89, .91]}, "fonte": "verify_retrieval"},
                  "valores": {"refinada": {"PixCorr": .3}}, "config": {"x": 1}, "rotulo": "do treina.py",
                  "config_fonte": "train_logs/m/experimento.json", "commit": "abc"}
        novo = {"retrieval": {"bwd": {"media": .8}, "fonte": "final_evaluations"}, "valores": {"refinada": {"PixCorr": .3}},
                "config": {"x": 2}, "rotulo": "m", "config_fonte": "src/experimentos.py", "commit": None}
        r = catalogo.mescla(antigo, novo)
        self.assertEqual(r["retrieval"]["fonte"], "verify_retrieval")
        self.assertEqual((r["config"], r["rotulo"], r["commit"]), ({"x": 1}, "do treina.py", "abc"))
        novo["retrieval"]["fonte"] = "verify_retrieval"                              # mesma fonte: o novo vale
        self.assertEqual(catalogo.mescla(antigo, novo)["retrieval"]["bwd"]["media"], .8)

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
        js = (ASSETS / "site.js").read_text(encoding="utf-8")
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
            antes_apr = (saida / "apresentacao.html").stat().st_mtime_ns
            make_site.main(["--saida", str(saida), "--quieto", "--sem-galeria"])
            self.assertEqual((saida / "index.html").stat().st_mtime_ns, antes)
            self.assertEqual((saida / "apresentacao.html").stat().st_mtime_ns, antes_apr)
            (self.d["tables"] / "outro_frr.json").write_text(json.dumps(FRR))   # um modelo novo
            make_site.main(["--saida", str(saida), "--quieto", "--sem-galeria"])
        self.assertIn("outro", (saida / "index.html").read_text(encoding="utf-8"))
        self.assertTrue((saida / "modelos" / "outro.json").exists())


def _ids_e_dados(html):
    ids = re.findall(r'\sid="([^"]+)"', html)
    bloco = re.search(r'<script type="application/json" id="dados">(.*?)</script>', html, re.S)[1]
    return ids, json.loads(bloco)


class Apresentacao(Arvore):
    def test_monta_com_qualquer_conjunto_de_modelos(self):
        modelos, planejados = self.atualiza()
        cabeca, corpo = apresentacao.monta(modelos, planejados, None, None, None)
        html = site.documento(cabeca, corpo)
        self.assertNotIn("{{", html)
        ids, dados = _ids_e_dados(html)
        self.assertEqual(len(ids), len(set(ids)), "id repetido")    # um slide id="dados" escondia o bloco de dados
        self.assertEqual(dados["retrieval"], [])                     # nenhum dos pares do benchmark nesta arvore
        self.assertGreaterEqual(html.count('class="slide'), 15)

    def test_chave_desconhecida_continua_sendo_erro(self):
        v = apresentacao._Valores({"x": "1"})
        self.assertEqual(site.preenche("{{x}} {{bwd:sumiu}} {{mi:sumiu:ridge}}", v, []), "1 — —")
        with self.assertRaises(KeyError):
            site.preenche("{{nao_existe}}", v, [])


class ApresentacaoDoRepositorio(unittest.TestCase):
    """A apresentacao com o que esta gravado em results/site: os registros, a galeria e os exemplos."""

    @classmethod
    def setUpClass(cls):
        raiz = Path(paths.SITE)
        cls.modelos = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((raiz / "modelos").glob("*.json"))]
        cls.galeria = galeria_site.atualiza(cls.modelos, raiz, gerar=False, log=lambda *a: None)
        cls.semantica = json.loads((raiz / "dados" / "semantica.json").read_text(encoding="utf-8"))
        cls.exemplos = exemplos_site.atualiza(raiz, cls.semantica, gerar=False, log=lambda *a: None)
        cabeca, corpo = apresentacao.monta(cls.modelos, [], cls.galeria, cls.semantica, cls.exemplos)
        cls.html = site.documento(cabeca, corpo)

    def test_numeros_preenchidos(self):
        self.assertNotIn("{{", self.html)
        ids, dados = _ids_e_dados(self.html)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(dados["retrieval"]), len(apresentacao.PARES))
        self.assertTrue(dados["gama"] and dados["hub"] and dados["custo"])
        for nome, de, para in apresentacao.PARES:            # o slide de retrieval tem as duas pontas
            linha = next(r for r in dados["retrieval"] if r["nome"] == nome)
            self.assertIsNotNone(linha["de"]["bwd"], de)
            self.assertIsNotNone(linha["para"]["bwd"], para)

    def test_reproducao(self):
        self.assertIn('id="reproducao"', self.html)
        for rotulo, _, diferenca in apresentacao.DIFERENCAS:       # docs/REEXECUTAR.md
            self.assertIn(diferenca.replace("&", "&amp;"), self.html, rotulo)
        self.assertNotIn("p,p", self.html)
        self.assertNotIn("contra o release", self.html)

    def test_toda_imagem_citada_existe(self):
        G, E = self.galeria, self.exemplos
        for spec in set(re.findall(r'data-tile="([^"]+)"', self.html)):
            p = spec.split(":")
            with self.subTest(spec=spec):
                if p[0] == "vista":
                    self.assertLess(int(p[1]), G["n"])
                else:
                    self.assertIn(p[2], G["modelos"].get(p[1], {}))
        for c in apresentacao.COLUNAS_GALERIA:
            self.assertIn(c[1], G["modelos"].get(c[0], {}), c[0])
        self.assertTrue(E and E["recorte"])
        for s in apresentacao.ORDEM_CLASSES:
            self.assertIn(s, E["classes"])
            for chave in ("fotos", "mascaras"):
                self.assertTrue((Path(paths.SITE) / E["classes"][s][chave]).exists())

    def test_exemplos_em_dia_com_o_semantica_json(self):
        from PIL import Image
        ind = json.loads((Path(paths.SITE) / "exemplos" / "indice.json").read_text(encoding="utf-8"))
        self.assertEqual(ind["params"], exemplos_site._params(self.semantica))
        fx, col = exemplos_site.faixas(), {s: j for j, s in enumerate(self.semantica["sup"])}
        for s, c in ind["classes"].items():
            for e in c["exemplos"]:
                lo, hi = fx[e["faixa"]]
                self.assertTrue(lo <= 100 * e["pix"][col[s]] / self.semantica["tela"] <= hi, (s, e["nsd"]))
            for arq in (c["fotos"], c["mascaras"]):
                w, h = Image.open(Path(paths.SITE) / "exemplos" / arq).size
                self.assertEqual((w, h), (len(c["exemplos"]) * ind["lado"], ind["lado"]), arq)


class Exemplos(unittest.TestCase):
    def test_faixas_cobrem_de_ponta_a_ponta(self):
        fx = exemplos_site.faixas()
        self.assertEqual((fx[0][0], fx[-1][1]), (exemplos_site.PISO, 100.0))
        for (_, b), (c, _) in zip(fx, fx[1:]):
            self.assertEqual(b, c)
        for p, (lo, hi) in zip(exemplos_site.PARADAS, fx):
            self.assertTrue(lo <= p < hi)

    def test_sorteio_fica_na_faixa_e_se_repete(self):
        import numpy as np
        rng = np.random.default_rng(1)
        fr = rng.uniform(0, 100, (600, 12)) * (rng.random((600, 12)) < 0.5)
        a, b = exemplos_site.escolhe(fr), exemplos_site.escolhe(fr)
        self.assertEqual(a, b)
        fx = exemplos_site.faixas()
        for j, s in enumerate(semantica_site.SUPERCATEGORIAS):
            contagem, linhas = a[s]
            for i, f in linhas:
                self.assertTrue(fx[f][0] <= fr[i, j] <= fx[f][1])
            for f in range(len(fx)):
                n = sum(1 for _, g in linhas if g == f)
                self.assertEqual(n, min(exemplos_site.N_POR_FAIXA, contagem[f]))

    def test_camada_da_mascara(self):
        import numpy as np
        m = np.zeros((425, 425), bool)
        m[100:300, 50:200] = True
        a = np.asarray(exemplos_site.camada_mascara(m, lado=224))
        self.assertEqual(a.shape, (224, 224, 4))
        self.assertEqual(a[105, 66, 3], 0)                 # a classe fica sem camada
        self.assertGreater(a[10, 10, 3], 100)              # o resto escurece
        self.assertTrue((a[45:53, 66, :3] == 255).all(axis=1).any())   # o contorno, logo acima da classe

    def test_decodifica_desfaz_codifica(self):
        import numpy as np
        import pandas as pd
        cols = [f"sup_{s}" for s in semantica_site.SUPERCATEGORIAS]
        fr = pd.DataFrame(np.round(np.random.default_rng(0).uniform(0, 100, (3, 12)), 4), index=[5, 9, 70],
                          columns=cols)
        d = semantica_site.codifica([5, 5, 9, 70, 70, 70], [0, 3, 1, 2, 4, 39], fr)
        sess, pix = semantica_site.decodifica(d)
        self.assertEqual(sess.tolist(), [[0, 3, 255], [1, 255, 255], [2, 4, 39]])
        self.assertTrue(np.allclose(100 * pix / d["tela"], fr.values, atol=100 / d["tela"]))


@unittest.skipUnless(NODE, "sem node")
class JavaScript(unittest.TestCase):
    def test_sintaxe(self):
        for arq in ("site.js", "simulador.js", "apresentacao.js"):
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
