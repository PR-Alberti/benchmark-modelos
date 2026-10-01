"""Integracao do FRR no benchmark: so retrieval, fora das tabelas de legenda e de cerebro.

    cd tests && python -m unittest test_benchmark -v

Usa dados sinteticos, sem depender dos resultados em disco.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from report import config, fmt, markdown, page, rows, texts


def frr_fake(fwd=0.5, bwd=0.05, sessoes=1):
    n_f = 20
    fracs = [round(0.05 * (i + 1), 2) for i in range(n_f)]
    hist = [900] + [10] * (n_f - 1)
    return {
        "modelo": "fake", "parametros_lineares": 15724 * 425984,
        "config": {"sessoes": sessoes, "exibicoes_treino": 688, "imagens_treino": 536, "voxels": 15724},
        "custo": {"segundos_ajuste": 12.0, "pico_gpu_gib": 3.9, "pico_ram_gib": 8.5},
        "cv": {"fracs": fracs, "cv_r2_por_fracao": [0.004 - 0.01 * i for i in range(n_f)],
               "cv_r2_escolhida": 0.007, "histograma_fracoes": hist, "segundos_cv": 9.0},
        "retrieval": {"fwd": {"media": fwd}, "bwd": {"media": bwd}},
        "retrieval_centrado": {"fwd": {"media": fwd + 0.01}, "bwd": {"media": 0.3}},
        "embedding": {"cosseno_medio": 0.5, "pearson_medio": 0.49, "cosseno_centrado_medio": 0.21},
    }


def dados_fake(frr1=True, frr40=False):
    vazio = {"enh": None, "base": None}
    tab = {"enh": {"FwdRetrieval": 0.928, "BwdRetrieval": 0.881, "PixCorr": 0.18, "Brain Corr. V1": 0.3},
           "base": {"PixCorr": 0.17}}
    modelos = []
    for m in config.MODELOS:
        d = dict(m, tabelas=dict(vazio), publicada=dict(vazio) if m.get("paper") else None,
                 legendas=None, params=None, tempo=None)
        if m.get("frr"):
            ligado = frr1 if m["sessoes"] == 1 else frr40
            d["frr_dados"] = frr_fake(sessoes=m["sessoes"]) if ligado else None
            d["tempo"] = fmt.duracao(12.0) if ligado else None
        elif m["id"] == "subj01_ridgeonly_1sess_prior":
            d["tabelas"] = tab
        modelos.append(d)
    return {"modelos": modelos,
            "frr_variantes": [{"id": i, "rotulo": r, "grupo": g, "dados": frr_fake(sessoes=g) if (frr1 if g == 1 else frr40) else None}
                              for i, r, g in config.FRR_VARIANTES]}


def texto(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


class FrrNaPagina(unittest.TestCase):
    def test_linha_do_frr_so_tem_retrieval(self):
        linhas = {l["rotulo"]: l for l in rows.linhas_de(dados_fake(), 1)}
        f = linhas["FRR · 1 sessão"]
        self.assertEqual(set(f["enh"]), {"FwdRetrieval", "BwdRetrieval"})
        self.assertEqual(f["enh"], f["base"])               # retrieval nao depende da reconstrucao

    def test_frr_sem_resultado_nao_vira_linha_de_modelo_comum(self):
        d = dados_fake(frr1=False)
        self.assertTrue(all(not l["enh"] for l in rows.linhas_de(d, 1) if l["rotulo"].startswith("FRR")))
        self.assertNotIn("FRR", texto(page.tabela_legendas(d)))

    def test_tabela_de_reconstrucao_mostra_o_frr_com_travessoes(self):
        h = page.tabela_metricas(dados_fake(), config.METRICAS, [("Baixo nível", 4), ("Alto nível", 4), ("Retrieval", 2)])
        linha = [t for t in h.split("<tr") if "FRR · 1 sessão" in t][0]
        self.assertIn("50,0%", linha)                       # Img→cér
        self.assertGreaterEqual(linha.count("—"), 8)        # sem as metricas de reconstrucao

    def test_frr_fica_fora_das_tabelas_de_cerebro_e_legendas(self):
        cer = page.tabela_metricas(dados_fake(), [(k, r, True, "dec") for k, r in config.CEREBRO], [("x", 6)])
        self.assertNotIn("FRR", cer)
        self.assertNotIn("FRR", page.tabela_legendas(dados_fake()))

    def test_frr_nao_entra_no_destaque_do_melhor_se_perde(self):
        h = page.tabela_metricas(dados_fake(), config.METRICAS, [("a", 8), ("b", 2)])
        linha = [t for t in h.split("<tr") if "FRR · 1 sessão" in t][0]
        self.assertNotIn("data-melhor", linha)

    def test_rotulos_do_retrieval_sao_os_do_codigo(self):
        rot = {k: r for k, r, *_ in config.METRICAS}
        self.assertEqual(rot["FwdRetrieval"], "Img→cér")     # fwd: dada a imagem, achar o cerebro
        self.assertEqual(rot["BwdRetrieval"], "Cér→img")

    def test_tabela_de_variantes_agrupa_por_sessoes(self):
        h = page.tabela_frr(dados_fake(frr1=True, frr40=True))
        for g in config.GRUPOS.values():
            self.assertIn(g, texto(h))
        self.assertIn("Grade estendida", texto(h))
        self.assertIn("(principal)", texto(h))

    def test_graficos_tem_histograma_e_curva_por_modelo_com_resultado(self):
        h = page.graficos_frr(dados_fake(frr1=True, frr40=False))
        self.assertEqual(h.count('class="barras"'), 1)
        self.assertEqual(h.count("<svg"), 1)
        self.assertEqual(h.count('class="barra"'), 20)
        self.assertEqual(page.graficos_frr(dados_fake(frr1=False)), "")

    def test_custo_usa_segundos_para_o_que_dura_pouco(self):
        self.assertEqual(fmt.duracao(12.0), "12 s")
        self.assertEqual(fmt.duracao(3600 * 7 + 60 * 51), "7 h 51")
        self.assertIn("12 s", texto(page.tabela_custo_frr(dados_fake())))

    def test_situacao_conta_o_frr_so_quando_ha_resultado(self):
        sem = page.situacao(dados_fake(frr1=False))[0]
        com = page.situacao(dados_fake(frr1=True))[0]
        self.assertEqual(com - sem, 1)

    def test_tabela_de_modelos_descreve_o_frr(self):
        h = texto(page.tabela_modelos(dados_fake()))
        self.assertIn("regressão linear", h)
        self.assertIn("6,70 bi", h)


class FrrNoMarkdown(unittest.TestCase):
    def md(self, **kw):
        d = dados_fake(**kw)
        d["ruido"] = {"treino": [("42", 0.9, 0.8)], "controle": [("com prior", 0.9, 0.8)],
                      "amostragem": {42: {"enh": None, "base": None}, 7: {"enh": None, "base": None}}}
        return markdown.markdown(d, None)

    def test_secao_do_frr_e_a_nota_do_retrieval(self):
        t = self.md()
        self.assertIn("## FRR: o baseline linear", t)
        self.assertIn("Img→cér", t)
        self.assertIn("o oposto do comentário", t)
        self.assertIn("hubness", t)

    def test_tabela_de_modelos_tem_a_linha_do_frr(self):
        t = self.md()
        self.assertRegex(t, r"\| FRR · 1 sessão \| 1 \| — \| não \| 6,70 bi de coeficientes")

    def test_frr_aparece_na_tabela_de_retrieval_e_nao_na_de_cerebro(self):
        t = self.md()
        refinadas = t.split("## Reconstrução e retrieval — refinadas")[1].split("## ")[0]
        self.assertIn("FRR · 1 sessão", refinadas)
        cerebro = t.split("## Correlação cerebral")[1].split("## ")[0]
        self.assertNotIn("| FRR", cerebro)

    def test_sem_resultado_do_frr_nao_quebra(self):
        t = self.md(frr1=False)
        self.assertIn("## FRR: o baseline linear", t)


class ConclusaoDoFrr(unittest.TestCase):
    def dados(self, **kw):
        d = dados_fake(**kw)
        for m in d["modelos"]:
            if m["id"] == "subj01_ridgeonly_40sess_prior":
                m["tabelas"] = {"enh": {"FwdRetrieval": 0.999, "BwdRetrieval": 0.995}, "base": None}
        return d

    def texto_todo(self, d):
        return " ".join(texts.leitura_frr(d))

    def sem_extendida(self, d):
        d["frr_variantes"] = [v for v in d["frr_variantes"] if not v["id"].endswith("_ext")]
        return d

    def test_usa_os_numeros_medidos(self):
        t = self.texto_todo(self.dados(frr1=True, frr40=True))
        self.assertIn("50,0%", t)        # FRR fwd (fake)
        self.assertIn("92,8%", t)        # ridge 1024 + prior, 1 sessao
        self.assertIn("99,9%", t)        # ridge, 40 sessoes
        self.assertIn("hubness", t)
        self.assertIn("0,21", t)         # cosseno centrado (fake)

    def test_tres_topicos_quando_ha_tudo(self):
        self.assertEqual(len(texts.leitura_frr(self.dados(frr1=True, frr40=True))), 3)

    def test_sem_a_variante_estendida_sao_dois_topicos_e_nada_sobre_a_grade(self):
        d = self.sem_extendida(self.dados(frr1=True, frr40=True))
        self.assertEqual(len(texts.leitura_frr(d)), 2)
        self.assertNotIn("estender a grade", self.texto_todo(d))

    def com_extendida(self, fwd_ext, r2_ext=0.03):
        d = self.sem_extendida(self.dados(frr1=True, frr40=True))
        ext = frr_fake(fwd=fwd_ext, sessoes=40)
        ext["cv"]["cv_r2_escolhida"] = r2_ext
        d["frr_variantes"].append({"id": "subj01_frr_40sess_ext", "rotulo": "ext", "grupo": 40, "dados": ext})
        return d

    def test_conta_o_efeito_da_grade_estendida_na_direcao_medida(self):
        pior = self.texto_todo(self.com_extendida(fwd_ext=0.4))
        self.assertIn("piora o Img→cér", pior)
        self.assertIn("não é o critério do retrieval", pior)
        melhor = self.texto_todo(self.com_extendida(fwd_ext=0.6))
        self.assertIn("melhora o Img→cér", melhor)
        self.assertNotIn("não é o critério do retrieval", melhor)

    def test_empate_nao_vira_melhora(self):
        self.assertIn("não muda o Img→cér", self.texto_todo(self.com_extendida(fwd_ext=0.5)))

    def test_fracao_unica_que_coincide_com_a_principal_e_dita(self):
        t = self.texto_todo(self.dados(frr1=True, frr40=True))      # no fake todas valem 50%
        self.assertIn("as duas coincidem", t)

    def test_sem_resultado_dos_dois_frr_nao_ha_conclusao(self):
        self.assertEqual(texts.leitura_frr(self.dados(frr1=True, frr40=False)), [])
        self.assertEqual(texts.leitura_frr(self.dados(frr1=False, frr40=False)), [])

    def test_leitura_acrescenta_os_topicos_so_quando_ha_numeros(self):
        self.assertEqual(len(texts.leitura(self.dados(frr1=True, frr40=True))), len(texts.LEITURA) + 3)
        self.assertEqual(len(texts.leitura(self.dados(frr1=False))), len(texts.LEITURA))


if __name__ == "__main__":
    unittest.main()
