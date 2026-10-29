"""O site do benchmark (results/site/index.html): Inicio, Como usar e Resultados numa pagina so.

A pagina abre direto do disco, sem servidor, e troca de parte pelo endereco (#inicio, #uso,
#resultados, #tabela...). Os textos de Inicio e Como usar estao em conteudo/*.html; as tabelas que
dependem do codigo (hiperparametros, campos do dataset, agregacao) saem do proprio codigo, para nao
ficarem desatualizadas. Os resultados vem dos registros do catalogo (catalogo.py), a galeria das
tiras (galeria_site.py) e o simulador de dataset dos dados de semantica_site.py.
"""
import datetime
import html
import json
import pathlib
import re

from . import texts
from .catalogo import PUBLICADA
from .config import REPO_URL
from .fmt import commit, duracao

AQUI = pathlib.Path(__file__).resolve().parent

# (id, rotulo curto, nome, grupo, maior e melhor, formato, acaso, descricao)
_M = [
    ("FwdRetrieval", "Img→cér", "Retrieval imagem → cérebro", "retrieval", True, "pct", 1 / 300,
     "dada cada imagem, achar o seu cérebro entre 300 previsões (fwd no código)"),
    ("BwdRetrieval", "Cér→img", "Retrieval cérebro → imagem", "retrieval", True, "pct", 1 / 300,
     "dada cada previsão do cérebro, achar a sua imagem entre 300 (bwd no código; o Image Retrieval do artigo)"),
    ("PixCorr", "PixCorr", "PixCorr", "baixo", True, "dec", None, "correlação de pixels com a imagem vista"),
    ("SSIM", "SSIM", "SSIM", "baixo", True, "dec", None, "semelhança estrutural com a imagem vista"),
    ("AlexNet(2)", "Alex(2)", "AlexNet, camada 2", "baixo", True, "pct", 0.5,
     "identificação 2-way nas features da camada 2 da AlexNet: com que frequência a reconstrução fica mais "
     "perto da sua imagem do que de outra"),
    ("AlexNet(5)", "Alex(5)", "AlexNet, camada 5", "baixo", True, "pct", 0.5,
     "identificação 2-way nas features da camada 5 da AlexNet"),
    ("InceptionV3", "Incep", "InceptionV3", "alto", True, "pct", 0.5, "identificação 2-way nas features da InceptionV3"),
    ("CLIP", "CLIP", "CLIP", "alto", True, "pct", 0.5, "identificação 2-way no espaço do CLIP ViT-L/14"),
    ("EffNet-B", "Eff", "EfficientNet-B1", "alto", False, "dec", None,
     "distância de correlação nas features da EfficientNet-B1"),
    ("SwAV", "SwAV", "SwAV", "alto", False, "dec", None, "distância de correlação nas features do SwAV ResNet50"),
]
_REGIOES = [("nsd_general", "nsdgeneral", "todo o córtex visual usado pelos modelos"), ("V1", "V1", "V1"),
            ("V2", "V2", "V2"), ("V3", "V3", "V3"), ("V4", "V4", "V4"),
            ("higher_vis", "Alto nível", "áreas visuais de alto nível")]
for chave, rot, onde in _REGIOES:
    _M.append((f"Brain Corr. {chave}", rot, f"Correlação cerebral · {rot}", "cerebro", True, "dec", None,
               f"correlação entre a resposta que o GNet prevê para a reconstrução e a resposta medida à imagem "
               f"vista, em {onde}"))
_LEG = [("Meteor", "METEOR", "METEOR"), ("Rouge1", "ROUGE-1", "ROUGE-1"), ("RougeL", "ROUGE-L", "ROUGE-L"),
        ("Sentence", "Sentence", "Sentence-Transformers (similaridade de frases)"),
        ("CLIP-B", "CLIP-B", "similaridade no espaço de texto do CLIP ViT-B/32"),
        ("CLIP-L", "CLIP-L", "similaridade no espaço de texto do CLIP ViT-L/14")]
for chave, rot, nome in _LEG:
    _M.append((f"leg:{chave}", rot, f"Legenda · {nome}", "legendas", True, "dec", None,
               "a legenda que o GIT gera do embedding previsto do cérebro, contra as legendas humanas do COCO"))
METRICAS = [dict(zip(("id", "rotulo", "nome", "grupo", "maior", "fmt", "acaso", "desc"), m)) for m in _M]
# O nome de cada metrica liga a definicao (Como usar, #def-<slug>) aos resultados dela (Graficos,
# #res-<slug>) e vice-versa; o slug vai nos dados para o site.js montar os mesmos enderecos.
for _m in METRICAS:
    _m["slug"] = re.sub(r"[^a-z0-9]+", "-", _m["id"].lower()).strip("-")
IDS_METRICAS = {m["id"] for m in METRICAS}

GRUPOS_METRICAS = [
    {"id": "retrieval", "nome": "Retrieval", "curto": "Retrieval · top-1 entre 300"},
    {"id": "baixo", "nome": "Baixo nível", "curto": "Baixo nível"},
    {"id": "alto", "nome": "Alto nível", "curto": "Alto nível"},
    {"id": "cerebro", "nome": "Correlação cerebral", "curto": "Correlação com a resposta medida"},
    {"id": "legendas", "nome": "Legendas", "curto": "Legenda prevista × COCO"},
]

FAMILIAS = [
    {"id": "frr", "nome": "FRR", "h": 212, "s": 26, "l": 42},
    {"id": "mindeye2", "nome": "MindEye2 ridge-only", "h": 186, "s": 66, "l": 32},
    {"id": "artigo", "nome": "MindEye2 do artigo", "h": 266, "s": 46, "l": 48},
    {"id": "mindeye1", "nome": "MindEye1", "h": 24, "s": 82, "l": 45},
    {"id": "experimento", "nome": "Experimentos", "h": 140, "s": 52, "l": 35},
]
GRUPOS_SELECAO = [{"id": "frr", "nome": "FRR"}, {"id": "mindeye2", "nome": "MindEye2 ridge-only"},
                  {"id": "artigo", "nome": "MindEye2 · artigo"}, {"id": "mindeye1", "nome": "MindEye1"},
                  {"id": "experimento", "nome": "Experimentos"}]

CURVAS = [
    {"id": "teste_bwd", "nome": "Retrieval no teste · Cér→img", "fmt": "pct",
     "desc": "top-1 entre as 300 primeiras imagens de teste, ao fim de cada época"},
    {"id": "teste_fwd", "nome": "Retrieval no teste · Img→cér", "fmt": "pct",
     "desc": "top-1 entre as 300 primeiras imagens de teste, ao fim de cada época"},
    {"id": "treino_bwd", "nome": "Retrieval no treino · Cér→img", "fmt": "pct", "desc": "dentro de cada lote de treino"},
    {"id": "treino_fwd", "nome": "Retrieval no treino · Img→cér", "fmt": "pct", "desc": "dentro de cada lote de treino"},
    {"id": "treino_loss", "nome": "Loss de treino", "fmt": "dec", "desc": "a loss total, média da época"},
    {"id": "teste_loss", "nome": "Loss no teste", "fmt": "dec", "desc": "a loss total nas 300 imagens de teste"},
    {"id": "treino_loss_prior", "nome": "Loss do prior · treino", "fmt": "dec", "desc": "a parte da loss que vem do diffusion prior"},
    {"id": "teste_loss_prior", "nome": "Loss do prior · teste", "fmt": "dec", "desc": "a parte da loss que vem do diffusion prior"},
    {"id": "teste_cos_prior", "nome": "Cosseno do prior · teste", "fmt": "dec",
     "desc": "cosseno entre o embedding que o prior gera e o alvo CLIP"},
    {"id": "teste_pixcorr_borrada", "nome": "PixCorr da borrada · teste", "fmt": "dec",
     "desc": "só nos modelos com ramo blurry"},
    {"id": "lr", "nome": "Taxa de aprendizado", "fmt": "sci", "log": True, "desc": "o OneCycleLR, por época"},
]

FONTES_RETRIEVAL = {"frr": "run_frr.py, 30 sorteios de 300", "verify_retrieval": "verify_retrieval.py, 30 sorteios de 300",
                    "final_evaluations": "final_evaluations.py, 30 sorteios de 300",
                    "treino": "última época do treino, 300 imagens fixas"}

# Um resultado por configuracao: as repeticoes que mudam so a semente (do treino ou da reconstrucao)
# ficam fora do site, e fica o modelo de metricas completas. Os registros continuam em
# results/site/modelos/; para mostrar de novo, tire o id daqui.
OCULTOS = {"subj01_ridgeonly_1sess_prior_seed42", "subj01_ridgeonly_1sess_prior_seed1",
           "subj01_ridgeonly_1sess_prior_seed2", "subj01_ridgeonly_1sess_noprior_seed42",
           "subj01_ridgeonly_1sess_prior_rseed7",
           "repro_subj01_ridgeonly_1sess_prior_seed1", "repro_subj01_ridgeonly_1sess_prior_seed2"}

# Selecoes prontas (so entram os modelos que existem): a primeira e a selecao inicial da pagina, e os
# avisos de aba vazia oferecem algumas pelo id (presetBotao no site.js). Nao ha mais botoes na lista.
P = PUBLICADA
PRESETS = [
    ("principais", "Principais", "Os modelos de referência de cada família, com 1 e 40 sessões.",
     ["subj01_frr_1sess", "subj01_frr_40sess", "subj01_ridgeonly_1sess_prior", "subj01_ridgeonly_40sess_prior",
      "subj01_ridgeonly_1sess_4096blurry", "subj01_ridgeonly_40sess_4096blurry", "subj01_me1_1sess", "subj01_me1_40sess",
      "final_subj01_pretrained_1sess_24bs", "final_subj01_pretrained_40sess_24bs"]),
    ("s1", "1 sessão", "Todos os modelos de referência treinados com 1 sessão (~1 h de fMRI).",
     ["subj01_frr_1sess", "subj01_ridgeonly_1sess_prior", "subj01_ridgeonly_1sess", "subj01_ridgeonly_1sess_4096blurry",
      "subj01_me1_1sess", "final_subj01_pretrained_1sess_24bs"]),
    ("s40", "40 sessões", "Todos os modelos de referência treinados com as 40 sessões.",
     ["subj01_frr_40sess", "subj01_ridgeonly_40sess_prior", "subj01_ridgeonly_40sess_4096blurry", "subj01_me1_40sess",
      "final_subj01_pretrained_40sess_24bs"]),
    ("prior", "Efeito do prior", "Ridge 1024 de 1 sessão com e sem a loss do prior, treinados com o mesmo script e a "
     "mesma semente.",
     ["subj01_ridgeonly_1sess_prior", "subj01_ridgeonly_1sess"]),
    ("lowlevel", "MindEye1 e o low-level", "O MindEye1 com e sem a imagem borrada do low-level guiando a reconstrução.",
     ["subj01_me1_1sess", "subj01_me1_1sess_semlowlevel", "subj01_me1_40sess", "subj01_me1_40sess_semlowlevel"]),
    ("frr", "Variantes do FRR", "A grade de Doerig, uma fração só para todas as dimensões e a grade estendida.",
     ["subj01_frr_1sess", "subj01_frr_1sess_global", "subj01_frr_1sess_ext", "subj01_frr_40sess",
      "subj01_frr_40sess_global", "subj01_frr_40sess_ext"]),
    ("artigo", "Artigo e reconstruções publicadas", "Os modelos do artigo reconstruídos aqui e os tensores divulgados "
     "pelos autores: conferem o pipeline de reconstrução.",
     ["final_subj01_pretrained_1sess_24bs", "final_subj01_pretrained_1sess_24bs" + P,
      "final_subj01_pretrained_40sess_24bs", "final_subj01_pretrained_40sess_24bs" + P]),
]
PARES = [("FRR", "subj01_frr_1sess", "subj01_frr_40sess"),
         ("Ridge 1024 + prior", "subj01_ridgeonly_1sess_prior", "subj01_ridgeonly_40sess_prior"),
         ("Ridge 4096 + blurry", "subj01_ridgeonly_1sess_4096blurry", "subj01_ridgeonly_40sess_4096blurry"),
         ("MindEye1", "subj01_me1_1sess", "subj01_me1_40sess"),
         ("MindEye1 sem low-level", "subj01_me1_1sess_semlowlevel", "subj01_me1_40sess_semlowlevel"),
         ("Artigo (fine-tune completo)", "final_subj01_pretrained_1sess_24bs", "final_subj01_pretrained_40sess_24bs")]

HIPER_DESC = {
    "frr": {
        "folds": "dobras da validação cruzada, agrupada por imagem (as repetições de uma imagem nunca se separam)",
        "grid": '"doerig": as 20 frações de 0,05 a 1 (Doerig et al., 2025); "extended": mais 10, de 0,001 a 0,04 '
                '(análise de sensibilidade)',
        "fracs": "lista de frações em ordem crescente; quando dada, substitui a grade",
        "global_fraction": "uma fração só para todas as dimensões do alvo, em vez de uma por dimensão",
        "seed": "semente da divisão em dobras",
        "chunk": "dimensões do alvo por bloco; só muda a memória usada",
    },
    "frr_ms": {
        "k": "dimensão do espaço compartilhado entre os sujeitos (o gargalo voxels → k → CLIP); 1024, como o "
             "hidden_dim do MindEye2 de 1024",
        "pretreino": "sujeitos que aprendem a leitura compartilhada (2 a 8), ou \"proprio\": só o próprio subj01, "
                     "o controle de posto reduzido",
        "pesos": '"sujeito": peso igual por sujeito no pré-treino, como os lotes do MindEye2; "exibicao": cada '
                 "exibição pesa o mesmo",
        "iteracoes": "máximo de iterações dos mínimos quadrados alternados do pré-treino",
        "tolerancia": "ganho relativo mínimo da perda de validação para seguir iterando",
        "validacao": "fração das imagens de cada sujeito do pré-treino que fica de fora, para escolher as frações "
                     "e parar as iterações",
        "folds": "dobras da validação cruzada do ajuste fino no subj01, agrupada por imagem",
        "grid": "as frações testadas, como no FRR",
        "fracs": "lista de frações em ordem crescente; quando dada, substitui a grade",
        "seed": "semente da divisão de validação, das dobras e do início (SVD aleatorizado)",
        "chunk": "dimensões do alvo por bloco; só muda a memória usada",
    },
    "mindeye2": {
        "hidden_dim": "1024 (parte de multisubject_subj01_1024hid_nolow_300ep) ou 4096 (parte de final_multisubject_subj01)",
        "num_epochs": "épocas; pelo menos 2 (o OneCycleLR usa pct_start = 2/num_epochs)",
        "batch_size": "None: 16 no 1024 e 8 no 4096 + blurry, o que cabe em 20 GB",
        "max_lr": "taxa de aprendizado máxima do OneCycleLR",
        "prior": "treina com a loss do diffusion prior, que supervisiona a saída inteira do backbone (256 × 1.664)",
        "blurry": "liga o ramo da imagem borrada; exige hidden_dim 4096",
        "frozen_fp16": "guarda os módulos congelados em fp16 (~1,4 GB a menos); None: ligado no 4096 + blurry, "
                       "desligado no 1024",
        "seed": "semente",
        "ckpt_interval": "grava o last.pth a cada N épocas, para retomar; None: só na primeira época e no fim",
    },
    "mindeye1": {
        "num_epochs": "épocas do alto nível; pelo menos 2",
        "max_lr": "taxa de aprendizado máxima do OneCycleLR",
        "seed": "semente",
        "mixup_pct": "fração inicial das épocas com BiMixCo (mixup dos voxels); depois, SoftCLIP",
        "batch_size": "None: 16 (32 com paper)",
        "paper": "batch 32 e AdamW normal, como no artigo; não cabe em 20 GB",
        "save_every": "grava o last.pth a cada N épocas (cada gravação tem ~6 GB)",
    },
}
DATASET_DESC = {
    "sessoes": "o pool: as exibições das N primeiras sessões (1 a 40)",
    "n_imagens": "imagens únicas sorteadas do que sobrou; com balanceado, por classe. None: todas as elegíveis",
    "repeticoes": "exibições por imagem (1, 2 ou 3): só entram imagens com pelo menos esse número no pool, e esse "
                  "número delas é sorteado. None: todas as do pool",
    "classes": "filtro semântico. Regra exclusiva: {nome: [colunas sup_* ou cat_*]}; prioridade: lista com pessoa, "
               "animal e outro",
    "regra": '"exclusiva" ou "prioridade" (abaixo)',
    "minimo": "regra exclusiva: % mínima da tela que a classe ocupa",
    "maximo_outros": "regra exclusiva: % máxima de cada outra classe listada",
    "balanceado": "n_imagens de cada classe, em vez de no total; exige classes",
    "semente": "semente do sorteio: o mesmo dicionário dá sempre o mesmo subconjunto",
    "agregacao": "como o treino usa as repetições (tabela abaixo). None: o que cada modelo faz no benchmark",
    "subj": "sujeito; os scripts de treino são do 1",
}
AGREGACAO_DESC = {
    None: "o que cada modelo faz no benchmark: exibições separadas no FRR e no MindEye2, rodízio das repetições "
          "no MindEye1",
    "exibicoes": "cada exibição é uma amostra",
    "media": "a média dos betas das repetições: uma amostra por imagem, com menos ruído",
    "sorteio": "uma repetição sorteada por imagem a cada época",
    "combinacao": "uma combinação aleatória das repetições a cada época, como o low-level do MindEye1 "
                  "(média com pesos aleatórios, média ou uma repetição)",
}


def _e(s):
    return html.escape(str(s), quote=True)


def _py(v):
    """Valor como aparece num dicionario Python."""
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)


def _cor_familia(fid):
    f = next(x for x in FAMILIAS if x["id"] == fid)
    return f"hsl({f['h']} {f['s']}% calc({f['l']}% + var(--ajuste-l)))"


# --------------------------------------------------------------------------- tabelas geradas do codigo

def html_hiper(modelo):
    import treina
    padrao = treina.MODELOS[modelo].padrao
    linhas = "".join(f"<tr><td><code>{_e(k)}</code></td><td><code>{_e(_py(v))}</code></td>"
                     f"<td>{_e(HIPER_DESC[modelo].get(k, ''))}</td></tr>" for k, v in padrao.items())
    return (f'<div class="rolagem"><table class="tab"><thead><tr><th>Chave</th><th>Padrão</th><th>O que faz</th></tr>'
            f"</thead><tbody>{linhas}</tbody></table></div>")


def html_campos_dataset():
    from mindeye_ridge import dataset_controlado as dc
    padrao = dc.ConfigDataset().como_dict()
    linhas = "".join(f"<tr><td><code>{_e(k)}</code></td><td><code>{_e(_py(v))}</code></td>"
                     f"<td>{_e(DATASET_DESC.get(k, ''))}</td></tr>" for k, v in padrao.items())
    return (f'<div class="rolagem"><table class="tab"><thead><tr><th>Campo</th><th>Padrão</th><th>O que faz</th></tr>'
            f"</thead><tbody>{linhas}</tbody></table></div>")


def html_agregacao():
    import treina
    from mindeye_ridge import agregacao
    modelos = [("frr", "FRR"), ("mindeye2", "MindEye2"), ("mindeye1", "MindEye1")]
    cab = "".join(f'<th class="c">{n}</th>' for _, n in modelos)
    linhas = []
    for modo in (None,) + tuple(agregacao.MODOS):
        cels = "".join('<td class="c"><span class="sim">✓</span></td>' if modo is None or modo in treina.MODELOS[m].agregacoes
                       else '<td class="c"><span class="nao">não</span></td>' for m, _ in modelos)
        nome = "None" if modo is None else f'"{modo}"'
        linhas.append(f"<tr><td><code>{nome}</code></td><td>{_e(AGREGACAO_DESC[modo])}</td>{cels}</tr>")
    return (f'<div class="rolagem"><table class="tab"><thead><tr><th>agregacao</th><th>Amostra de treino</th>{cab}'
            f'</tr></thead><tbody>{"".join(linhas)}</tbody></table></div>')


def html_metricas():
    nomes = {g["id"]: g["nome"] for g in GRUPOS_METRICAS}
    linhas = []
    for m in METRICAS:
        acaso = "—" if m["acaso"] is None else ("1 em 300" if m["acaso"] < 0.01 else "50%")
        linhas.append(f'<tr id="def-{m["slug"]}"><td>{_e(nomes[m["grupo"]])}</td><td><a class="elo-metrica" '
                      f'href="#res-{m["slug"]}" title="Ver os resultados desta métrica"><b>{_e(m["nome"])}</b></a></td>'
                      f'<td class="c">{"↑" if m["maior"] else "↓"}</td><td>{acaso}</td><td>{_e(m["desc"])}</td></tr>')
    return ('<div class="rolagem"><table class="tab"><thead><tr><th>Grupo</th><th>Métrica</th><th class="c">Melhor</th>'
            f'<th>Acaso</th><th>O que mede</th></tr></thead><tbody>{"".join(linhas)}</tbody></table></div>')


def html_supercategorias(semantica):
    from .semantica_site import NOMES_PT, SUPERCATEGORIAS
    nomes = (semantica or {}).get("nomes") or NOMES_PT
    return '<div class="chips">' + "".join(
        f'<span class="chip">{_e(nomes.get(s, s))} <code>sup_{_e(s)}</code></span>' for s in SUPERCATEGORIAS) + "</div>"


def html_cartoes_familias():
    """Os tres modelos, para a abertura do Inicio."""
    cartoes = [
        ("frr", "FRR", "Regressão ridge fracionária dos voxels direto para o embedding CLIP. Sem rede e sem "
                       "reconstrução: é o baseline linear do plano de estágio."),
        ("mindeye2", "MindEye2 ridge-only", "Parte do modelo pré-treinado nos outros 7 sujeitos e treina só a camada "
                                           "ridge, a única parte específica de cada pessoa. Gera a imagem com o unCLIP "
                                           "do SDXL."),
        ("mindeye1", "MindEye1", "Treinado do zero nos mesmos dados, sem pré-treino. Gera a imagem com o Versatile "
                                 "Diffusion, guiado pela imagem borrada que um low-level separado prevê."),
    ]
    return '<div class="modelos-cards">' + "".join(
        f'<article class="modelo-card" style="--fam:{_cor_familia(f)}"><h3>{_e(t)}</h3><p>{_e(d)}</p></article>'
        for f, t, d in cartoes) + "</div>"


def preenche(texto, valores, modelos):
    """Troca os {{chave}} do texto. {{tempo:<id>}} e {{ridge:<id>}} vem dos registros dos modelos."""
    por_id = {m["id"]: m for m in modelos}

    def troca(r):
        chave = r[1]
        if chave.startswith("tempo:"):
            m = por_id.get(chave[6:])
            return _e(duracao(m["tempo_s"])) if m and m.get("tempo_s") else "—"
        if chave.startswith("ridge:"):
            m = por_id.get(chave[6:])
            p = (m or {}).get("params") or {}
            return (f"{100 * p['ridge'] / sum(p.values()):.2f}%".replace(".", ",")) if p.get("ridge") else "uma fração pequena"
        if chave not in valores:
            raise KeyError(f"conteudo: {{{{{chave}}}}} sem valor")
        return valores[chave]
    return re.sub(r"\{\{([^}]+)\}\}", troca, texto)


# --------------------------------------------------------------------------- dados da pagina

def _compat_leitura(modelos):
    """O formato que texts.leitura_frr espera (o da pagina antiga)."""
    ms, variantes = [], []
    for m in modelos:
        d = {"id": m["id"], "tabelas": {"enh": (m.get("valores") or {}).get("refinada"),
                                        "base": (m.get("valores") or {}).get("unclip")}}
        if m["familia"] == "frr" and m.get("frr") and m.get("retrieval"):
            d["frr_dados"] = {**m["frr"], "retrieval": {k: m["retrieval"][k] for k in ("fwd", "bwd")},
                              "retrieval_centrado": m.get("retrieval_centrado")}
            variantes.append({"id": m["id"], "dados": d["frr_dados"]})
        ms.append(d)
    return {"modelos": ms, "frr_variantes": variantes}


def _ordena_config(cfg):
    """Hiperparametros e dataset na ordem em que o codigo os declara (os registros guardam em ordem
    alfabetica); chaves fora dessa ordem vao para o fim."""
    import treina
    from mindeye_ridge import dataset_controlado as dc
    if not cfg or cfg.get("modelo") not in treina.MODELOS:
        return cfg

    def ordena(d, ordem):
        d = d or {}
        return {**{k: d[k] for k in ordem if k in d}, **{k: v for k, v in d.items() if k not in ordem}}
    return {**cfg, "hiper": ordena(cfg.get("hiper"), list(treina.MODELOS[cfg["modelo"]].padrao)),
            "dataset": ordena(cfg.get("dataset"), list(dc.ConfigDataset().como_dict()))}


def dados_pagina(modelos, planejados, galeria, semantica):
    import treina
    from mindeye_ridge import dataset_controlado as dc
    try:
        from experimentos import BENCHMARK
    except ImportError:
        BENCHMARK = {}
    ms = []
    for m in modelos:
        valores = {}
        for tipo, v in (m.get("valores") or {}).items():
            if v:
                valores[tipo] = {k: x for k, x in v.items() if k in IDS_METRICAS}
        leg = m.get("legendas")
        if leg:
            for chave, *_ in _LEG:
                for tipo in valores:
                    valores[tipo][f"leg:{chave}"] = leg.get(f"{chave}_brain_ref")
        ret = m.get("retrieval")
        if ret:
            ret = {"fwd": {k: ret["fwd"][k] for k in ("media", "ic95") if k in ret["fwd"]},
                   "bwd": {k: ret["bwd"][k] for k in ("media", "ic95") if k in ret["bwd"]}, "fonte": ret.get("fonte")}
        curva = m.get("curva")
        if curva:
            curva = {"epocas": curva["epocas"], "series": curva["series"]}
        ms.append({**{k: m.get(k) for k in (
            "id", "rotulo", "familia", "grupo", "sessoes", "status", "status_txt", "espaco_clip", "publicada", "config",
            "config_fonte", "config_erro", "config_descritiva", "resumo_dataset", "retrieval_centrado", "tempo_s",
            "params", "receita", "notas", "fontes", "commit", "frr")},
            "valores": valores or None, "retrieval": ret, "curva": curva,
            "config": m.get("config") if m.get("config_descritiva") else _ordena_config(m.get("config"))})
    ids = {m["id"] for m in ms if m["valores"] or m["curva"]}
    presets = [{"id": i, "nome": n, "desc": d, "modelos": [x for x in lista if x in ids]} for i, n, d, lista in PRESETS]
    presets = [p for p in presets if len(p["modelos"]) >= 2]
    exp = [m["id"] for m in ms if m["grupo"] == "experimento" and m["id"] in ids]
    if exp:
        presets.append({"id": "experimentos", "nome": "Experimentos", "desc": "Os modelos treinados num dataset "
                        "controlado (fora do benchmark).", "modelos": exp})
    padrao = presets[0]["modelos"] if presets else sorted(ids)[:8]
    teto = next((m["legendas"] for m in modelos if m.get("legendas")), None)
    leitura = texts.leitura(_compat_leitura(modelos))
    return {
        "modelos": ms, "planejados": planejados, "metricas": METRICAS, "grupos_metricas": GRUPOS_METRICAS,
        "familias": FAMILIAS, "grupos": GRUPOS_SELECAO, "curvas": CURVAS, "fontes_retrieval": FONTES_RETRIEVAL,
        "presets": presets, "padrao": padrao,
        "pares": [{"nome": n, "de": a, "para": b} for n, a, b in PARES],
        "padroes": {"hiper": {k: v.padrao for k, v in treina.MODELOS.items()}, "dataset": dc.ConfigDataset().como_dict()},
        "benchmark_ids": list(BENCHMARK), "galeria": galeria, "semantica": semantica, "leitura": leitura,
        "teto_legendas": {k: teto.get(f"{k}_img_ref") for k, *_ in _LEG} if teto else None,
        "textos": {"frr_hubness": texts.FRR_NOTA_HUBNESS},
    }


# --------------------------------------------------------------------------- pagina

FONTES = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
          '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
          '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@'
          '0,8..60,400..700;1,8..60,400..700&family=Source+Sans+3:ital,wght@0,400..700;1,400..700'
          '&family=Source+Code+Pro:wght@400;500&display=swap">')

RESULTADOS = """
<section class="pagina" id="pag-resultados" data-pag="resultados" aria-labelledby="t-resultados">
  <header class="res-cab">
    <div class="linha"><h1 id="t-resultados">O benchmark, modelo a modelo</h1></div>
    <p class="nota">Escolha os modelos na lista; todas as abas mostram a mesma seleção. As escolhas ficam guardadas
    neste navegador. Um modelo novo entra aqui sozinho quando o treino ou a avaliação termina
    (<a href="#uso-novo">como</a>).</p>
  </header>
  <div class="bancada">
    <details class="trilho-modelos" id="trilho" open>
      <summary><b>Modelos</b><span class="mudo num" id="contagem"></span></summary>
      <div class="trilho-corpo">
        <label class="controle" for="busca">Filtrar <input type="search" id="busca" placeholder="nome do modelo"></label>
        <div id="lista-modelos"></div>
        <div class="trilho-pe">
          <button class="botao leve" type="button" id="sel-todos">Marcar os visíveis</button>
          <button class="botao leve" type="button" id="sel-nenhum">Limpar</button>
          <button class="botao leve" type="button" id="dobra-todos">Recolher grupos</button>
        </div>
      </div>
    </details>
    <div id="vistas">
      <div class="abas" id="abas" role="tablist" aria-label="Vistas dos resultados"></div>
      {vistas}
      <noscript><p class="vazio">Os resultados interativos precisam de JavaScript. As tabelas estão em
      BENCHMARK.md e em results/tables/.</p></noscript>
    </div>
  </div>
</section>
"""
ABAS = ["resumo", "tabela", "graficos", "dispersao", "treino", "galeria", "frr", "fichas"]


def monta(modelos, planejados, galeria, semantica):
    """Devolve (cabeca, corpo): o <title>, as fontes e o estilo, e o resto da pagina."""
    todos = modelos                       # os ocultos ainda valem para os tempos citados nos textos
    referencias = {m["reproducao"] for m in modelos if m.get("reproducao")}    # lidas no lugar do modelo de referencia
    modelos = [m for m in modelos if m["id"] not in OCULTOS and m["id"] not in referencias]
    agora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    rev = commit()
    avaliados = sum(m["status"] == "completo" for m in modelos)
    valores = {
        "repo": REPO_URL, "repo_curto": REPO_URL.replace("https://", ""), "n_modelos": str(len(modelos)),
        "n_avaliados": str(avaliados), "n_planejados": str(len(planejados)),
        "metricas": html_metricas(), "campos_dataset": html_campos_dataset(), "agregacao": html_agregacao(),
        "supercategorias": html_supercategorias(semantica), "hiper:frr": html_hiper("frr"),
        "hiper:frr_ms": html_hiper("frr_ms"), "hiper:mindeye2": html_hiper("mindeye2"), "hiper:mindeye1": html_hiper("mindeye1"),
        "cartoes_familias": html_cartoes_familias(),
        "cor:frr": _cor_familia("frr"), "cor:mindeye2": _cor_familia("mindeye2"), "cor:mindeye1": _cor_familia("mindeye1"),
    }
    inicio = preenche((AQUI / "conteudo" / "inicio.html").read_text(encoding="utf-8"), valores, todos)
    uso = preenche((AQUI / "conteudo" / "uso.html").read_text(encoding="utf-8"), valores, todos)
    vistas = "".join(f'<div class="vista" id="vista-{a}" data-aba="{a}" role="tabpanel" aria-labelledby="aba-{a}" hidden>'
                     f"</div>" for a in ABAS)
    dados = json.dumps(dados_pagina(modelos, planejados, galeria, semantica), ensure_ascii=False,
                       separators=(",", ":")).replace("</", "<\\/")
    estilo = (AQUI / "assets" / "site.css").read_text(encoding="utf-8")
    script = (AQUI / "assets" / "simulador.js").read_text(encoding="utf-8") + "\n" + (AQUI / "assets" / "site.js").read_text(encoding="utf-8")
    cabeca = f"<title>benchmark-modelos</title>\n{FONTES}\n<style>{estilo}</style>"
    corpo = f"""<script>document.documentElement.classList.add('js');</script>
<header class="barra">
  <div class="barra-dentro">
    <a class="marca" href="#inicio">benchmark-modelos</a>
    <nav class="paginas" aria-label="Páginas">
      <a href="#inicio" data-pag="inicio" aria-current="page">Início</a>
      <a href="#uso" data-pag="uso">Como usar</a>
      <a href="#resultados" data-pag="resultados">Resultados</a>
      <a href="apresentacao.html">Apresentação</a>
    </nav>
    <button class="tema" type="button">Tema escuro</button>
  </div>
</header>
<main>
<section class="pagina ativa" id="pag-inicio" data-pag="inicio" aria-label="Início">{inicio}</section>
<section class="pagina" id="pag-uso" data-pag="uso" aria-label="Como usar">{uso}</section>
{RESULTADOS.replace("{vistas}", vistas)}
</main>
<footer class="rodape">Gerado em {agora} a partir do commit <span class="mono">{_e(rev)}</span> por
<span class="mono">src/make_site.py</span>. {len(modelos)} modelos com resultado, {avaliados} avaliados por completo,
{len(planejados)} experimentos planejados.</footer>
<dialog class="lupa" id="lupa" aria-label="Imagens ampliadas"><div class="lupa-corpo"></div></dialog>
<script type="application/json" id="dados">{dados}</script>
<script>{script}</script>
"""
    return cabeca, corpo


def documento(cabeca, corpo):
    """A pagina completa, para abrir do disco."""
    return ('<!doctype html>\n<html lang="pt-BR">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{cabeca}\n</head>\n<body>\n{corpo}\n</body>\n</html>\n")
