"""Os modelos do benchmark, os grupos de comparacao e as metricas de cada tabela."""

from mindeye_ridge import paths


REPO, DATA, SAIDA = paths.REPO, paths.DATA, paths.BENCHMARK


# Ordem das linhas e das colunas da galeria. "grupo" separa as comparacoes
# justas: 1 sessao contra 1 sessao, 40 contra 40.
MODELOS = [
    # baseline linear: sem rede, sem reconstrucao; so retrieval e similaridade com o embedding CLIP
    dict(id="subj01_frr_1sess", rotulo="FRR · 1 sessão", frr=True, grupo=1, sessoes=1, hidden=None,
         blurry=False, prior="sem prior, sem rede",
         treino="20 frações · CV 5 dobras · uma fração por dimensão", partida=None),
    dict(id="subj01_frr_40sess", rotulo="FRR · 40 sessões", frr=True, grupo=40, sessoes=40, hidden=None,
         blurry=False, prior="sem prior, sem rede",
         treino="20 frações · CV 5 dobras · uma fração por dimensão", partida=None),
    # MindEye1 treinado do zero nas mesmas sessoes e testado no mesmo teste (MINDEYE1.md). Nao tem
    # refinamento: a reconstrucao final (Versatile Diffusion + img2img da borrada) entra nas duas abas
    dict(id="subj01_me1_1sess", rotulo="MindEye1 · 1 sessão", me1=True, grupo=1, sessoes=1, hidden=4096,
         blurry=True, prior="do zero, com o prior", baixo="subj01_me1_lowlevel_1sess",
         treino="240 épocas + 120 do baixo nível · batch 16 · Adam 8 bits", partida=None),
    dict(id="subj01_me1_40sess", rotulo="MindEye1 · 40 sessões", me1=True, grupo=40, sessoes=40, hidden=4096,
         blurry=True, prior="do zero, com o prior", baixo="subj01_me1_lowlevel_40sess",
         treino="240 épocas + 120 do baixo nível · batch 16 · Adam 8 bits", partida=None),
    dict(id="subj01_ridgeonly_1sess_prior", rotulo="Ridge 1024 + prior",
         grupo=1, sessoes=1, hidden=1024, blurry=False, prior="com a loss do prior",
         treino="150 épocas · batch 16", partida="multisubject_subj01_1024hid_nolow_300ep",
         tempo_doc="~1 h 50 (corrida anterior)"),
    dict(id="subj01_ridgeonly_1sess_4096blurry", rotulo="Ridge 4096 + blurry",
         grupo=1, sessoes=1, hidden=4096, blurry=True, prior="com a loss do prior",
         treino="150 épocas · batch 8 · congelados em fp16", partida="final_multisubject_subj01"),
    dict(id="subj01_ridgeonly_1sess", rotulo="Ridge 1024 sem prior",
         grupo=1, sessoes=1, hidden=1024, blurry=False,
         prior="sem o prior; a reconstrução usa o do ponto de partida",
         treino="150 épocas · batch 16", partida="multisubject_subj01_1024hid_nolow_300ep"),
    dict(id="subj01_ridgeonly_40sess_prior", rotulo="Ridge 1024 + prior · 40 sessões",
         grupo=40, sessoes=40, hidden=1024, blurry=False, prior="com a loss do prior",
         treino="20 épocas · batch 16 · congelados em fp16", partida="multisubject_subj01_1024hid_nolow_300ep"),
    dict(id="final_subj01_pretrained_1sess_24bs", rotulo="Paper · 1 sessão", paper=True,
         grupo=1, sessoes=1, hidden=4096, blurry=True, prior="fine-tune completo",
         treino="150 épocas · batch 24 · 8×A100", partida="final_multisubject_subj01"),
    dict(id="final_subj01_pretrained_40sess_24bs", rotulo="Paper · 40 sessões", paper=True,
         grupo=40, sessoes=40, hidden=4096, blurry=True, prior="fine-tune completo",
         treino="150 épocas · batch 24 · 8×A100", partida="final_multisubject_subj01"),
]


GRUPOS = {1: "1 sessão · ~1 h de fMRI, 750 exibições",
          40: "40 sessões · ~40 h de fMRI, 30.000 exibições"}


# Variantes do FRR para a tabela de sensibilidade (so entram as que ja rodaram)
FRR_VARIANTES = [
    ("subj01_frr_1sess", "Grade de Doerig: 0,05 a 1 (principal)", 1),
    ("subj01_frr_1sess_global", "Uma fração para todas as dimensões", 1),
    ("subj01_frr_1sess_ext", "Grade estendida: 0,001 a 1", 1),
    ("subj01_frr_40sess", "Grade de Doerig: 0,05 a 1 (principal)", 40),
    ("subj01_frr_40sess_global", "Uma fração para todas as dimensões", 40),
    ("subj01_frr_40sess_ext", "Grade estendida: 0,001 a 1", 40),
]


ROT_FWD, ROT_BWD = "Img→cér", "Cér→img"


# (chave na tabela do final_evaluations, rotulo, maior e melhor, formato)
METRICAS = [
    ("PixCorr", "PixCorr", True, "dec"), ("SSIM", "SSIM", True, "dec"),
    ("AlexNet(2)", "Alex(2)", True, "pct"), ("AlexNet(5)", "Alex(5)", True, "pct"),
    ("InceptionV3", "Incep", True, "pct"), ("CLIP", "CLIP", True, "pct"),
    ("EffNet-B", "Eff", False, "dec"), ("SwAV", "SwAV", False, "dec"),
    ("FwdRetrieval", ROT_FWD, True, "pct"), ("BwdRetrieval", ROT_BWD, True, "pct"),
]


# O codigo do MindEye2 chama de fwd o sentido imagem -> cerebro (cada imagem procura o seu
# cerebro entre as 300 previsoes) e de bwd o contrario. E o oposto do que o comentario do
# final_evaluations.py diz ("fwd: brain, clip") e do que o artigo define como "Image Retrieval"
# (cerebro -> imagem); conferido com um exemplo sintetico (embedding_metrics.py). As colunas
# seguem o que o codigo calcula, e por isso levam o sentido em vez de "Imagem" e "Cérebro".
CEREBRO = [("Brain Corr. nsd_general", "nsdgeneral"), ("Brain Corr. V1", "V1"),
           ("Brain Corr. V2", "V2"), ("Brain Corr. V3", "V3"), ("Brain Corr. V4", "V4"),
           ("Brain Corr. higher_vis", "Alto nível")]


# O final_evaluations.py grava o *_caption_metrics.csv com index=False: sai so a
# coluna de valores, nesta ordem (a do dicionario caption_metrics de la).
CHAVES_LEGENDA = [f"{m}_{s}" for m in ("Rouge1", "RougeL", "Meteor", "Sentence", "CLIP-B", "CLIP-L")
                  for s in ("img_ref", "brain_ref", "brain_img", "relative")]


LEGENDAS = [("Meteor", "METEOR"), ("Rouge1", "ROUGE-1"), ("RougeL", "ROUGE-L"),
            ("Sentence", "Sentence-T"), ("CLIP-B", "CLIP-B"), ("CLIP-L", "CLIP-L")]


TIPOS = [("enh", "Refinadas"), ("base", "unCLIP"), ("blur", "Blurry")]


ARQ = {"enh": "all_enhancedrecons", "base": "all_recons", "blur": "all_blurryrecons"}
