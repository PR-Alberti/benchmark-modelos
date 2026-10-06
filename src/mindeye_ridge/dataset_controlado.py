"""Subconjuntos controlados do treino: sessoes, imagens unicas, repeticoes e classes semanticas.

O plano de estagio pergunta quanto dado e preciso; para isso o treino tem de variar um eixo de
cada vez. Hoje o unico eixo e o numero de sessoes, que mistura imagens unicas com repeticoes (com
20 sessoes, 6.708 imagens divididas em tercos entre 1, 2 e 3 exibicoes). Aqui o subconjunto e
sorteado sobre as **exibicoes**:

    cfg = ConfigDataset(sessoes=40, n_imagens=1000, repeticoes=1, semente=0)
    sub = monta(cfg, data_path)          # mesmo formato do nsd_data.exibicoes_treino
    salva(sub, "dataset.json")           # o que os treinos leem (--dataset)

A ordem das escolhas:
1. pool: as exibicoes das primeiras `sessoes` sessoes;
2. classes (opcional): so as imagens com rotulo semantico (semantica.py) numa das classes;
3. repeticoes (opcional): so as imagens com pelo menos `repeticoes` exibicoes no pool;
4. n_imagens (opcional): sorteio de imagens com a semente, no total ou por classe (`balanceado`);
5. de cada imagem, `repeticoes` exibicoes sorteadas (ou todas, se `repeticoes` e None).

Com tudo no padrao, o subconjunto e exatamente o das N primeiras sessoes, o que os modelos ja
usavam. O teste nao muda: sao sempre as 1.000 imagens compartilhadas.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import nsd_data, semantica

REGRAS = ("exclusiva", "prioridade")
CLASSES_PRIORIDADE = ("pessoa", "animal", "outro")


@dataclass
class ConfigDataset:
    """O dataset de treino controlado. Os campos sao os do dicionario "dataset" dos experimentos."""
    sessoes: int = 40                   # pool: as primeiras N sessoes (1 a 40)
    n_imagens: int | None = None        # imagens unicas; None = todas as elegiveis
    repeticoes: int | None = None       # exibicoes por imagem (1 a 3); None = todas as do pool
    classes: dict | list | None = None  # exclusiva: {nome: [colunas sup_*/cat_*]}; prioridade: nomes a manter
    regra: str = "exclusiva"            # "exclusiva" ou "prioridade" (semantica.py)
    minimo: float = 10                  # exclusiva: % minima da tela da classe
    maximo_outros: float = 2            # exclusiva: % maxima de cada outra classe
    balanceado: bool = False            # n_imagens por classe, em vez de no total
    semente: int = 0
    subj: int = 1
    extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not 1 <= self.sessoes <= 40:
            raise ValueError(f"sessoes = {self.sessoes}: tem de estar entre 1 e 40")
        if self.repeticoes is not None and not 1 <= self.repeticoes <= 3:
            raise ValueError(f"repeticoes = {self.repeticoes}: tem de ser 1, 2, 3 ou None")
        if self.n_imagens is not None and self.n_imagens < 1:
            raise ValueError(f"n_imagens = {self.n_imagens}: tem de ser positivo ou None")
        if self.regra not in REGRAS:
            raise ValueError(f"regra = {self.regra!r}: use uma de {REGRAS}")
        if self.balanceado and not self.classes and self.regra == "exclusiva":
            raise ValueError("balanceado exige classes")
        if self.regra == "prioridade" and self.classes is not None:
            invalidas = set(self.classes) - set(CLASSES_PRIORIDADE)
            if isinstance(self.classes, dict) or invalidas:
                raise ValueError(f"com a regra prioridade, classes e uma lista de {CLASSES_PRIORIDADE} (ou None)")
        if self.extra:
            raise ValueError(f"campos desconhecidos no dataset: {sorted(self.extra)}")

    @classmethod
    def de_dict(cls, d):
        """Do dicionario do experimento; chaves desconhecidas viram erro, nao sao ignoradas."""
        d = dict(d or {})
        conhecidos = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__ and k != "extra"}
        return cls(**conhecidos, extra=d)

    @property
    def semantico(self):
        return self.classes is not None or self.regra == "prioridade"

    def como_dict(self):
        d = asdict(self)
        d.pop("extra")
        return d

    def assinatura(self):
        """Hash curto da configuracao: o mesmo dicionario da sempre o mesmo subconjunto."""
        return hashlib.sha1(json.dumps(self.como_dict(), sort_keys=True).encode()).hexdigest()[:8]


def rotulos(cfg, fracoes):
    """Rotulo semantico de cada imagem da tabela de fracoes (None = descartada)."""
    if cfg.regra == "prioridade":
        rot = semantica.rotulo_prioridade(fracoes)
        if cfg.classes is not None:
            rot = rot.where(rot.isin(list(cfg.classes)), None)
        return rot
    return semantica.rotulo_exclusivo(fracoes, cfg.classes, cfg.minimo, cfg.maximo_outros)


def monta(cfg, data_path, exibicoes=None, fracoes=None):
    """Exibicoes de treino do subconjunto: dict com imagem, beta e sessao (como o
    nsd_data.exibicoes_treino), mais rotulo (por exibicao, None sem filtro semantico), config e resumo.

    `exibicoes` e `fracoes` substituem a leitura do disco (testes).
    """
    if exibicoes is None:
        exibicoes = nsd_data.exibicoes_treino(data_path, cfg.subj, cfg.sessoes)
    no_pool = exibicoes["sessao"] < cfg.sessoes
    ex = {k: np.asarray(v)[no_pool] for k, v in exibicoes.items()}
    rng = np.random.default_rng(cfg.semente)

    ids, contagem = np.unique(ex["imagem"], return_counts=True)
    elegivel = pd.Series(True, index=ids)
    rot = pd.Series(None, index=ids, dtype=object)
    if cfg.semantico:
        if fracoes is None:
            fracoes = semantica.tabela_fracoes(data_path, cfg.subj)
        rot = rotulos(cfg, fracoes).reindex(ids)
        rot = rot.astype(object).where(rot.notna(), None)      # imagem fora da tabela: descartada
        elegivel &= rot.notna()
    if cfg.repeticoes is not None:
        elegivel &= contagem >= cfg.repeticoes

    candidatas = elegivel.index[elegivel].to_numpy()
    if cfg.n_imagens is not None:
        if cfg.balanceado:
            grupos = {c: candidatas[rot[candidatas].to_numpy() == c] for c in _nomes_classes(cfg)}
            faltam = {c: len(g) for c, g in grupos.items() if len(g) < cfg.n_imagens}
            if faltam:
                raise ValueError(f"n_imagens = {cfg.n_imagens} por classe, mas ha so {faltam} "
                                 f"(disponiveis: { {c: len(g) for c, g in grupos.items()} })")
            escolhidas = np.concatenate([rng.choice(g, cfg.n_imagens, replace=False) for g in grupos.values()])
        else:
            if cfg.n_imagens > len(candidatas):
                raise ValueError(f"n_imagens = {cfg.n_imagens}, mas so {len(candidatas)} imagens sao elegiveis")
            escolhidas = rng.choice(candidatas, cfg.n_imagens, replace=False)
    else:
        escolhidas = candidatas
    if not len(escolhidas):
        raise ValueError(f"nenhuma imagem elegivel: {cfg.como_dict()}")
    escolhidas = np.sort(escolhidas)

    # as exibicoes de cada imagem escolhida: todas, ou `repeticoes` sorteadas
    linhas = np.flatnonzero(np.isin(ex["imagem"], escolhidas))
    if cfg.repeticoes is not None:
        por_imagem = pd.Series(linhas).groupby(ex["imagem"][linhas])
        linhas = np.sort(np.concatenate([rng.choice(g.to_numpy(), cfg.repeticoes, replace=False)
                                         for _, g in por_imagem]))
    sub = {k: v[linhas] for k, v in ex.items()}
    sub["rotulo"] = np.array([r if isinstance(r, str) else None for r in rot.reindex(sub["imagem"])], dtype=object)
    sub["config"] = cfg.como_dict()
    sub["resumo"] = resumo(sub)
    return sub


def _nomes_classes(cfg):
    if cfg.regra == "prioridade":
        return list(cfg.classes) if cfg.classes is not None else list(CLASSES_PRIORIDADE)
    return list(cfg.classes)


def resumo(sub):
    imagens, reps = np.unique(sub["imagem"], return_counts=True)
    r = {"exibicoes": int(len(sub["imagem"])), "imagens_unicas": int(len(imagens)),
         "repeticoes": {int(k): int(v) for k, v in zip(*np.unique(reps, return_counts=True))},
         "sessoes_usadas": int(len(np.unique(sub["sessao"])))}
    if pd.Series(sub["rotulo"]).notna().any():
        rot_img = pd.Series(sub["rotulo"]).groupby(sub["imagem"]).first()
        r["imagens_por_classe"] = {str(k): int(v) for k, v in rot_img.value_counts().sort_index().items()}
    return r


def salva(sub, caminho):
    """Manifesto JSON: config, resumo e as exibicoes (imagem, beta, sessao, rotulo)."""
    with open(caminho, "w") as f:
        json.dump({"config": sub["config"], "resumo": sub["resumo"],
                   "imagem": sub["imagem"].tolist(), "beta": sub["beta"].tolist(),
                   "sessao": sub["sessao"].tolist(), "rotulo": list(sub["rotulo"])}, f)


def carrega(caminho):
    """O manifesto de volta, com as exibicoes em arrays (as chaves do nsd_data.exibicoes_treino)."""
    with open(caminho) as f:
        d = json.load(f)
    return {"imagem": np.asarray(d["imagem"], dtype=int), "beta": np.asarray(d["beta"], dtype=int),
            "sessao": np.asarray(d["sessao"], dtype=int), "rotulo": np.asarray(d["rotulo"], dtype=object),
            "config": d["config"], "resumo": d["resumo"]}


def exibicoes(data_path, subj, num_sessions, dataset=None):
    """O que os treinos chamam: as exibicoes do manifesto `dataset`, se houver; senao as N
    primeiras sessoes inteiras, que e tambem o que o manifesto padrao (o dataset completo) contem.

    Confere que o manifesto e do mesmo sujeito, que cada exibicao dele existe, com a mesma imagem,
    nas sessoes de treino que o treino abre (o que tambem garante que nenhuma imagem do teste
    entrou) e que nenhuma exibicao se repete.
    """
    pool = nsd_data.exibicoes_treino(data_path, subj, num_sessions)
    if not dataset:
        return pool
    sub = carrega(dataset)
    if sub["config"]["subj"] != subj:
        raise ValueError(f"o dataset {dataset} e do subj0{sub['config']['subj']}, nao do subj0{subj}")
    if not len(sub["beta"]):
        raise ValueError(f"o dataset {dataset} esta vazio")
    if sub["sessao"].max() >= num_sessions:
        raise ValueError(f"o dataset {dataset} usa a sessao {sub['sessao'].max() + 1}, "
                         f"mas o treino abre so {num_sessions}; use num_sessions >= {sub['config']['sessoes']}")
    imagem_da_linha = dict(zip(pool["beta"].tolist(), pool["imagem"].tolist()))
    fora = [(i, b) for i, b in zip(sub["imagem"].tolist(), sub["beta"].tolist()) if imagem_da_linha.get(b) != i]
    if fora:
        raise ValueError(f"o dataset {dataset} tem {len(fora)} exibicoes (imagem, linha de betas) que nao estao "
                         f"nas {num_sessions} primeiras sessoes de treino do subj0{subj}, por exemplo {fora[:3]}")
    if len(np.unique(sub["beta"])) != len(sub["beta"]):
        raise ValueError(f"o dataset {dataset} repete exibicoes")
    print(f"dataset controlado {dataset}: {sub['resumo']}", flush=True)
    return {k: sub[k] for k in ("imagem", "beta", "sessao")}


def amostras_por_epoca_mindeye2(n_exibicoes, n_pool, num_sessions):
    """Amostras por epoca do MindEye2 (train_ridgeonly.py) treinando em `n_exibicoes` das `n_pool`
    exibicoes de treino das `num_sessions` sessoes.

    O original define a epoca como 750 * num_sessions, os trials nominais das sessoes, que incluem
    os das imagens de teste (fora dos tars de treino): com 40 sessoes, 30.000 sorteios para 27.000
    exibicoes. Com um subconjunto a epoca encolhe na mesma proporcao das exibicoes, e o pool inteiro
    da exatamente o numero do original.
    """
    return 750 * num_sessions * n_exibicoes // n_pool
