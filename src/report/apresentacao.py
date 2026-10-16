"""A apresentacao do trabalho (results/site/apresentacao.html): slides que abrem do disco, ao lado do site.

Conta a historia do benchmark em ordem (a pergunta, os dados, os tres modelos com os seus diagramas, as
metricas, os resultados e o controle do dataset), com os numeros tirados dos mesmos registros que o
site usa (catalogo.py), as imagens das tiras da galeria (galeria_site.py) e os exemplos do controle
semantico (exemplos_site.py). O texto dos slides esta em conteudo/apresentacao.html, com {{chave}}
para os numeros; o comportamento, em assets/apresentacao.js (e a regra exclusiva no simulador.js).
"""
import datetime
import json

from . import site
from .fmt import duracao, milhar, mi
from .semantica_site import NOMES_PT

AQUI = site.AQUI
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
         "novembro", "dezembro"]

# as classes da regra exclusiva no slide e no explorador: as do notebook e de src/experimentos.py
CLASSES_5 = [("pessoa", ["person"]), ("animal", ["animal"]), ("veículo", ["vehicle"]), ("comida", ["food"]),
             ("móveis", ["furniture"])]
ORDEM_CLASSES = ["person", "animal", "vehicle", "food", "furniture", "kitchen", "electronic", "appliance", "indoor",
                 "accessory", "sports", "outdoor"]

# retrieval com 1 e 40 sessoes: os pares do site, menos o MindEye1 sem low-level (o mesmo embedding)
PARES = [p for p in site.PARES if "semlowlevel" not in p[1]]

# a galeria da apresentacao: (id, tipo de reconstrucao, rotulo, detalhe)
COLUNAS_GALERIA = [
    ("subj01_ridgeonly_40sess_prior", "refinada", "Ridge-only", " · 40 sessões"),
    ("subj01_ridgeonly_1sess_prior", "refinada", "Ridge-only", " · 1 sessão"),
    ("final_subj01_pretrained_1sess_24bs", "refinada", "Artigo", " · 1 sessão"),
    ("subj01_me1_40sess", "unclip", "MindEye1", " · 40 sessões"),
    ("subj01_me1_1sess", "unclip", "MindEye1", " · 1 sessão"),
]

# custo x desempenho: (id, rotulo no grafico, dx, dy, ancora) -- as posicoes evitam rotulos sobrepostos
CUSTO = [
    ("subj01_frr_1sess", "FRR · 1 sessão", 14, 5, "start"),
    ("subj01_frr_40sess", "FRR · 40", 14, 5, "start"),
    ("subj01_ridgeonly_1sess_prior_seed42", "Ridge 1024 · 1 sessão", -14, 19, "end"),
    ("subj01_ridgeonly_1sess_4096blurry", "Ridge 4096 + blurry · 1", -14, -9, "end"),
    ("subj01_ridgeonly_40sess_prior", "Ridge 1024 · 40", 0, -16, "middle"),
    ("subj01_ridgeonly_40sess_4096blurry", "4096 + blurry · 40", 10, 24, "start"),
    ("subj01_me1_1sess", "MindEye1 · 1 sessão", 14, 5, "start"),
    ("subj01_me1_40sess", "MindEye1 · 40", -14, 22, "end"),
]

# a reexecucao (docs/REEXECUTAR.md): cada repro_<id> ao lado do <id> do benchmark. (id, rotulo, contra quem
# comparar as metricas de imagem quando o original nao as tem: as sementes 1 e 2 so tem as do release)
PREFIXO_REPRO = "repro_"
REPRODUCAO = [
    ("subj01_frr_1sess", "FRR · 1 sessão", None),
    ("subj01_frr_40sess", "FRR · 40 sessões", None),
    ("subj01_ridgeonly_1sess", "Ridge 1024 sem prior · 1", None),
    ("subj01_ridgeonly_1sess_prior", "Ridge 1024 + prior · 1", None),
    ("subj01_ridgeonly_1sess_prior_seed1", "semente 1", "subj01_ridgeonly_1sess_prior"),
    ("subj01_ridgeonly_1sess_prior_seed2", "semente 2", "subj01_ridgeonly_1sess_prior"),
    ("subj01_ridgeonly_1sess_4096blurry", "Ridge 4096 + blurry · 1", None),
    ("subj01_ridgeonly_40sess_prior", "Ridge 1024 + prior · 40", None),
    ("subj01_ridgeonly_40sess_4096blurry", "Ridge 4096 + blurry · 40", None),
]
PCT = {m["id"] for m in site.METRICAS if m["fmt"] == "pct"}
RETRIEVAL = {"FwdRetrieval", "BwdRetrieval"}


def _imagem(m):
    """As metricas de imagem de um registro (sem o retrieval, que toda tabela tem): {tipo: {metrica: valor}}."""
    out = {}
    for tipo, t in ((m or {}).get("valores") or {}).items():
        t = {k: v for k, v in (t or {}).items() if k not in RETRIEVAL and v is not None}
        if t:
            out[tipo] = t
    return out


def diferenca(a, b):
    """A maior diferenca entre as metricas de imagem de dois registros: (p.p. nas porcentagens, nas demais),
    ou None se um dos dois nao tem metricas de imagem."""
    ia, ib = _imagem(a), _imagem(b)
    pares = [(k, ia[t][k], ib[t][k]) for t in set(ia) & set(ib) for k in set(ia[t]) & set(ib[t])]
    if not pares:
        return None
    pp = max((100 * abs(x - y) for k, x, y in pares if k in PCT), default=0.0)
    resto = max((abs(x - y) for k, x, y in pares if k not in PCT), default=0.0)
    return pp, resto


def _dec(x, casas):
    return f"{x:.{casas}f}".replace(".", ",")


def _ret_como(r, o):
    """O retrieval cerebro -> imagem da reproducao medido como o do original: se o original so tem o das tabelas
    do final_evaluations (sem intervalo), o das tabelas da reproducao, e nao o do verify_retrieval.
    Devolve (valor, mesma fonte do original?)."""
    if ((o or {}).get("retrieval") or {}).get("fonte") == "final_evaluations":
        img = _imagem(r)     # so uma tabela com metricas de imagem veio do final_evaluations
        for tipo in ("refinada", "unclip"):
            t = (r.get("valores") or {}).get(tipo) or {}
            if tipo in img and t.get("BwdRetrieval") is not None:
                return t["BwdRetrieval"], True
        return _ret(r, "bwd"), False
    return _ret(r, "bwd"), True


def reproducao(por_id):
    """As linhas do slide da reproducao, so dos modelos que tem o original e a reproducao."""
    linhas = []
    for mid, rotulo, base_img in REPRODUCAO:
        o, r = por_id.get(mid), por_id.get(PREFIXO_REPRO + mid)
        if not o or not r:
            continue
        ret_r, mesma = _ret_como(r, o)
        ret = [_ret(o, "bwd"), ret_r]
        dif = diferenca(por_id.get(base_img) or o, r)
        if o.get("familia") == "frr":         # o FRR para no embedding: so o retrieval se compara
            img = "0 (idêntico)" if ret[0] == ret[1] else "só retrieval"
        elif dif is None:
            img = "reconstrução pendente" if not _imagem(r) else "—"
        elif dif == (0.0, 0.0):
            img = "0 (idêntico)"
        else:
            img = (f"{_dec(dif[0], 2)} p.p. · {_dec(dif[1], 4)}" + (" <small>contra o release</small>" if base_img else ""))
        linhas.append({"id": mid, "rotulo": rotulo, "familia": o.get("familia"), "sub": bool(base_img),
                       "pp": dif[0] if dif else None, "mesma_fonte": mesma,
                       "ret": [_pct(x, 2) for x in ret],
                       "tempo": [duracao(m.get("tempo_s")) or "—" for m in (o, r)], "img": img})
    return linhas


def _tabela_reproducao(linhas):
    fam = {"frr": "fam-frr", "mindeye2": "fam-me2", "mindeye1": "fam-me1"}
    corpo = "".join(
        f'<tr class="{fam.get(l["familia"], "")}{" sub" if l["sub"] else ""}"><th>{site._e(l["rotulo"])}</th>'
        f'<td class="num">{l["ret"][0]} → {l["ret"][1]}{"" if l["mesma_fonte"] else "*"}</td>'
        f'<td class="num">{site._e(l["tempo"][0])} → {site._e(l["tempo"][1])}</td>'
        f'<td class="num">{l["img"]}</td></tr>' for l in linhas)
    return ('<table class="tab tab-repro"><thead><tr><th>Modelo</th><th>Retrieval cér→img</th><th>Treino</th>'
            '<th>Maior diferença nas métricas de imagem</th></tr></thead>'
            f'<tbody>{corpo}</tbody></table>')


class _Valores(dict):
    """Os {{chave}} do texto: as que dependem de um modelo ({{bwd:<id>}}, {{mi:<id>:<parte>}}...) valem
    "—" quando o modelo nao existe nesta arvore; qualquer outra chave desconhecida continua sendo erro."""
    POR_MODELO = ("bwd:", "fwd:", "mi:", "total:")

    def __contains__(self, chave):
        return super().__contains__(chave) or chave.startswith(self.POR_MODELO)

    def __missing__(self, chave):
        if chave.startswith(self.POR_MODELO):
            return "—"
        raise KeyError(chave)


def _pct(x, casas=1):
    return "—" if x is None else f"{100 * x:.{casas}f}%".replace(".", ",")


def _ret(m, sentido, chave="retrieval"):
    r = (m or {}).get(chave) or {}
    return (r.get(sentido) or {}).get("media")


def _ic(m, sentido):
    return (((m or {}).get("retrieval") or {}).get(sentido) or {}).get("ic95")


def _barra_params(partes, nota):
    """A barra de 'o que treina': [(nome, n, treina)] proporcionais, e a legenda."""
    total = sum(n for _, n, _ in partes) or 1
    barra = "".join(f'<span class="{"treina" if t else ""}" style="flex:{n / total:.5f}" title="{site._e(nome)}"></span>'
                    for nome, n, t in partes)
    leg = "".join(f'<span><i class="{"treina" if t else ""}"></i>{site._e(nome)} {mi(n)}</span>' for nome, n, t in partes)
    return f'<div class="params-barra">{barra}</div><div class="params-leg">{leg}<span class="mudo">{nota}</span></div>'


def valores(modelos, planejados):
    """Os {{chave}} do texto dos slides (os {{tempo:}} e {{ridge:}} ficam com site.preenche)."""
    por_id = {m["id"]: m for m in modelos}
    agora = datetime.date.today()
    v = {"data": f"{MESES[agora.month - 1]} de {agora.year}", "n_planejados": str(len(planejados)),
         "repo": site.REPO_URL, "repo_curto": site.REPO_URL.replace("https://", "")}
    repro = [m for m in modelos if m["id"].startswith(PREFIXO_REPRO)]
    linhas = reproducao(por_id)
    v["n_repro"] = str(len(repro))
    v["n_repro_completas"] = str(sum(1 for m in repro if m.get("status") == "completo"))
    v["tabela:reproducao"] = _tabela_reproducao(linhas)
    horas = sum(m.get("tempo_s") or 0 for m in repro) / 3600
    v["repro:horas_treino"] = f"{horas:.0f} h"
    frr = [l for l in linhas if l["familia"] == "frr"]
    pps = [l["pp"] for l in linhas if l["familia"] == "mindeye2" and l["pp"] is not None]
    v["repro:resumo"] = " ".join(
        ([("O FRR sai idêntico" if all(l["ret"][0] == l["ret"][1] for l in frr) else "O FRR muda") + ";"] if frr else [])
        + ([f"o ridge-only fica a até {_dec(max(pps), 2)} p.p. do original nas métricas de imagem."]
           if pps else []))
    v["repro:nota_fonte"] = ("" if all(l["mesma_fonte"] for l in linhas) else
                             " (* a reprodução, pelo verify_retrieval; o original, pela tabela, sem intervalo)")
    pendentes = [l["rotulo"] for l in linhas if l["img"] == "reconstrução pendente"]
    v["repro:pendente"] = (f"a reconstrução da reprodução do {pendentes[0]} sessão" if len(pendentes) == 1 else
                           f"a reconstrução de {len(pendentes)} reproduções" if pendentes else "")
    for m in modelos:
        v[f"bwd:{m['id']}"] = _pct(_ret(m, "bwd"))
        v[f"fwd:{m['id']}"] = _pct(_ret(m, "fwd"))
        for parte, n in ((m.get("params") or {}).items()):
            v[f"mi:{m['id']}:{parte}"] = mi(n)
        if m.get("params"):
            total = sum(m["params"].values())
            v[f"total:{m['id']}"] = mi(total) if total < 1e9 else f"{total / 1e9:.2f} bi".replace(".", ",")

    frr1, frr40 = por_id.get("subj01_frr_1sess"), por_id.get("subj01_frr_40sess")

    def no_menor(m):
        h = (((m or {}).get("frr") or {}).get("cv") or {}).get("histograma_fracoes")
        return f"{100 * h[0] / sum(h):.0f}%" if h and sum(h) else "—"

    def cos(m):
        c = (((m or {}).get("frr") or {}).get("hubness") or {}).get("cos_previsao_media_treino")
        return "—" if c is None else f"{c:.2f}".replace(".", ",")
    v.update({"frr:g1": no_menor(frr1), "frr:g40": no_menor(frr40), "frr:cos1": cos(frr1), "frr:cos40": cos(frr40)})

    # o que treina, por familia
    p_frr = (((frr40 or {}).get("frr") or {}).get("parametros_lineares"))
    v["params:frr"] = (f"todos os {milhar(p_frr)} coeficientes, em forma fechada (sem gradiente)" if p_frr
                       else "todos os coeficientes, em forma fechada")
    p2 = (por_id.get("subj01_ridgeonly_1sess_prior") or {}).get("params") or {}
    if p2.get("ridge"):
        tot = sum(p2.values())
        parte = f"{100 * p2['ridge'] / tot:.2f}%".replace(".", ",")
        v["params:mindeye2"] = f"a ridge: {mi(p2['ridge'])} de {mi(tot)} ({parte})"
        nomes = {"ridge": "ridge (treina)", "backbone": "backbone", "diffusion_prior": "diffusion prior"}
        v["barra:mindeye2"] = _barra_params([(nomes.get(k, k), n, k == "ridge") for k, n in
                                             sorted(p2.items(), key=lambda kv: kv[0] != "ridge")],
                                            f"· {parte} dos parâmetros treinam")
    else:
        v["params:mindeye2"] = "só a ridge"
        v["barra:mindeye2"] = ""
    p1 = {k: n for k, n in ((por_id.get("subj01_me1_40sess") or {}).get("params") or {}).items() if n > 1e5}
    if p1:
        tot = sum(p1.values())
        v["params:mindeye1"] = f"todos: {mi(tot) if tot < 1e9 else str(round(tot / 1e9, 2)).replace('.', ',') + ' bi'}"
        nomes = {"voxel2clip": "MLP alto nível", "net": "diffusion prior"}
        v["barra:mindeye1"] = _barra_params([(nomes.get(k, k), n, True) for k, n in sorted(p1.items(), key=lambda kv: -kv[1])],
                                            "· tudo treina (mais o MLP de baixo nível)")
    else:
        v["params:mindeye1"] = "todos"
        v["barra:mindeye1"] = ""
    return v


def dados_pagina(modelos, galeria, semantica, exemplos):
    por_id = {m["id"]: m for m in modelos}

    def ponta(mid):
        m = por_id.get(mid)
        return {"id": mid, "bwd": _ret(m, "bwd"), "fwd": _ret(m, "fwd"), "ic_bwd": _ic(m, "bwd"), "ic_fwd": _ic(m, "fwd")}
    retrieval = [{"nome": nome, "familia": (por_id.get(a) or por_id.get(b) or {}).get("familia"), "de": ponta(a),
                  "para": ponta(b)} for nome, a, b in PARES if a in por_id or b in por_id]

    gama = None
    cv = [(((por_id.get(i) or {}).get("frr") or {}).get("cv") or {}) for i in ("subj01_frr_1sess", "subj01_frr_40sess")]
    if all(c.get("histograma_fracoes") and c.get("fracs") for c in cv):
        def agrupa(c):
            h, tot = c["histograma_fracoes"], sum(c["histograma_fracoes"])
            return [100 * x / tot for x in h[:4]] + [100 * sum(h[4:]) / tot]
        f = cv[0]["fracs"]
        gama = {"rotulos": [f"{x:.2f}".replace(".", ",") for x in f[:4]] + [f"≥ {f[4]:.2f}".replace(".", ",")],
                "um": agrupa(cv[0]), "quarenta": agrupa(cv[1])}

    hub = None
    f1, f40 = por_id.get("subj01_frr_1sess"), por_id.get("subj01_frr_40sess")
    if f1 and f40 and f1.get("retrieval_centrado") and f40.get("retrieval_centrado"):
        hub = {k: {"bruto": 100 * _ret(m, "bwd"), "centrado": 100 * _ret(m, "bwd", "retrieval_centrado")}
               for k, m in (("um", f1), ("quarenta", f40))}

    custo = []
    for mid, rot, dx, dy, ancora in CUSTO:
        m = por_id.get(mid)
        if not (m and m.get("tempo_s")) and (por_id.get(PREFIXO_REPRO + mid) or {}).get("tempo_s"):
            m = por_id[PREFIXO_REPRO + mid]     # o original nao tem tempo (checkpoint do release): o da reproducao
        if m and m.get("tempo_s") and _ret(m, "bwd") is not None:
            custo.append({"id": m["id"], "nome": m.get("rotulo") or mid, "rotulo": rot, "familia": m["familia"],
                          "sessoes": m.get("sessoes"), "tempo_s": m["tempo_s"], "bwd": _ret(m, "bwd"),
                          "dx": dx, "dy": dy, "ancora": ancora})

    colunas = [{"id": i, "tipo": t, "rotulo": r, "sub": s, "familia": (por_id.get(i) or {}).get("familia")}
               for i, t, r, s in COLUNAS_GALERIA]
    return {"galeria": galeria, "exemplos": exemplos, "semantica": semantica, "nomes": NOMES_PT,
            "ordem_classes": ORDEM_CLASSES, "classes5": [{"nome": n, "sups": s} for n, s in CLASSES_5],
            "retrieval": retrieval, "gama": gama, "hub": hub, "custo": custo, "galeria_colunas": colunas}


COMANDOS = """<nav class="comandos" aria-label="Apresentação">
  <button type="button" data-cmd="ant" aria-label="Slide anterior">←</button>
  <span class="contador num" aria-live="polite"></span>
  <button type="button" data-cmd="prox" aria-label="Próximo slide">→</button>
  <button type="button" data-cmd="sumario" aria-label="Sumário">Sumário</button>
  <button type="button" data-cmd="modo">Documento</button>
  <button type="button" data-cmd="tema">Tema escuro</button>
  <button type="button" data-cmd="cheia" aria-label="Tela cheia">Tela cheia</button>
  <a href="index.html">Site</a>
</nav>
<div class="sumario" hidden></div>
<div class="progresso" aria-hidden="true"></div>"""


def monta(modelos, planejados, galeria, semantica, exemplos):
    """Devolve (cabeca, corpo) da apresentacao, como site.monta."""
    todos = modelos
    modelos = [m for m in modelos if m["id"] not in site.OCULTOS]
    texto = (AQUI / "conteudo" / "apresentacao.html").read_text(encoding="utf-8")
    slides = site.preenche(texto, _Valores(valores(todos, planejados), n_modelos=str(len(modelos))), todos)
    dados = json.dumps(dados_pagina(todos, galeria, semantica, exemplos), ensure_ascii=False,
                       separators=(",", ":")).replace("</", "<\\/")
    estilo = (AQUI / "assets" / "apresentacao.css").read_text(encoding="utf-8")
    script = ((AQUI / "assets" / "simulador.js").read_text(encoding="utf-8") + "\n" +
              (AQUI / "assets" / "apresentacao.js").read_text(encoding="utf-8"))
    cabeca = f"<title>benchmark-modelos · apresentação</title>\n{site.FONTES}\n<style>{estilo}</style>"
    corpo = f"""<main class="palco"><div class="deck">
{slides}
</div></main>
{COMANDOS}
<script type="application/json" id="dados">{dados}</script>
<script>{script}</script>
"""
    return cabeca, corpo
