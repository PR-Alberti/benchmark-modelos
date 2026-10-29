"""FRR multi-sujeito: uma ridge por sujeito para um espaco compartilhado de k dimensoes e uma
leitura compartilhada desse espaco para o embedding CLIP, tudo linear e em forma fechada.

    previsao do sujeito s:   y = ybar_s + (x - xbar_s) R_s W        R_s: (voxels_s, k)   W: (k, 425.984)

Pre-treino (nos sujeitos 2-8, ou no proprio sujeito, o controle de posto reduzido): minimos quadrados
alternados sobre  sum_s w_s |X_s R_s W - Y_s|^2,  com peso igual por sujeito (w_s = 1/n_s) ou por
exibicao (w_s = 1):

  passo R  com W fixo, cada R_s e uma ridge (FRR) dos voxels de s para o alvo reduzido T_s = Y_s Q',
           independente por sujeito;
  passo W  com os R_s fixos, W e um FRR das entradas Z_s = X_s R_s, empilhadas com os pesos, para Y.
           So entram somas (Z'Z e Z'Y), acumuladas sujeito a sujeito: o resultado e o de um ajuste
           conjunto, com um sujeito por vez na memoria;
  escala   R_s W nao muda com R_s A e A^-1 W. Depois de cada passo W, W = L Q com Q de linhas
           ortonormais (L = E sqrt(e), de W W' = E e E'), Q fica como a leitura e L passa para os R_s.
           Com Q ortonormal, |Z Q - Y|^2 = |Z - Y Q'|^2 + constante: o passo R e uma ridge comum.

As fracoes (Rokem & Kay) sao escolhidas numa validacao separada por imagem (uma fracao por dimensao do
alvo, como no FRR), e o laco para quando a perda de validacao deixa de cair; fica a melhor iteracao.
O inicio e a base das k componentes principais dos alvos (SVD aleatorizado), sem dado de cerebro.

O ajuste fino num sujeito novo (o subj01) e o FRR comum (frr.FracRidge) dos voxels para T = Y Q', com a
leitura Q congelada: alvo_reduzido() e expande() fazem a ida e a volta.

Os sujeitos so precisam de: nome, carrega() -> X (n, p) das exibicoes, linhas (n,) -> linha de Y da imagem
de cada exibicao, e Y (m, D), aberto como memmap. A base espectral de cada sujeito (o que depende so dos
voxels) e calculada uma vez e guardada; dela saem os escores que os dois passos usam, e os voxels nao
voltam a ser lidos.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from . import frr


@dataclass
class Sujeito:
    nome: str
    carrega: Callable          # () -> X (n, p) das exibicoes, na ordem de `linhas`
    linhas: np.ndarray         # (n,) linha de Y da imagem de cada exibicao
    Y: object                  # (m, D) alvos por imagem (np.ndarray ou memmap)


def _log(*a):
    print("[frr_ms]", *a, flush=True)


def mascara_validacao(linhas, fracao, seed):
    """Exibicoes de validacao: todas as de uma fracao das imagens (uma imagem nunca fica dos dois lados)."""
    imgs = np.unique(linhas)
    n_val = int(round(fracao * len(imgs))) if fracao > 0 else 0
    n_val = min(max(n_val, 1 if fracao > 0 else 0), len(imgs) - 1)
    escolhidas = np.random.RandomState(seed).permutation(imgs)[:n_val]
    return np.isin(linhas, escolhidas)


def base_espectral(X, linhas, val, rtol=1e-10, guarda_V=False):
    """O que os dois passos precisam dos voxels de um sujeito, na base espectral das exibicoes de treino.

    S_agg (m_tr, r): soma, por imagem de treino, dos escores (x - xbar) V das exibicoes dela;
    S_val (n_val, r): os escores de cada exibicao de validacao. Como S'S = diag(lam) no treino, a
    Gram de Z = S B e B' diag(lam) B, sem passar pelas exibicoes.
    """
    X = torch.as_tensor(X)
    tr = ~val
    Xt = X[torch.from_numpy(tr)].to(torch.float64)
    media = Xt.mean(0)
    Xt -= media
    V, lam = frr.spectral_basis(Xt, rtol)
    img_tr, inv = np.unique(linhas[tr], return_inverse=True)
    agg = torch.zeros(len(img_tr), Xt.shape[1], dtype=torch.float64)
    agg.index_add_(0, torch.from_numpy(inv), Xt)
    del Xt
    S_agg = (agg @ V).float()
    del agg
    img_val, inv_val = np.unique(linhas[val], return_inverse=True)
    S_val = ((X[torch.from_numpy(val)].to(torch.float64) - media) @ V).float()
    extra = {"V": V.float(), "media": media.float()} if guarda_V else {}       # so para os testes
    return {**extra, "lam": lam, "S_agg": S_agg, "img_tr": img_tr, "cont_tr": np.bincount(inv), "n_tr": int(tr.sum()),
            "S_val": S_val, "img_val": img_val, "inv_val": inv_val, "cont_val": np.bincount(inv_val),
            "n_val": int(val.sum()), "voxels": int(X.shape[1]), "posto": int(len(lam))}


class _Blocos:
    """Leitura dos alvos de um sujeito por blocos de colunas, centrados pela media de treino do bloco."""

    def __init__(self, base, Y, device):
        self.b, self.Y, self.device = base, Y, device

    def le(self, c0, c1):
        b = self.b
        ytr = torch.from_numpy(np.ascontiguousarray(self.Y[b["img_tr"], c0:c1])).to(self.device).float()
        cont = torch.from_numpy(b["cont_tr"]).to(self.device).float()
        ybar = (cont[:, None] * ytr).sum(0) / b["n_tr"]
        yval = torch.from_numpy(np.ascontiguousarray(self.Y[b["img_val"], c0:c1])).to(self.device).float() \
            if len(b["img_val"]) else ytr[:0]
        return ytr - ybar, yval - ybar


def _pesos(bases, modo):
    """(w_treino, w_validacao) por sujeito: peso igual por sujeito (1/n) ou por exibicao (1)."""
    if modo == "sujeito":
        return [1.0 / b["n_tr"] for b in bases], [1.0 / max(b["n_val"], 1) for b in bases]
    if modo == "exibicao":
        return [1.0] * len(bases), [1.0] * len(bases)
    raise ValueError(f"pesos {modo!r}: use sujeito ou exibicao")


def pca_inicial(bases, Ys, pesos, k, D, chunk, device, seed, extra=64, potencia=1):
    """Base (k_ef, D), de linhas ortonormais, das k componentes principais dos alvos de treino
    centrados por sujeito e com os pesos: SVD aleatorizado (Halko et al.), 2 + 2*potencia passadas."""
    blocos = [_Blocos(b, Y, device) for b, Y in zip(bases, Ys)]
    sw = [torch.from_numpy(np.sqrt(w * b["cont_tr"])).to(device).float() for b, w in zip(bases, pesos)]
    N = sum(len(b["img_tr"]) for b in bases)
    l = min(k + extra, N)
    k_ef = min(k, l)
    faixas = [(c, min(c + chunk, D)) for c in range(0, D, chunk)]

    def M(s, c0, c1):
        return sw[s][:, None] * blocos[s].le(c0, c1)[0]

    A = [torch.zeros(len(b["img_tr"]), l, device=device) for b in bases]
    for i, (c0, c1) in enumerate(faixas):
        g = torch.Generator().manual_seed(seed * 7919 + i)
        om = torch.randn(c1 - c0, l, generator=g).to(device)
        for s in range(len(bases)):
            A[s] += M(s, c0, c1) @ om
    for _ in range(potencia):
        q = torch.linalg.qr(torch.cat(A))[0]
        partes = torch.split(q, [len(a) for a in A])
        Bt = []
        for c0, c1 in faixas:
            Bt.append(sum(M(s, c0, c1).T @ partes[s] for s in range(len(bases))).cpu())
        A = [torch.zeros_like(a) for a in A]
        for (c0, c1), bt in zip(faixas, Bt):
            bt = bt.to(device)
            for s in range(len(bases)):
                A[s] += M(s, c0, c1) @ bt
        del Bt
    q = torch.linalg.qr(torch.cat(A))[0]
    partes = torch.split(q, [len(a) for a in A])
    Bm, BB = [], torch.zeros(l, l, dtype=torch.float64, device=device)
    for c0, c1 in faixas:
        bm = sum(partes[s].T @ M(s, c0, c1) for s in range(len(bases)))
        BB += (bm @ bm.T).double()
        Bm.append(bm.cpu())
    e, U = torch.linalg.eigh(BB)
    e, U = e.flip(0), U.flip(1)
    k_ef = min(k_ef, int((e > 1e-10 * e[0]).sum()))       # com poucas imagens, a centragem zera direcoes
    e, U = e[:k_ef], U[:, :k_ef]
    proj = (U / e.sqrt()).T.float()                                     # (k_ef, l)
    return np.concatenate([(proj @ bm.to(device)).cpu().numpy() for bm in Bm], axis=1), k_ef


def alvo_reduzido(Y, img, cont, Q, chunk, device):
    """T = (Y[img] - ybar) Q' (m, k_ef) e ybar (D,), com a media ponderada pelas exibicoes (cont)."""
    D = Y.shape[1]
    cont_t = torch.from_numpy(np.asarray(cont)).to(device).float()
    T = torch.zeros(len(img), Q.shape[0], device=device)
    ybar = torch.zeros(D)
    for c0 in range(0, D, chunk):
        c1 = min(c0 + chunk, D)
        y = torch.from_numpy(np.ascontiguousarray(Y[img, c0:c1])).to(device).float()
        yb = (cont_t[:, None] * y).sum(0) / cont_t.sum()
        T += (y - yb) @ torch.from_numpy(np.ascontiguousarray(Q[:, c0:c1])).to(device).T
        ybar[c0:c1] = yb.cpu()
    return T.cpu().numpy(), ybar


def expande(t, ybar, Q, chunk, device, saida=torch.float16):
    """Do espaco compartilhado de volta ao embedding: ybar + t Q, (n, D) na CPU."""
    t = torch.as_tensor(t).to(device).float()
    D = Q.shape[1]
    out = torch.empty(t.shape[0], D, dtype=saida)
    for c0 in range(0, D, chunk):
        c1 = min(c0 + chunk, D)
        q = torch.from_numpy(np.ascontiguousarray(Q[:, c0:c1])).to(device)
        out[:, c0:c1] = (ybar[c0:c1].to(device) + t @ q).to(saida).cpu()
    return out


def passo_R(base, T_tr, T_val, fracs, device):
    """Ridge (FRR) da base espectral do sujeito para o alvo reduzido; fracao por dimensao na validacao.

    T_tr (m_tr, k) por imagem de treino, T_val (m_val, k) por imagem de validacao, ja centrados.
    Devolve B (r, k), com R_s = V B, e o R2 de validacao no espaco reduzido.
    """
    lam = base["lam"].to(device)
    m = base["S_agg"].to(device).T @ T_tr                                # (r, k) = V' X' T
    grade = frr.alpha_grid(base["lam"].numpy())
    alphas = frr.fraction_alphas(m, lam, grade, fracs)                   # (F, k)
    lam32 = lam.float()[:, None]
    if base["n_val"] == 0:                                               # sem validacao: OLS
        return m / (lam32 + alphas[-1][None, :]), float("nan")
    S_val = base["S_val"].to(device)
    Tv = T_val[torch.from_numpy(base["inv_val"]).to(device)]
    sse = torch.stack([((S_val @ (m / (lam32 + a[None, :])) - Tv) ** 2).sum(0) for a in alphas])
    melhor = sse.argmin(0)
    B = m / (lam32 + alphas.gather(0, melhor[None])[0][None, :])
    return B, 1.0 - float(sse.gather(0, melhor[None]).sum()) / float((Tv * Tv).sum())


def passo_W(bases, Ys, Bs, w, wv, fracs, D, chunk, device, saida, rtol=1e-10):
    """FRR das entradas Z_s = S_s B_s (empilhadas, com pesos) para os alvos, por blocos de colunas.

    A fracao de cada dimensao do alvo sai do erro de validacao, calculado pela forma quadratica
    b' (Zv'Zv) b - 2 b' Zv'y + |y|^2, sem materializar as previsoes. Grava W (k, D) em `saida`
    (memmap) e devolve W W' (k, k), a perda relativa de validacao e o histograma das fracoes.
    """
    k = Bs[0].shape[1]
    G = torch.zeros(k, k, dtype=torch.float64, device=device)
    Gv = torch.zeros(k, k, dtype=torch.float64, device=device)
    Zagg, Zvagg = [], []
    for b, B, ws, wvs in zip(bases, Bs, w, wv):
        B = B.to(device)
        G += ws * (B.T.double() * b["lam"].to(device)[None, :]) @ B.double()
        Zagg.append(b["S_agg"].to(device) @ B)
        Zv = b["S_val"].to(device) @ B
        Gv += wvs * (Zv.T @ Zv).double()
        agg = torch.zeros(len(b["img_val"]), k, device=device)
        agg.index_add_(0, torch.from_numpy(b["inv_val"]).to(device), Zv)
        Zvagg.append(agg)
    mu, Qg = torch.linalg.eigh(G)
    mu, Qg = mu.flip(0), Qg.flip(1)
    manter = mu > rtol * mu[0]
    mu, Qg = mu[manter], Qg[:, manter].float()
    Gv_q = (Qg.T @ Gv.float() @ Qg)
    grade = frr.alpha_grid(mu.cpu().numpy())
    blocos = [_Blocos(b, Y, device) for b, Y in zip(bases, Ys)]
    WW = torch.zeros(k, k, dtype=torch.float64, device=device)
    sse_tot, sst_tot = 0.0, 0.0
    hist = torch.zeros(len(fracs), dtype=torch.long)
    cont_val = [torch.from_numpy(b["cont_val"]).to(device).float() for b in bases]
    for c0 in range(0, D, chunk):
        c1 = min(c0 + chunk, D)
        C = torch.zeros(k, c1 - c0, device=device)
        Cv = torch.zeros(k, c1 - c0, device=device)
        yy = torch.zeros(c1 - c0, dtype=torch.float64, device=device)
        for s, bl in enumerate(blocos):
            ytr, yval = bl.le(c0, c1)
            C += w[s] * (Zagg[s].T @ ytr)
            if len(yval):
                Cv += wv[s] * (Zvagg[s].T @ yval)
                yy += wv[s] * (cont_val[s][:, None] * yval * yval).sum(0).double()
        m = Qg.T @ C
        cvq = Qg.T @ Cv
        alphas = frr.fraction_alphas(m, mu, grade, fracs)                 # (F, dc)
        mu32 = mu.float()[:, None]
        sse = []
        for a in alphas:
            bt = m / (mu32 + a[None, :])
            sse.append(((bt * (Gv_q @ bt)).sum(0) - 2 * (bt * cvq).sum(0)).double() + yy)
        sse = torch.stack(sse)
        melhor = sse.argmin(0)
        beta = m / (mu32 + alphas.gather(0, melhor[None])[0][None, :])
        Wb = Qg @ beta                                                     # (k, dc)
        saida[:, c0:c1] = Wb.cpu().numpy()
        WW += (Wb @ Wb.T).double()
        sse_tot += float(sse.gather(0, melhor[None]).sum())
        sst_tot += float(yy.sum())
        hist += torch.bincount(melhor.cpu(), minlength=len(fracs))
    perda = sse_tot / sst_tot if sst_tot > 0 else float("nan")
    return WW, perda, hist.tolist()


def escala(WW, rtol=1e-10):
    """W = L Q com Q de linhas ortonormais: de W W' = E e E', L = E sqrt(e) (k, k_ef) e a projecao
    P = e^-1/2 E' (k_ef, k) que leva W em Q. Os B_s com a escala absorvida sao B_s L: S B_s W = S (B_s L) Q."""
    e, E = torch.linalg.eigh(WW)
    e, E = e.flip(0), E.flip(1)
    manter = e > rtol * e[0]
    e, E = e[manter], E[:, manter]
    return (E * e.sqrt()[None, :]).float(), (E / e.sqrt()[None, :]).T.float()


def projeta(W, P, saida, chunk, device):
    """saida (k_ef, D) = P W, por blocos de colunas."""
    for c0 in range(0, W.shape[1], chunk):
        c1 = min(c0 + chunk, W.shape[1])
        saida[:, c0:c1] = (P.to(device) @ torch.from_numpy(np.ascontiguousarray(W[:, c0:c1])).to(device)).cpu().numpy()


def _hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


class FRRMultiSujeito:
    """Pre-treino por minimos quadrados alternados; ao fim, `Q` (k_ef, D) e a leitura compartilhada."""

    def __init__(self, k=1024, fracs=None, pesos="sujeito", iteracoes=10, tolerancia=1e-3, validacao=0.1,
                 seed=42, chunk=8192, device=None, rtol=1e-10):
        self.k, self.pesos, self.iteracoes, self.tolerancia = k, pesos, iteracoes, tolerancia
        self.fracs = np.asarray(frr.FRACS_DOERIG if fracs is None else fracs, dtype=float)
        self.validacao, self.seed, self.chunk, self.rtol = validacao, seed, chunk, rtol
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def config(self):
        return {"k": self.k, "fracs": [float(f) for f in self.fracs], "pesos": self.pesos,
                "iteracoes": self.iteracoes, "tolerancia": self.tolerancia, "validacao": self.validacao,
                "seed": self.seed, "chunk": self.chunk, "rtol": self.rtol}

    def _base(self, suj, workdir):
        """Base espectral do sujeito, calculada uma vez (e guardada em workdir, se houver)."""
        val = mascara_validacao(suj.linhas, self.validacao, self.seed)
        meta = {"sujeito": suj.nome, "validacao": self.validacao, "seed": self.seed, "rtol": self.rtol,
                "linhas": hashlib.md5(np.ascontiguousarray(suj.linhas).tobytes()).hexdigest()}
        arq = os.path.join(workdir, f"base_{suj.nome}.pt") if workdir else None
        if arq and os.path.exists(arq):
            b = torch.load(arq)
            if b["meta"] != meta:
                raise ValueError(f"{arq} foi gerada com outros dados ou outra configuracao; apague para refazer")
            _log(f"{suj.nome}: base espectral de {arq}")
            return b
        t0 = time.time()
        X = suj.carrega()
        b = base_espectral(X, suj.linhas, val, self.rtol)
        del X
        b["meta"] = meta
        _log(f"{suj.nome}: base espectral em {time.time() - t0:.0f} s (posto {b['posto']} de {b['voxels']} "
             f"voxels; {b['n_tr']} exibicoes de treino, {b['n_val']} de validacao)")
        if arq:
            torch.save(b, arq + ".tmp")
            os.replace(arq + ".tmp", arq)
        return b

    def ajusta(self, sujeitos, workdir):
        """Pre-treina (ou retoma de workdir) e deixa self.Q (memmap) e self.historico."""
        os.makedirs(workdir, exist_ok=True)
        t0 = time.time()
        D = sujeitos[0].Y.shape[1]
        bases = [self._base(s, workdir) for s in sujeitos]
        Ys = [s.Y for s in sujeitos]
        w, wv = _pesos(bases, self.pesos)
        assinatura = {"config": self.config(), "bases": [b["meta"] for b in bases], "D": D}
        arq_estado = os.path.join(workdir, "estado.json")
        estado = None
        if os.path.exists(arq_estado):
            with open(arq_estado) as f:
                estado = json.load(f)
            if estado["assinatura"] != json.loads(json.dumps(assinatura)):
                raise ValueError(f"{workdir} e de outro pre-treino; apague o diretorio para refazer")
        arq_Q = os.path.join(workdir, "Q.npy")
        if estado is None:
            _log(f"inicio: PCA dos alvos ({sum(len(b['img_tr']) for b in bases)} imagens de treino, k={self.k})")
            Q0, k_ef = pca_inicial(bases, Ys, w, self.k, D, self.chunk, self.device, self.seed)
            np.save(arq_Q + ".tmp.npy", Q0)
            os.replace(arq_Q + ".tmp.npy", arq_Q)
            del Q0
            estado = {"assinatura": assinatura, "iteracao": 0, "historico": [], "melhor": None,
                      "concluido": False, "segundos": time.time() - t0}
            self._grava(arq_estado, estado)
        while not estado["concluido"]:
            t1 = time.time()
            it = estado["iteracao"] + 1
            Q = np.load(arq_Q, mmap_mode="r")
            Ts, Tvs = self._reduz(bases, Ys, Q)
            Bs, r2R = [], []
            for b, T, Tv in zip(bases, Ts, Tvs):
                B, r2 = passo_R(b, T, Tv, self.fracs, self.device)
                Bs.append(B.cpu()); r2R.append(r2)
            del Ts, Tvs
            Wf = np.lib.format.open_memmap(os.path.join(workdir, "W_tmp.npy"), mode="w+", dtype=np.float32,
                                           shape=(Bs[0].shape[1], D))
            WW, perda, hist = passo_W(bases, Ys, Bs, w, wv, self.fracs, D, self.chunk, self.device, Wf, self.rtol)
            Wf.flush()
            _, P = escala(WW, self.rtol)
            k_ef = P.shape[0]
            Qn = np.lib.format.open_memmap(os.path.join(workdir, "Q_tmp.npy"), mode="w+", dtype=np.float32,
                                           shape=(k_ef, D))
            projeta(Wf, P, Qn, self.chunk, self.device)
            Qn.flush()
            del Wf, Qn, Q
            os.replace(os.path.join(workdir, "Q_tmp.npy"), arq_Q)
            os.remove(os.path.join(workdir, "W_tmp.npy"))
            registro = {"iteracao": it, "perda_validacao": perda, "r2_validacao": 1.0 - perda,
                        "r2_passo_R": r2R, "k_efetivo": k_ef, "histograma_fracoes_W": hist,
                        "segundos": time.time() - t1}
            estado["historico"].append(registro)
            _log(f"iteracao {it}: R2 de validacao {1 - perda:.5f} (k_ef {k_ef}, {registro['segundos']:.0f} s)")
            melhor = estado["melhor"]
            if melhor is None or perda < melhor["perda_validacao"]:
                np.save(os.path.join(workdir, "Q_melhor.tmp.npy"), np.load(arq_Q, mmap_mode="r"))
                os.replace(os.path.join(workdir, "Q_melhor.tmp.npy"), os.path.join(workdir, "Q_melhor.npy"))
                ganho = None if melhor is None else (melhor["perda_validacao"] - perda) / melhor["perda_validacao"]
                estado["melhor"] = registro
            else:
                ganho = -1.0
            estado["iteracao"] = it
            estado["segundos"] += time.time() - t1
            if it >= self.iteracoes or (ganho is not None and ganho < self.tolerancia):
                estado["concluido"] = True
                motivo = "maximo de iteracoes" if it >= self.iteracoes else "a validacao parou de melhorar"
                _log(f"fim: {motivo}; melhor iteracao {estado['melhor']['iteracao']} "
                     f"(R2 de validacao {estado['melhor']['r2_validacao']:.5f})")
            self._grava(arq_estado, estado)
        self.Q = np.load(os.path.join(workdir, "Q_melhor.npy"), mmap_mode="r")
        self.historico, self.melhor, self.segundos = estado["historico"], estado["melhor"], estado["segundos"]
        self.sujeitos = [{"nome": s.nome, "voxels": b["voxels"], "posto": b["posto"], "exibicoes_treino": b["n_tr"],
                          "exibicoes_validacao": b["n_val"], "imagens_treino": len(b["img_tr"]),
                          "imagens_validacao": len(b["img_val"])} for s, b in zip(sujeitos, bases)]
        return self

    def _reduz(self, bases, Ys, Q):
        """Alvos reduzidos de cada sujeito, T = (Y - ybar_treino) Q', das imagens de treino e de validacao,
        numa passada pelos alvos (blocos de colunas por fora, para ler cada bloco de Q uma vez)."""
        k = Q.shape[0]
        blocos = [_Blocos(b, Y, self.device) for b, Y in zip(bases, Ys)]
        Ts = [torch.zeros(len(b["img_tr"]), k, device=self.device) for b in bases]
        Tvs = [torch.zeros(len(b["img_val"]), k, device=self.device) for b in bases]
        for c0 in range(0, Q.shape[1], self.chunk):
            c1 = min(c0 + self.chunk, Q.shape[1])
            q = torch.from_numpy(np.ascontiguousarray(Q[:, c0:c1])).to(self.device)
            for s, bl in enumerate(blocos):
                ytr, yval = bl.le(c0, c1)
                Ts[s] += ytr @ q.T
                Tvs[s] += yval @ q.T
        return Ts, Tvs

    @staticmethod
    def _grava(arq, estado):
        with open(arq + ".tmp", "w") as f:
            json.dump(estado, f, indent=1)
        os.replace(arq + ".tmp", arq)

    def resumo(self):
        return {"config": self.config(), "sujeitos": self.sujeitos, "historico": self.historico,
                "melhor_iteracao": self.melhor["iteracao"], "r2_validacao": self.melhor["r2_validacao"],
                "k_efetivo": int(self.Q.shape[0]), "segundos": self.segundos}


def diretorio_pretreino(raiz, config, sujeitos_meta):
    """Onde fica um pre-treino: o dos outros sujeitos e compartilhado entre os modelos que o usam
    (o mesmo para 1 e 40 sessoes do subj01); o do proprio sujeito fica no modelo."""
    return os.path.join(raiz, "_pretreino_frr_ms", _hash({"config": config, "sujeitos": sujeitos_meta}))
