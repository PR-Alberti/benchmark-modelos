"""Fractional ridge regression (Rokem & Kay, 2020) para alvos de centenas de milhares de dimensoes.

Na ridge comum, beta(alpha) = (X'X + alpha I)^-1 X'y e o alpha so tem sentido
para um dado X e um dado y. A FRR troca o alpha pela fracao

    gamma = |beta(alpha)| / |beta_OLS|        (0 = tudo encolhido, 1 = sem regularizacao)

que significa a mesma coisa para qualquer alvo e qualquer escala de X, e acha o
alpha de cada (fracao, alvo) na decomposicao espectral X = U S V':

    beta(alpha) = V diag(s / (s^2 + alpha)) U'y     |beta(alpha)|^2 = sum_k m_k^2 / (lam_k + alpha)^2

com lam = s^2 e m = V'X'y. Como |beta| so cai quando alpha sobe, a curva sobre uma
grade de alphas se inverte por interpolacao: o mesmo esquema do pacote `fracridge`
de referencia (grade logaritmica, interpolacao linear em log(1 + alpha)), que os
testes usam como controle.

Esta implementacao existe porque o alvo do MindEye2 e o embedding CLIP achatado
(256 x 1664 = 425.984 dimensoes): a matriz de coeficientes teria 6,7 bilhoes de
entradas (27 GB em fp32), entao nada pode ser materializado por inteiro. Tudo
corre em blocos de colunas do alvo, e a escolha da fracao de cada dimensao por
validacao cruzada (como em Doerig et al., 2025) sai de graca da mesma
decomposicao: so muda o alpha, nunca a decomposicao.

Uso:

    modelo = FracRidge(n_folds=5).fit(X, Y, groups)
    previsto = modelo.predict(X_novo)

    X       (n, p)  uma linha por exibicao (voxels)
    Y       (m, d)  uma linha por alvo unico (por imagem), aceita memmap; so e lido por blocos
    groups  (n,)    em que linha de Y esta o alvo de cada exibicao (varias exibicoes da mesma
                    imagem repetem o alvo sem copiar Y)
"""
import hashlib
import os
import time

import numpy as np
import torch

# Limites e passo da grade de alphas, os mesmos do fracridge de referencia: de
# 1e-2 * s_min^2 a 1e4 * s_max^2, em passos de 0,2 decada, mais o alpha = 0 (OLS).
BIG_BIAS = 10e3
SMALL_BIAS = 10e-3
BIAS_STEP = 0.2

# 20 fracoes de 0,05 a 1, como em Doerig et al. (2025)
FRACS_DOERIG = np.linspace(0.05, 1.0, 20)

# Com os 15.724 voxels do nsdgeneral do subj01, o otimo do ERRO QUADRATICO na validacao cruzada cai
# abaixo da grade de Doerig: com 1 sessao 86% das dimensoes do alvo escolhem a menor fracao (0,05) e
# com 40 sessoes, todas. Fracoes pequenas sao normais: a fracao e relativa a norma do OLS, que com
# poucos dados e dominada pelo ruido amplificado nas direcoes de menor variancia. Esta grade
# acrescenta 10 fracoes menores, ate o otimo ficar no interior -- e e so uma analise de
# sensibilidade, nao um melhor ajuste: com 40 sessoes o R2 da CV dobra (1,6% -> 3,2%) mas o
# retrieval no teste cai (96,4% -> 92,4%), porque encolher mais aproxima as previsoes da media e
# o ranking piora. O erro quadratico nao e o criterio do retrieval.
FRACS_ESTENDIDA = np.concatenate([[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.04],
                                  FRACS_DOERIG])


def spectral_basis(Xc, rtol=1e-10):
    """Base espectral de X centrada: devolve V (p, r) e lam (r,), em float64.

    lam sao os autovalores de X'X (s^2), em ordem decrescente. Diagonaliza a
    menor das duas matrizes de Gram: X X' (n x n) quando ha menos exibicoes que
    voxels, X'X (p x p) no contrario. Direcoes com lam <= rtol * lam_max sao
    descartadas: sao o posto que falta (n < p, ou a media subtraida), e o OLS
    delas nao esta definido.
    """
    n, p = Xc.shape
    if n <= p:
        lam, U = torch.linalg.eigh(Xc @ Xc.T)
        lam, U = lam.flip(0), U.flip(1)
        keep = lam > rtol * lam[0]
        lam, U = lam[keep], U[:, keep]
        V = Xc.T @ (U / lam.sqrt())
    else:
        lam, V = torch.linalg.eigh(Xc.T @ Xc)
        lam, V = lam.flip(0), V.flip(1)
        keep = lam > rtol * lam[0]
        lam, V = lam[keep], V[:, keep]
    return V, lam


def alpha_grid(lam, bias_step=BIAS_STEP):
    """Grade de alphas candidatos (G,), com alpha = 0 na frente. lam em ordem decrescente."""
    hi = np.ceil(np.log10(BIG_BIAS * float(lam[0])))
    lo = np.floor(np.log10(SMALL_BIAS * float(lam[-1])))
    return np.concatenate([[0.0], 10.0 ** np.arange(lo, hi, bias_step)])


def fraction_alphas(m, lam, grid, fracs, return_unreachable=False):
    """alpha que da cada fracao em cada alvo: (F, dc).

    m (r, dc) sao os momentos V'X'y do bloco de alvos; lam (r,); grid (G,) com
    grid[0] = 0; fracs (F,) em ordem crescente. Fracao acima da maior alcancavel
    na grade fica em alpha = 0, abaixo da menor, no maior alpha (como np.interp).
    Com return_unreachable, devolve tambem quantos pares (fracao, alvo) cairam
    abaixo da menor fracao que a grade alcanca: ficam todos no mesmo alpha.
    """
    dt, dev = m.dtype, m.device
    lam = lam.to(dev, torch.float64)
    grid_t = torch.as_tensor(grid, dtype=torch.float64, device=dev)
    # |beta(alpha)|^2 para a grade inteira de uma vez: uma multiplicacao de matrizes
    w = (1.0 / (lam[:, None] + grid_t[None, :]) ** 2).to(dt)            # (r, G)
    norm2 = (m * m).T @ w                                                # (dc, G)
    norm2_ols = norm2[:, :1]
    ok = norm2_ols > 0                      # alvo sem sinal nenhum: qualquer alpha serve
    ratio = torch.sqrt(norm2 / torch.where(ok, norm2_ols, torch.ones_like(norm2_ols)))
    # ratio decresce com alpha; invertido fica crescente, como o searchsorted exige
    # (cummax tira o ruido numerico nas pontas, onde a curva e plana)
    r_asc = torch.cummax(ratio.flip(1), dim=1).values.contiguous()
    log_alpha = torch.log1p(grid_t.to(dt)).flip(0)                       # (G,)

    f = torch.as_tensor(np.asarray(fracs), dtype=dt, device=dev)
    q = f[None, :].expand(r_asc.shape[0], -1).contiguous()               # (dc, F)
    G = r_asc.shape[1]
    hi = torch.searchsorted(r_asc, q).clamp(0, G - 1)
    lo = (hi - 1).clamp(0, G - 1)
    x0, x1 = r_asc.gather(1, lo), r_asc.gather(1, hi)
    y0, y1 = log_alpha[lo], log_alpha[hi]
    span = x1 - x0
    t = torch.where(span > 0, (q - x0) / torch.where(span > 0, span, torch.ones_like(span)),
                    torch.zeros_like(span)).clamp(0, 1)
    alpha = torch.expm1(y0 + t * (y1 - y0))                              # (dc, F)
    alpha = torch.where(ok, alpha, torch.zeros_like(alpha))
    if return_unreachable:
        n_baixo = int(((q < r_asc[:, :1]) & ok).sum())
        return alpha.T.contiguous(), n_baixo
    return alpha.T.contiguous()                                          # (F, dc)


def impressao_dos_dados(X, groups):
    """Identifica os dados de uma corrida: o agrupamento das exibicoes e uma amostra das linhas de X.

    Vai junto de cada dobra da CV guardada, para que uma retomada com OUTROS dados do mesmo formato
    (por exemplo, outro sorteio de imagens com o mesmo numero de exibicoes) nao reaproveite dobras
    que nao sao dela.
    """
    passo = max(1, len(X) // 64)
    return {"groups": hashlib.md5(np.ascontiguousarray(np.asarray(groups)).tobytes()).hexdigest(),
            "x_amostra": round(float(X[::passo].double().sum()), 3)}


def group_folds(groups, n_folds, seed):
    """Dobras que nunca separam as exibicoes de uma mesma imagem: lista de indices de exibicao."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    perm = np.random.RandomState(seed).permutation(len(uniq))
    fold_of = np.empty(len(uniq), dtype=int)
    fold_of[perm] = np.arange(len(uniq)) % n_folds
    por_exibicao = fold_of[np.searchsorted(uniq, groups)]
    return [np.flatnonzero(por_exibicao == k) for k in range(n_folds)]


class _Ajuste:
    """Uma decomposicao espectral (um treino: uma dobra, ou o ajuste final) e o que depende so dela."""

    def __init__(self, X, groups, treino, fracs, rtol, bias_step, device, dtype):
        self.device, self.dtype, self.fracs = device, dtype, fracs
        Xt = torch.as_tensor(X[treino]).to(torch.float64)
        self.n = len(treino)
        media = Xt.mean(0)
        Xt -= media
        posto_max = min(Xt.shape)
        V, lam = spectral_basis(Xt, rtol)
        self.n_descartadas = posto_max - len(lam)    # com n < p, ao menos 1: a media subtraida
        # soma das exibicoes (centradas) de cada imagem: X'y = sum_img (soma_img) y_img'
        self.linhas, inv = np.unique(np.asarray(groups)[treino], return_inverse=True)
        agg = torch.zeros(len(self.linhas), Xt.shape[1], dtype=torch.float64)
        agg.index_add_(0, torch.from_numpy(inv), Xt)
        self.contagem = torch.from_numpy(np.bincount(inv)).to(device, dtype)
        del Xt
        self.V = V.to(device, dtype)
        self.lam = lam.to(device)
        self.grade = alpha_grid(lam.numpy(), bias_step)
        self.XaggT = agg.T.contiguous().to(device, dtype)
        self.media = media.to(device, dtype)
        self.posto = len(lam)

    def projeta(self, X_novo):
        """Escores (n_novo, r) das exibicoes novas na base espectral, com a media do treino."""
        Xn = torch.as_tensor(X_novo).to(self.device, self.dtype)
        return (Xn - self.media) @ self.V

    def momentos(self, Y_linhas):
        """Momentos m = V'X'y de um bloco de alvos e sua media de treino.

        Y_linhas (len(linhas), dc): alvos das imagens de treino, na ordem de `linhas`.
        Centra antes da multiplicacao: a media dos alvos nao muda m (as exibicoes
        centradas somam zero), mas em fp32 sobraria ruido de arredondamento.
        """
        ybar = (self.contagem[:, None] * Y_linhas).sum(0) / self.n
        m = self.V.T @ (self.XaggT @ (Y_linhas - ybar))
        return m, ybar

    def alphas(self, m, return_unreachable=False):
        return fraction_alphas(m, self.lam, self.grade, self.fracs, return_unreachable)

    def pesos(self, m, alpha):
        """Coeficientes no espaco espectral (r, dc); a previsao e escores @ pesos."""
        return m / (self.lam.to(m.dtype)[:, None] + alpha[None, :].to(m.dtype))


class FracRidge:
    """FRR com a fracao escolhida por validacao cruzada, uma por dimensao do alvo (ou uma so)."""

    def __init__(self, fracs=None, n_folds=5, rtol=1e-10, bias_step=BIAS_STEP, chunk=8192,
                 per_target=True, seed=0, device=None, dtype=torch.float32, verbose=True):
        self.fracs = np.asarray(FRACS_DOERIG if fracs is None else fracs, dtype=float)
        if np.any(np.diff(self.fracs) <= 0):
            raise ValueError("fracs precisa ser estritamente crescente")
        self.n_folds, self.rtol, self.bias_step, self.chunk = n_folds, rtol, bias_step, chunk
        self.per_target, self.seed, self.dtype, self.verbose = per_target, seed, dtype, verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _log(self, *a):
        if self.verbose:
            print("[frr]", *a, flush=True)

    def _blocos(self, d):
        return [(c, min(c + self.chunk, d)) for c in range(0, d, self.chunk)]

    def _cv_dobra(self, X, Y, groups, treino, val):
        """Soma dos quadrados dos residuos de validacao por (fracao, alvo), e a variancia total."""
        ajuste = _Ajuste(X, groups, treino, self.fracs, self.rtol, self.bias_step, self.device, self.dtype)
        Z = ajuste.projeta(X[val])
        linhas_val = np.asarray(groups)[val]
        d = Y.shape[1]
        sse = torch.zeros(len(self.fracs), d, dtype=torch.float64)
        sst = torch.zeros(d, dtype=torch.float64)
        inalcancaveis = 0
        for c0, c1 in self._blocos(d):
            todas = np.union1d(ajuste.linhas, linhas_val)
            Yb = torch.from_numpy(np.ascontiguousarray(Y[todas, c0:c1])).to(self.device, self.dtype)
            m, ybar = ajuste.momentos(Yb[np.searchsorted(todas, ajuste.linhas)])
            alphas, n_baixo = ajuste.alphas(m, return_unreachable=True)
            inalcancaveis += n_baixo
            Yv = Yb[np.searchsorted(todas, linhas_val)] - ybar
            sst[c0:c1] = (Yv * Yv).sum(0).double().cpu()
            for i in range(len(self.fracs)):
                res = Yv - Z @ ajuste.pesos(m, alphas[i])
                sse[i, c0:c1] = (res * res).sum(0).double().cpu()
            del Yb, m, alphas, Yv
        info = (ajuste.posto, ajuste.n_descartadas, inalcancaveis)
        del ajuste, Z
        torch.cuda.empty_cache()
        return sse, sst, info

    def fit(self, X, Y, groups, workdir=None):
        """Escolhe as fracoes por CV nas exibicoes de treino e deixa o ajuste final pronto.

        workdir: guarda o resultado de cada dobra (cv_dobra{k}.pt), e uma corrida
        interrompida retoma da primeira que falta.
        """
        t0 = time.time()
        X = X if isinstance(X, torch.Tensor) else torch.from_numpy(np.asarray(X))
        groups = np.asarray(groups)
        n, p = X.shape
        d = Y.shape[1]
        self._log(f"n={n} exibicoes, p={p} voxels, d={d} dimensoes do alvo, {len(self.fracs)} fracoes, "
                  f"{self.n_folds} dobras, dispositivo {self.device}")
        sse = torch.zeros(len(self.fracs), d, dtype=torch.float64)
        sst = torch.zeros(d, dtype=torch.float64)
        self.cv_seconds, self.postos, self.inalcancaveis = [], [], 0
        dados_id = impressao_dos_dados(X, groups)
        for k, val in enumerate(group_folds(groups, self.n_folds, self.seed)):
            arq = os.path.join(workdir, f"cv_dobra{k}.pt") if workdir else None
            # o que define o resultado de uma dobra; uma retomada com outra config nao pode aproveita-la
            meta = {"k": k, "fracs": [float(f) for f in self.fracs], "n_folds": self.n_folds,
                    "seed": self.seed, "n": n, "p": p, "d": d, "rtol": self.rtol,
                    "bias_step": self.bias_step, "dados": dados_id}
            if arq and os.path.exists(arq):
                r = torch.load(arq)
                guardado = r["meta"]
                # dobras gravadas antes da impressao digital dos dados so podem ser conferidas pelo formato
                esperado = meta if "dados" in guardado else {c: v for c, v in meta.items() if c != "dados"}
                if guardado != esperado:
                    raise ValueError(f"{arq} foi gerado com outra configuracao ou outros dados "
                                     f"({guardado} != {esperado}); apague o diretorio para refazer")
                aviso = "" if "dados" in guardado else " (sem impressao digital dos dados: conferido so o formato)"
                self._log(f"dobra {k + 1}/{self.n_folds}: retomada de {arq}{aviso}")
            else:
                t1 = time.time()
                treino = np.setdiff1d(np.arange(n), val)
                s, st, info = self._cv_dobra(X, Y, groups, treino, val)
                r = {"sse": s, "sst": st, "segundos": time.time() - t1, "posto": info[0],
                     "inalcancaveis": info[2], "meta": meta}
                if arq:
                    os.makedirs(workdir, exist_ok=True)
                    torch.save(r, arq + ".tmp")
                    os.replace(arq + ".tmp", arq)
                self._log(f"dobra {k + 1}/{self.n_folds}: {r['segundos']:.0f} s, posto {info[0]} "
                          f"({info[1]} direcoes descartadas)")
            sse += r["sse"]
            sst += r["sst"]
            self.cv_seconds.append(r["segundos"])
            self.postos.append(r["posto"])
            self.inalcancaveis += r["inalcancaveis"]
        self.cv_sse, self.cv_sst = sse, sst
        if self.per_target:
            self.best_ = sse.argmin(0)
        else:
            self.best_ = torch.full((d,), int(sse.sum(1).argmin()), dtype=torch.long)

        t1 = time.time()
        self._Y = Y
        self._final = _Ajuste(X, groups, np.arange(n), self.fracs, self.rtol, self.bias_step,
                              self.device, self.dtype)
        self.final_seconds = time.time() - t1
        # soma o que cada dobra gastou, inclusive as retomadas de uma corrida anterior
        self.fit_seconds = sum(self.cv_seconds) + self.final_seconds
        self._log(f"ajuste final: posto {self._final.posto}; total {self.fit_seconds:.0f} s")
        return self

    def predict(self, X_novo, saida=torch.float16):
        """Previsao (n_novo, d) em CPU, calculada por blocos de alvos."""
        Y, aj = self._Y, self._final
        Z = aj.projeta(X_novo)
        d = Y.shape[1]
        out = torch.empty(Z.shape[0], d, dtype=saida)
        melhor = self.best_.to(self.device)
        for c0, c1 in self._blocos(d):
            Yb = torch.from_numpy(np.ascontiguousarray(Y[aj.linhas, c0:c1])).to(self.device, self.dtype)
            m, ybar = aj.momentos(Yb)
            alphas = aj.alphas(m)                                       # (F, dc)
            alpha = alphas.gather(0, melhor[c0:c1][None, :])[0]          # fracao escolhida de cada alvo
            out[:, c0:c1] = (ybar + Z @ aj.pesos(m, alpha)).to(saida).cpu()
            del Yb, m, alphas
        return out

    def resumo(self):
        """Diagnosticos da validacao cruzada: R2 (agregado nas dimensoes) por fracao e o histograma das escolhas."""
        sst = float(self.cv_sst.sum())
        r2_fixa = [1.0 - float(self.cv_sse[i].sum()) / sst for i in range(len(self.fracs))]
        sse_escolhida = self.cv_sse.gather(0, self.best_[None, :])[0].sum()
        return {
            "fracs": [round(float(f), 4) for f in self.fracs],
            "cv_r2_por_fracao": r2_fixa,
            "cv_r2_escolhida": 1.0 - float(sse_escolhida) / sst,
            "histograma_fracoes": torch.bincount(self.best_, minlength=len(self.fracs)).tolist(),
            "por_alvo": self.per_target,
            "posto_final": self._final.posto,
            "pares_fracao_alvo_inalcancaveis": self.inalcancaveis,   # fracao menor que a grade alcanca; 0 = grade suficiente
            "segundos_cv": float(sum(self.cv_seconds)),
            "segundos_final": self.final_seconds,
            "segundos_total": self.fit_seconds,
        }
