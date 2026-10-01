"""Metricas sobre embeddings CLIP previstos: o retrieval top-1 do benchmark e a similaridade com o alvo."""
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from . import utils


def retrieval_top1(prev, alvo, seed, device="cuda", n_sorteios=30, n_candidatas=300, centro=None):
    """Retrieval top-1 do verify_retrieval.py / final_evaluations.py.

    n_sorteios sorteios de n_candidatas imagens entre as do teste, nos dois sentidos:
      fwd: dada cada imagem, achar o seu cerebro entre as previsoes;
      bwd: dado cada cerebro (previsao), achar a sua imagem.
    Cosseno sob autocast fp16, mesma semente e mesma sequencia de sorteios dos outros
    modelos: a comparacao e pareada.

    E o que o codigo calcula, nao o que o comentario do final_evaluations.py diz ("fwd:
    brain, clip"): batchwise_cosine_similarity(Z, B) devolve a matriz com as linhas de B,
    e o topk escolhe a melhor coluna de cada linha. O artigo define "Image Retrieval" como
    cerebro -> imagem, que aqui e o bwd.

    prev, alvo: (n, ...) em ordem de imagem, a mesma nos dois. Devolve media e IC95% por sentido.

    centro: vetor (achatado) subtraido de previsao e alvo antes do cosseno, em geral a media
    de treino. Previsoes de regressao ficam encolhidas em direcao a media e, no cosseno
    bruto, a imagem mais proxima da media vence qualquer previsao ("hubness"); tirar a
    media de treino separa esse efeito do sinal. So o diagnostico: a comparacao entre
    modelos usa o cosseno bruto, como o artigo.
    """
    centro = None if centro is None else centro.reshape(-1).to(device).float()
    fwds, bwds = [], []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        np.random.seed(seed)
        for _ in range(n_sorteios):
            samps = np.random.choice(np.arange(len(alvo)), size=n_candidatas, replace=False)
            emb = alvo[samps].to(device).float()
            emb_ = prev[samps].to(device).float()
            emb, emb_ = emb.reshape(len(emb), -1), emb_.reshape(len(emb_), -1)
            if centro is not None:
                emb, emb_ = emb - centro, emb_ - centro
            emb = nn.functional.normalize(emb, dim=-1)
            emb_ = nn.functional.normalize(emb_, dim=-1)
            labels = torch.arange(len(emb)).to(device)
            fwds = np.append(fwds, utils.topk(utils.batchwise_cosine_similarity(emb_, emb), labels, k=1).item())
            bwds = np.append(bwds, utils.topk(utils.batchwise_cosine_similarity(emb, emb_), labels, k=1).item())
    out = {}
    for chave, arr in (("fwd", fwds), ("bwd", bwds)):
        m, sd = np.mean(arr), np.std(arr) / np.sqrt(len(arr))
        ic = stats.norm.interval(0.95, loc=m, scale=sd)
        out[chave] = {"media": float(m), "ic95": [float(ic[0]), float(ic[1])]}
    return out


def cosseno_com_vetor(x, v, device="cuda", lote=100):
    """Cosseno de cada linha de x com o vetor v (por exemplo, a media de treino), por imagem."""
    v = v.reshape(-1).to(device).float()
    out = []
    for i in range(0, len(x), lote):
        a = x[i:i + lote].reshape(len(x[i:i + lote]), -1).to(device).float()
        out.append(nn.functional.cosine_similarity(a, v[None], dim=1))
    return torch.cat(out).cpu().numpy()


def similaridades(prev, alvo, device="cuda", lote=100, centro=None):
    """Cosseno e correlacao de Pearson, por imagem, entre o embedding previsto e o verdadeiro.

    Com `centro` (a media de treino), tira-o dos dois antes: o cosseno bruto de dois embeddings CLIP
    ja e ~0,5 so por eles dividirem a media, e o centrado mede o que a previsao acrescenta a ela.
    """
    centro = None if centro is None else centro.reshape(-1).to(device).float()
    cos, pearson = [], []
    for i in range(0, len(prev), lote):
        a = prev[i:i + lote].reshape(min(lote, len(prev) - i), -1).to(device).float()
        b = alvo[i:i + lote].reshape(a.shape[0], -1).to(device).float()
        if centro is not None:
            a, b = a - centro, b - centro
        cos.append(nn.functional.cosine_similarity(a, b, dim=1))
        a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
        pearson.append(nn.functional.cosine_similarity(a, b, dim=1))
    return torch.cat(cos).cpu().numpy(), torch.cat(pearson).cpu().numpy()
