"""Como as repeticoes de uma imagem viram amostras de treino: o campo "agregacao" do dataset.

O NSD mostra cada imagem ate 3 vezes. Os modelos do benchmark tratam isso de jeitos diferentes, e
aqui qualquer um deles pode ser escolhido para qualquer modelo:

    exibicoes   cada exibicao e uma amostra (o que o MindEye2 e o FRR fazem)
    media       a media dos betas das repeticoes: uma amostra por imagem, com menos ruido
    sorteio     uma amostra por imagem cada vez que ela e usada, com uma das repeticoes sorteada
                (o alto nivel do MindEye1 faz um rodizio: a repeticao train_i % 3 de cada lote)
    combinacao  uma amostra por imagem cada vez que ela e usada, com uma combinacao aleatoria das
                repeticoes, como o baixo nivel do MindEye1 (utils.voxel_select): em metade dos
                lotes uma media com pesos aleatorios, em 30% a media e em 20% uma repeticao sorteada

None, o padrao, e o que cada modelo faz no benchmark. O teste nao muda: e sempre a media das 3
repeticoes. "sorteio" e "combinacao" mudam a amostra a cada epoca, entao so valem para modelos
treinados por epocas (nao para o FRR, que ajusta numa passada).
"""
import numpy as np

MODOS = ("exibicoes", "media", "sorteio", "combinacao")
POR_EPOCA = ("sorteio", "combinacao")


def grupos(imagem, beta):
    """Ids das imagens (crescentes) e, para cada uma, as linhas de betas das suas exibicoes."""
    imagem, beta = np.asarray(imagem), np.asarray(beta)
    ids = np.unique(imagem)
    return ids, [beta[imagem == i] for i in ids]


def tres(linhas):
    """3 linhas de betas: as que existem, repetidas na ordem ate completar 3 (como no MindEye1)."""
    return np.resize(np.asarray(linhas), 3)


def seleciona(vox3, n_rep, modo, gerador=None):
    """Uma amostra por linha a partir das repeticoes empilhadas.

    vox3: (n, 3, voxels), as repeticoes de cada amostra completadas ate 3 (`tres`); n_rep: (n,),
    quantas sao de verdade (so elas entram na media e nos sorteios). Devolve (n, voxels), no dtype
    de vox3. `gerador` (torch.Generator) torna os sorteios reprodutiveis.
    """
    import torch          # aqui dentro: o dataset_controlado (e o treina.py) importa este modulo sem torch

    if modo not in MODOS:
        raise ValueError(f"agregacao {modo!r}: use uma de {MODOS}")
    if modo == "exibicoes":
        return vox3[:, 0]
    n = vox3.shape[0]
    n_rep = torch.as_tensor(n_rep).long().view(n)
    v = vox3.float()
    validas = (torch.arange(3)[None] < n_rep[:, None]).float()                 # (n, 3)
    if modo == "combinacao":
        u = torch.rand(1, generator=gerador).item()
        if u <= 0.5:
            pesos = torch.rand(n, 3, generator=gerador) * validas
            return ((pesos[..., None] * v).sum(1) / pesos.sum(1, keepdim=True)).to(vox3.dtype)
        modo = "media" if u <= 0.8 else "sorteio"
    if modo == "media":
        return ((validas[..., None] * v).sum(1) / n_rep[:, None]).to(vox3.dtype)
    escolha = (torch.rand(n, generator=gerador) * n_rep).long()                # uniforme entre as que existem
    return vox3[torch.arange(n), escolha]
