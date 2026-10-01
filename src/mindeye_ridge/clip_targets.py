"""Cache em disco dos embeddings CLIP (ViT-bigG/14, 256 x 1664 tokens) das imagens do NSD.

E o alvo de treino de qualquer modelo que decodifique o embedding CLIP direto
(a FRR): 425.984 numeros por imagem, 852 KB em fp16. As 9.000 imagens de treino
do sujeito ocupam 7,7 GB, e calcular tudo leva uns 10 minutos; o cache evita
refazer isso a cada experimento.

Um diretorio por sujeito, <data_path>/clip_targets/subj0N/:
    ids.npy        ids das imagens cobertas, em ordem crescente
    emb_fp16.npy   (len(ids), 425984), uma linha por id, aberto como memmap
    pronto.npy     quais linhas ja foram calculadas (gravado so depois de gravar as linhas)

O embedder e chamado como no treino do MindEye2: ViT-bigG em fp16, sob autocast, sobre
a imagem 224 x 224 em fp16 do coco_images_224_float16.hdf5. Para o teste, o
verify_retrieval.py e o final_evaluations.py fazem diferente: a imagem entra em fp32
(all_images.pt). A saida do embedder muda com isso (cosseno 0,996 entre as duas
versoes da mesma imagem), entao o alvo de avaliacao tem funcao propria,
embeddings_avaliacao(), e nao passa por este cache.
"""
import os
import time

import h5py
import numpy as np
import torch

from . import paths

SEQ, DIM = 256, 1664
D_ALVO = SEQ * DIM


def carrega_embedder(device="cuda"):
    paths.add_vendored_to_path()
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
    emb = FrozenOpenCLIPImageEmbedder(arch="ViT-bigG-14", version="laion2b_s39b_b160k",
                                      output_tokens=True, only_tokens=True)
    return emb.half().to(device)


def embeddings_avaliacao(imagens, device="cuda", lote=100):
    """Embeddings das imagens do teste, como o verify_retrieval.py e o final_evaluations.py os calculam.

    imagens (n, 3, 224, 224) em fp32 (o all_images.pt). Devolve (n, 425984) em fp16, na CPU.
    """
    embedder = carrega_embedder(device)
    saida = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(0, len(imagens), lote):
            saida.append(embedder(imagens[i:i + lote].to(device)).flatten(1).cpu())
    del embedder
    torch.cuda.empty_cache()
    return torch.cat(saida)


class ClipTargets:
    def __init__(self, data_path, subj, image_ids):
        """image_ids: todas as imagens de treino que o sujeito pode vir a usar (as 40 sessoes)."""
        self.data_path, self.subj = data_path, subj
        self.dir = f"{data_path}/clip_targets/subj0{subj}"
        os.makedirs(self.dir, exist_ok=True)
        self.ids = np.unique(np.asarray(image_ids))
        f_ids, f_emb, f_ok = (f"{self.dir}/{n}" for n in ("ids.npy", "emb_fp16.npy", "pronto.npy"))
        if os.path.exists(f_ids):
            if not np.array_equal(np.load(f_ids), self.ids):
                raise ValueError(f"{f_ids} cobre outras imagens; apague {self.dir} para refazer o cache")
            self.pronto = np.load(f_ok)
            self.emb = np.load(f_emb, mmap_mode="r+")
        else:
            self.pronto = np.zeros(len(self.ids), dtype=bool)
            self.emb = np.lib.format.open_memmap(f_emb, mode="w+", dtype=np.float16,
                                                 shape=(len(self.ids), D_ALVO))
            np.save(f_ok, self.pronto)
            np.save(f_ids, self.ids)

    def linhas(self, image_ids):
        """Linha de cada imagem no cache."""
        image_ids = np.asarray(image_ids)
        r = np.searchsorted(self.ids, image_ids)
        if np.any(r >= len(self.ids)) or np.any(self.ids[np.minimum(r, len(self.ids) - 1)] != image_ids):
            raise KeyError("imagem fora do cache")
        return r

    def media_ponderada(self, linhas_por_exibicao, bloco=500):
        """Media dos alvos sobre as exibicoes (a imagem de cada uma, em `linhas_por_exibicao`): (425984,) fp32."""
        linhas, cont = np.unique(linhas_por_exibicao, return_counts=True)
        soma = torch.zeros(D_ALVO, dtype=torch.float64)
        for i in range(0, len(linhas), bloco):
            y = torch.from_numpy(np.asarray(self.emb[linhas[i:i + bloco]])).double()
            soma += (torch.from_numpy(cont[i:i + bloco]).double()[:, None] * y).sum(0)
        return (soma / cont.sum()).float()

    def _grava_pronto(self):
        self.emb.flush()                  # as linhas antes do aviso de que estao prontas
        np.save(f"{self.dir}/pronto.npy.tmp.npy", self.pronto)
        os.replace(f"{self.dir}/pronto.npy.tmp.npy", f"{self.dir}/pronto.npy")

    def garante(self, image_ids, device="cuda", lote=32):
        """Calcula os embeddings que ainda faltam entre `image_ids`. Devolve quantos calculou."""
        faltam = self.linhas(np.unique(np.asarray(image_ids)))
        faltam = faltam[~self.pronto[faltam]]
        if len(faltam) == 0:
            return 0
        print(f"[clip] calculando {len(faltam)} embeddings CLIP bigG (cache em {self.dir})", flush=True)
        embedder = carrega_embedder(device)
        t0 = time.time()
        with h5py.File(f"{self.data_path}/coco_images_224_float16.hdf5", "r") as f, \
                torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            imagens = f["images"]
            for i in range(0, len(faltam), lote):
                linhas = faltam[i:i + lote]                       # ja em ordem crescente de id
                x = torch.from_numpy(imagens[self.ids[linhas]]).to(device)
                self.emb[linhas] = embedder(x).flatten(1).cpu().numpy()
                self.pronto[linhas] = True
                if (i // lote) % 20 == 19:
                    self._grava_pronto()
                    feito = i + len(linhas)
                    print(f"[clip] {feito}/{len(faltam)}  {feito / (time.time() - t0):.1f} img/s", flush=True)
        self._grava_pronto()
        del embedder
        torch.cuda.empty_cache()
        print(f"[clip] pronto em {time.time() - t0:.0f} s", flush=True)
        return len(faltam)
