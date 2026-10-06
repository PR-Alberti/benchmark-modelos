#!/usr/bin/env python
"""Cria shards sinteticos no formato do webdataset_avg_split do MindEye1 (subj01).

Servem para medir memoria e tempo do treino sem baixar os 39 GB de dados reais: voxels e
imagens sao ruido, entao as metricas nao significam nada. Mesmos nomes de arquivo, chaves e
formas dos shards reais (3 repeticoes x 15.724 voxels em fp16, JPEG 425 x 425):

    python scripts/me1_fake_data.py /tmp/me1_fake
    ME1_DATA=/tmp/me1_fake scripts/me1_run.sh train --num_epochs=3 --no-ckpt_saving

O treino le um numero fixo de batches por epoca (8.859 exemplos), reamostrando os shards,
entao poucos exemplos bastam. ~360 MB.
"""
import io
import json
import os
import sys

import numpy as np
import webdataset as wds
from PIL import Image


def main(saida, n_por_shard=40, n_teste=491, seed=0):
    rng = np.random.default_rng(seed)

    def amostra(k):
        jpg = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (425, 425, 3), dtype=np.uint8)).save(jpg, format="JPEG")
        return {"__key__": f"s{k:06d}", "jpg": jpg.getvalue(),
                "nsdgeneral.npy": rng.standard_normal((3, 15724)).astype(np.float16),
                "coco73k.npy": np.array([k]), "trial.npy": np.array([k]), "num_uniques.npy": np.array([3])}

    base = os.path.join(saida, "webdataset_avg_split")
    shards = ([f"train/train_subj01_{i}.tar" for i in range(18)] + ["val/val_subj01_0.tar"])
    k = 0
    for nome, n in [(s, n_por_shard) for s in shards] + [(f"test/test_subj01_{i}.tar", n_teste) for i in range(2)]:
        os.makedirs(os.path.dirname(os.path.join(base, nome)), exist_ok=True)
        with wds.TarWriter(os.path.join(base, nome)) as w:
            for _ in range(n):
                w.write(amostra(k))
                k += 1
    with open(os.path.join(base, "metadata_subj01.json"), "w") as f:
        json.dump({"totals": {"train": 8559, "val": 300, "test": 982}}, f)
    print(f"{k} exemplos sinteticos em {base}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
