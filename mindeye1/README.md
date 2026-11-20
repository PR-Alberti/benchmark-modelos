# mindeye1/: o MindEye1, adaptado ao benchmark

Código do [MindEye1](https://github.com/MedARC-AI/fMRI-reconstruction-NSD) (Scotti et al., 2023). O
README original está em [README-original.md](README-original.md); instalação, protocolo e a lista dos
ajustes, em [docs/MINDEYE1.md](../docs/MINDEYE1.md).

O primeiro commit que adiciona esta pasta traz o código como publicado. Os ajustes vêm depois e estão
marcados no código com `# benchmark-modelos`.

| Arquivo em `src/` | Papel |
|---|---|
| `Train_MindEye.py` | treino do alto nível (voxels → CLIP ViT-L/14 e prior). *Ajustado* |
| `train_autoencoder.py` | treino do baixo nível (voxels → imagem borrada). *Ajustado* |
| `Reconstructions.py` | reconstruções com o Versatile Diffusion. *Ajustado* |
| `Retrievals_testset.py` | retrieval no teste do benchmark. *Novo* |
| `nsd_benchmark.py` | lê as mesmas exibições de treino e o mesmo teste dos outros modelos (usa `mindeye_ridge`). *Novo* |
| `Train_MindEye_multisubject.py`, `run_me1_multisubject.sh` | o MindEye1 multi-sujeito: a rede do MindEye1 com o protocolo de treino do MindEye2 (pré-treino nos outros 7 sujeitos, ajuste no sujeito 1). *Novo*, ainda fora do `treina.py` e do site |
| `vd_compat.py` | compatibilidade do Versatile Diffusion com as bibliotecas atuais. *Novo* |
| `utils.py`, `models.py`, `convnext.py` | do original (`utils.py` ajustado) |
| os demais (`.ipynb`, `Retrievals.py`, `Reconstruction_Metrics.py`, `accel.slurm`, `setup.sh`...) | do original, como vieram; os `.ipynb` são as versões notebook dos `.py` |

Os scripts rodam de dentro de `mindeye1/src/` e usam caminhos relativos (`../train_logs`,
`../../results`): **não mova arquivos desta pasta**. Para rodar, use `scripts/me1_run.sh` ou o
`src/treina.py` com `"modelo": "mindeye1"` (o multi-sujeito, `src/run_me1_multisubject.sh`). As saídas vão para `mindeye1/train_logs/` (fora do git); as do multi-sujeito, para
`$ROOT/train_logs/` (padrão `~/mindeye1_multisubject`).
