# Notebooks do MindEye2 original

Os notebooks do repositório [MedARC-AI/MindEyeV2](https://github.com/MedARC-AI/MindEyeV2), como
vieram, só de referência: importam `utils` e `models` do layout antigo (tudo em `src/`) e não rodam
daqui sem ajustes. O que este repositório usa são as versões em script, em `src/`
(`train_ridgeonly.py`, `recon_inference.py`, `enhanced_recon_inference.py`,
`final_evaluations.py`), geradas a partir deles (`legacy/generators/`). O
[docs/README-mindeye2-original.md](../../docs/README-mindeye2-original.md) descreve cada um.

Para treinar, veja [`../treino.ipynb`](../treino.ipynb).
