# vendor/: código de terceiros

Cópias de outros projetos que o pipeline usa e que não estão no `pip`. **Não edite estes arquivos**: o
código deste repositório fica em `src/`. Se algo daqui precisar mudar, a mudança vai em `src/` (por
exemplo, `src/mindeye_ridge/decodificador.py` ajusta a configuração do unCLIP depois de carregá-la).

| Pasta | O que é | Origem | Quem usa |
|---|---|---|---|
| `generative_models/` | SDXL e o unCLIP da Stability AI (pacote `sgm`), mais as configurações `.yaml` dos modelos | [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models), como veio no [MindEye2](https://github.com/MedARC-AI/MindEyeV2) | reconstrução, refinamento, alvos CLIP bigG (`src/mindeye_ridge/decodificador.py`, `clip_targets.py`, as etapas em `src/`) |
| `autoencoder/` | ConvNeXt usado pela perda do ramo *blurry* do MindEye2 | [MindEye2](https://github.com/MedARC-AI/MindEyeV2) | `src/train_ridgeonly.py` (só com `--blurry_recon`) |

O caminho entra no `sys.path` por `mindeye_ridge.paths.add_vendored_to_path()`: o `generative_models` se
importa tanto como `generative_models.sgm` quanto como `sgm`, e os dois precisam funcionar. Cada pasta
mantém a licença própria.

O MindEye1 (`mindeye1/`, na raiz) também é código de terceiros, mas com ajustes para o benchmark; ele
fica fora daqui porque os scripts dele usam caminhos relativos à própria pasta. Veja `mindeye1/README.md`.
