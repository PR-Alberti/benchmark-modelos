# src/: o código do repositório

O pipeline está explicado em [docs/GUIA.md](../docs/GUIA.md). O código fica instalado no ambiente
(`pip install -e .`, veja o `pyproject.toml` na raiz), então `import mindeye_ridge`, `import treina` e
`import report` funcionam de qualquer pasta.

## Comandos (rodados à mão)

| Arquivo | O que faz |
|---|---|
| `treina.py` | treina um experimento: monta o dataset, grava a configuração e chama o script do modelo (`--lista`, `--dry_run`, `--so_dataset`, `--reconstrucao`) |
| `experimentos.py` | os experimentos, um dicionário cada: `BENCHMARK` (os modelos do benchmark), `REPRODUCOES`, `EXPERIMENTOS` (os novos) |
| `make_site.py` | gera o site (`results/site/index.html` e `apresentacao.html`); os scripts o chamam sozinhos |
| `make_benchmark.py` | gera o `BENCHMARK.md` e `results/benchmark/index.html` |
| `make_comparison.py` | figura imagem vista × reconstruções de vários modelos |

## `etapas/`: o pipeline (chamado pelos `scripts/run_*.sh`)

Vieram dos notebooks do MindEye2 e rodam de cima para baixo. Não têm teste automático: depois de mexer,
rode `--help` (importa tudo) e, de verdade, só na máquina de treino. Veja "Mudar uma etapa" no guia.

| Arquivo | Etapa | Grava em `results/evals/<m>/` |
|---|---|---|
| `train_ridgeonly.py` | treino do MindEye2 ridge-only | `train_logs/<m>/last.pth` |
| `recon_inference.py` | MindEye2: embedding do teste → prior → unCLIP | `_all_clipvoxels.pt`, `_all_recons.pt`, `_all_predcaptions.pt`, `_all_blurryrecons.pt` |
| `run_frr.py` | FRR: ajuste por validação cruzada e previsão do teste | `_all_clipvoxels.pt`; `results/tables/<m>_frr.json` |
| `run_frr_ms.py` | FRR multi-sujeito | idem |
| `recon_embeddings.py` | um embedding pronto (o do FRR) → unCLIP | `_all_recons.pt`, `_all_predcaptions.pt` |
| `enhanced_recon_inference.py` | refino das reconstruções com a legenda | `_all_enhancedrecons.pt` |
| `final_evaluations.py` | métricas de imagem e retrieval | `results/tables/<m>_all_{recons,enhancedrecons}.csv` |
| `verify_retrieval.py` | só o retrieval do MindEye2, sem gerar imagens | `_all_clipvoxels.pt`, `_retrieval.json` |

## `mindeye_ridge/`: a biblioteca

O código reutilizável, com testes em `tests/`.

| Módulo | O que tem |
|---|---|
| `paths.py` | todos os caminhos (repositório, dados, resultados) num lugar só |
| `nsd_data.py` | lê as exibições e os betas do NSD |
| `dataset_controlado.py` | recortes do treino: sessões, imagens, repetições, classes semânticas |
| `agregacao.py` | como as repetições de uma imagem viram amostras de treino |
| `semantica.py` | que fração da tela cada categoria do COCO ocupa em cada imagem |
| `clip_targets.py` | cache dos embeddings CLIP bigG das imagens (os alvos) |
| `frr.py` | a ridge fracionária, para alvos de centenas de milhares de dimensões |
| `multisujeito.py` | o FRR multi-sujeito |
| `embedding_metrics.py` | o retrieval e as similaridades entre embeddings |
| `decodificador.py` | do embedding bigG para imagem e legenda (unCLIP e GIT) |
| `resultados.py` | o retrieval de um modelo pelo nome, de onde estiver gravado |
| `models.py`, `utils.py`, `modeling_git.py` | vieram do MindEye2 (redes, funções de treino, a rede GIT). Os checkpoints dependem dos nomes das camadas: mude com cuidado |

## `report/`: o site e o BENCHMARK.md

| Arquivo | O que tem |
|---|---|
| `catalogo.py` | descobre cada modelo treinado em disco e monta o registro dele (`results/site/modelos/<m>.json`) |
| `site.py`, `apresentacao.py` | o site e a apresentação |
| `config.py` | os modelos do benchmark, os grupos de comparação e as métricas de cada tabela |
| `data.py`, `rows.py`, `fmt.py` | leitura dos resultados, linhas das tabelas, formatação |
| `page.py`, `markdown.py`, `texts.py` | a página do benchmark, o `BENCHMARK.md` e as conclusões escritas |
| `galeria_site.py`, `exemplos_site.py`, `semantica_site.py` | galeria de imagens, exemplos e simulador do site |
| `conteudo/` | o texto fixo do site e da apresentação (HTML) |
| `assets/` | CSS e JavaScript |
