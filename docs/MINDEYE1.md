# MindEye1 — instalação, preparação e o que mudou

O código do MindEye1 ([MedARC-AI/fMRI-reconstruction-NSD](https://github.com/MedARC-AI/fMRI-reconstruction-NSD),
commit `6b30ea5`, artigo: [Scotti et al., NeurIPS 2023](https://arxiv.org/abs/2305.18274)) está em
`mindeye1/`. O primeiro commit que o adiciona traz o código como publicado; os ajustes vêm no
commit seguinte, e cada um está marcado no código com `# benchmark-modelos`.

## O modelo, em uma tabela

| | |
|---|---|
| Entrada | 15.724 voxels do `nsdgeneral` (subj01), média das 3 repetições no teste |
| Alto nível | MLP de ~950 M de parâmetros → embedding CLIP **ViT-L/14** de 257 × 768 (camada oculta), mais um diffusion prior (~56 M); treinados juntos, 240 épocas |
| Baixo nível | outro MLP (`Voxel2StableDiffusionModel`) → latente do VAE do Stable Diffusion, 120 épocas |
| Imagem | **Versatile Diffusion** a partir do embedding previsto, com img2img a partir da imagem borrada (força 0,85) |
| Dados | `webdataset_avg_split` de [`pscotti/naturalscenesdataset`](https://huggingface.co/datasets/pscotti/naturalscenesdataset): as 40 sessões do subj01 (8.859 imagens de treino + validação, 982 de teste) |

Diferenças para o MindEye2 que importam no benchmark: CLIP ViT-L/14 (e não bigG), dados de 40
sessões sempre (não há versão de 1 sessão publicada), 982 imagens de teste (não 1.000) e um
gerador diferente. O retrieval do MindEye1 é medido do mesmo jeito (top-1 entre 300, 30 sorteios).

## Instalação

Pré-requisito: o ambiente e os dados do MindEye2 (`setup/bootstrap.sh`, veja o [SETUP.md](SETUP.md)).
O MindEye1 usa o mesmo ambiente `~/envs/fmri`; nada do MindEye2 muda.

```bash
scripts/me1_setup.sh --check          # o que falta, sem instalar nada
scripts/me1_setup.sh                  # instala 3 pacotes e cria os links em mindeye1/train_logs
mindeye1/download.py --stage test ckpts vd     # inferência com os modelos publicados, ~31 GB
mindeye1/download.py --stage train             # + dados de treino, ~39 GB
```

| Etapa do `download.py` | O que traz | Tamanho | Onde fica |
|---|---|---|---|
| `test` | 2 shards de teste do subj01 + metadados | 4,3 GB | `$ME1_DATA` (padrão `~/mindeye1`) |
| `train` | 18 shards de treino + 1 de validação | 38,7 GB | `$ME1_DATA` |
| `ckpts` | modelos publicados: alto nível (`prior_257_final_subj01_bimixco_softclip_byol/last.pth`, 12 GB — inclui o estado do otimizador) e low-level (`autoencoder_subj01_4x_locont_no_reconst/epoch120.pth`, 2,5 GB) | 14,5 GB | `$ME1_DATA/mindeye_models` |
| `vd` | Versatile Diffusion, só as partes que o `diffusers` carrega (sem `pretrained_pth/`, 36 GB inúteis aqui) | 12 GB | cache do HF, em `$MINDEYE_DATA/.cache` |

O `me1_setup.sh` instala no ambiente, com `--no-deps` para não mexer no torch:

| Pacote | Para quê |
|---|---|
| `info-nce-pytorch==0.1.0`, `pytorch-msssim==1.0.0` | importados pelo `utils.py` do MindEye1 |
| `bitsandbytes==0.43.3` | AdamW de 8 bits, que faz o treino caber em 20 GB |

e liga em `mindeye1/train_logs/` (os scripts procuram tudo ali, por caminho relativo a `mindeye1/src`):
o VAE e o ConvNeXt que o MindEye2 já baixou (treino do low-level) e os dois modelos publicados.

Os outros pacotes do `environment.yaml` original (torch 2.0.1, diffusers 0.13, accelerate 0.19, …)
**não** são instalados: o MindEye1 roda nas versões do ambiente do MindEye2 com os ajustes abaixo.

## Rodar

```bash
scripts/me1_run.sh train                  # alto nível, config para 20 GB (~17,5 h numa A4500)
PAPER=1 scripts/me1_run.sh train          # config do artigo: batch 32, AdamW (precisa de ~24+ GB)
scripts/me1_run.sh lowlevel               # low-level, 120 épocas
scripts/me1_run.sh retrieval              # retrieval no teste, modelo publicado
scripts/me1_run.sh recon --max_images 8   # reconstruções (sem --max_images: as 982)
scripts/me1_run.sh metrics prior_257_final_subj01_bimixco_softclip_byol_recons_img2img0.85_1samples.pt
```

`MODEL_NAME=<pasta em mindeye1/train_logs>` troca o modelo de alto nível (padrão: o publicado;
no `train`, o nome da saída, padrão `subj01_me1`) e `AE_NAME` o de baixo nível. Argumentos extras
vão direto para o script Python (por exemplo `scripts/me1_run.sh train --num_epochs=3`).

## O que foi testado nesta máquina (RTX A4500, 20 GB; 30 GB de RAM)

| Etapa | Situação |
|---|---|
| Ambiente, `me1_setup.sh`, `download.py` | testados |
| Treino do alto nível, config do artigo (batch 32, AdamW) | **não cabe**: estoura já no 1º forward (atenção do prior com 257 tokens) |
| Treino do alto nível, batch 16 + AdamW normal | **não cabe**: estoura ao criar os estados do Adam (só a camada final, 4096 → 197.376, tem 808 M de parâmetros; pesos + gradientes + Adam ≈ 16 GB) |
| Treino do alto nível, `scripts/me1_run.sh train` (batch 16, Adam 8 bits, validação em blocos) | **roda**: 3 épocas completas com validação, em dados sintéticos no formato real; pico de 17,5 GiB; ~263 s por época → ~17,5 h para as 240 |
| Treino do low-level (dados do benchmark, 1 sessão) | **roda**: pico de 19,9 GB; 36 s por época com o cache de alvos (142 s sem ele) |
| Reconstrução (modelos publicados, 4 imagens do teste do benchmark) | **roda**: ~3 s por imagem, pico de 14,6 GB na GPU e 22 GB de RAM; as imagens saem com o conteúdo certo |
| `Retrievals_testset.py` e `Reconstruction_Metrics.py` (pipeline original) | não testados; no benchmark, o retrieval e as métricas vêm do `final_evaluations.py`, como nos outros modelos |
| Busca no LAION-5B (`Retrievals.py`, parte de cima) | **não roda em lugar nenhum**: o serviço `knn.laion.ai` saiu do ar com o LAION-5B |

O tempo de treino foi medido com dados sintéticos (`tools/me1_fake_data.py`); com os shards reais
a leitura pode ser mais lenta. A referência dos autores para o retrieval do modelo publicado está
no próprio `Retrievals.py`: subj01, fwd 97,18 % e bwd 94,68 %. O código do MindEye1 calcula as duas
direções como o do MindEye2 (o `batchwise_cosine_similarity` dos dois transpõe o resultado): fwd é
imagem → cérebro (cada imagem procura o seu cérebro entre 300) e bwd, cérebro → imagem.

## Os ajustes, um por um

**Mudam o treino em relação ao artigo** (desligados por padrão; o `me1_run.sh train` liga):

| Opção | Por quê |
|---|---|
| `--batch_size=16` (artigo: 32) | com 32, a atenção do prior sobre 257 tokens estoura no forward. A perda contrastiva fica com metade dos negativos e o número de passos dobra |
| `--adam8bit` | os estados do AdamW em fp32 ocupam 8 GB; em 8 bits, 2 GB. É o que faz caber |

**Não mudam nenhum número:**

| Onde | O quê |
|---|---|
| `Train_MindEye.py`, `--val_chunk=50` | a validação usa batch 300 (o retrieval é entre 300); o modelo vê os 300 em blocos de 50 e perda e retrieval continuam calculados sobre os 300 |
| `Train_MindEye.py` | no modo `--hidden`, o `Clipper` carrega dois CLIP ViT-L e só usa o do HF; o OpenAI vai para a CPU (1,7 GB livres) |
| `Train_MindEye.py` | os tensores de batch 300 da validação são soltos antes da época seguinte (sem isso a época 1 estoura por ~200 MB) |
| `Train_MindEye.py` | o `accelerate` 0.24 quebra com os DataLoaders do MindEye1 (`batch_size=None`); numa GPU, o `prepare` deles só movia os batches para a GPU, e um wrapper faz isso |
| `utils.py` | `clip_retrieval` passa a ser opcional (só a busca no LAION usa) |
| `Reconstructions.py` | checkpoints carregados na CPU (o de 12 GB não cabe na GPU junto com o Versatile Diffusion); `--max_images N` |
| `train_autoencoder.py` | o caminho dos dados era fixo no cluster dos autores; agora vem de `ME1_DATA` |
| `Retrievals_testset.py` (novo) | a seção de retrieval no teste do `Retrievals.py`, que no original só roda dentro do Jupyter, sem a parte do LAION |
| `vd_compat.py` (novo) | dois defeitos do `diffusers` 0.23 ao montar e rodar o Versatile Diffusion (argumentos que o `text_unet` e o `DualTransformer2DModel` não aceitam); os dois só afetam o ramo de texto, que o MindEye1 pesa com zero, ou um argumento que ele não usa |
| `Train_MindEye.py`, `train_autoencoder.py` | retomada: o alto nível recomeçava na época já concluída (e o `OneCycleLR` estouraria no fim); a do low-level zerava a época e só funcionava com DDP |
| `train_autoencoder.py` | opção com tipo errado na linha de comando era ignorada em silêncio; agora para o script |
| `me1_run.sh` | `MPLBACKEND=Agg`: o `Reconstructions.py` chama `plt.show()` na 1ª imagem, e numa máquina com tela o backend TkAgg trava esperando a janela fechar |

Para testar o treino sem baixar os 39 GB: `tools/me1_fake_data.py <pasta>` cria shards sintéticos
no formato real (ruído; só servem para medir memória e tempo), e
`ME1_DATA=<pasta> scripts/me1_run.sh train --num_epochs=3 --no-ckpt_saving` roda 3 épocas
(o `OneCycleLR` exige pelo menos 3).

## No benchmark: 1 e 40 sessões

Para a comparação com o MindEye2 ridge-only e o FRR ser justa, o MindEye1 é treinado e testado
nos **mesmos dados** deles, e não no `webdataset_avg_split`:

| | No benchmark | No MindEye1 original |
|---|---|---|
| Treino | as N primeiras sessões do subj01: 1 sessão = 688 exibições de 536 imagens; 40 sessões = 27.000 exibições de 9.000 imagens | 8.859 imagens, 40 sessões |
| Teste | as 1.000 imagens do teste novo, 3 repetições promediadas (o mesmo `all_images.pt` dos outros modelos) | 982 imagens |
| Voxels | `betas_all_subj01_fp32_renorm.hdf5` do MindEye2 (15.724 do `nsdgeneral`) | os do webdataset, normalizados por eles |
| Imagens | `coco_images_224_float16.hdf5` (224 × 224) | as do webdataset |

`mindeye1/src/nsd_benchmark.py` lê esses dados com o mesmo código do FRR (`mindeye_ridge.nsd_data`) e
os entrega no formato do MindEye1: uma amostra por imagem, com 3 repetições; quando a imagem tem
menos de 3 exibições nas sessões usadas (com 1 sessão, 413 das 536 têm uma só), as que existem se
repetem até completar 3, como o `recon_inference.py` do MindEye2 faz no teste. A opção é
`--num_sessions N` no `Train_MindEye.py` e no `train_autoencoder.py`, e `--benchmark` no
`Reconstructions.py`. Os modelos publicados reconstroem bem a partir desses betas (conferido em 4
imagens), então os dados são compatíveis com o modelo.

A receita do MindEye1 fica como está: 240 épocas no alto nível e 120 no baixo, mesma taxa de
aprendizado, aumentos e perdas, com 1 e com 40 sessões, como o MindEye2 ridge-only também usa a sua
receita nos dois casos. O que muda, e por quê:

| | Benchmark | Artigo | Por quê |
|---|---|---|---|
| Batch e otimizador do alto nível | 16, AdamW 8 bits | 32, AdamW | não cabe em 20 GB (acima) |
| Checkpoint avaliado | o da última época | o `best.pth`, escolhido pela perda no conjunto de teste | é o que o benchmark usa em todos os modelos; escolher pelo teste vazaria o teste |
| Amostras por imagem | 1 | 16, com escolha pela similaridade CLIP | o MindEye2 do benchmark também gera 1 |
| Alvos do low-level | latente do VAE e embedding do ConvNeXt da imagem limpa calculados uma vez por imagem | calculados a cada passo | mesmas operações, mesmos números; 40 sessões cairiam de ~42 h para ~13 h |

**Avaliação.** As reconstruções (img2img a partir da imagem borrada, força 0,85, como no
`Reconstructions.py`) passam pelo mesmo `final_evaluations.py` dos outros modelos, em 256 × 256, e
são avaliadas como "base": a imagem borrada já entra pelo img2img, então não há a mistura 75/25 que
o MindEye2 faz nas refinadas. O retrieval usa o mesmo sorteio (30 × 300) no espaço CLIP do próprio
modelo, como o de cada modelo do benchmark: a saída do projetor contra o CLIP ViT-L/14 da imagem
(257 × 768), que o `Reconstructions.py --benchmark` grava em `<modelo>_all_clipimages.pt`. Não há
legendas previstas, então as métricas de legenda ficam de fora.

```bash
scripts/run_me1_benchmark.sh               # 1 sessão e depois 40 (~37 h numa A4500)
SESSOES=1 scripts/run_me1_benchmark.sh     # só 1 sessão (~5 h)
NUM_SESSIONS=40 scripts/me1_run.sh train   # uma etapa só
```

O pipeline retoma de onde parou (marcadores `.completo`, `last.pth`) e escreve uma linha por etapa
em `logs/me1_benchmark.log`. Saídas: `mindeye1/train_logs/subj01_me1_<N>sess/` (alto nível),
`mindeye1/train_logs/models/subj01_me1_lowlevel_<N>sess/` (baixo nível),
`results/evals/subj01_me1_<N>sess/` (reconstruções e embeddings) e
`results/tables/subj01_me1_<N>sess_all_recons.csv` (métricas).
