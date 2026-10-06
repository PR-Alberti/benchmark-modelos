# O que foi feito

Fine-tune do MindEye2 no sujeito 1 do NSD com 1 sessão de dados, treinando
**apenas a camada ridge** e mantendo backbone e diffusion prior congelados.

## Onde a ridge entra

```
fMRI do subj01 — 15.724 voxels
        │
        ▼
   ridge          Linear(15724 → 1024)          16,1M params   POR SUJEITO
        │
        ▼
   backbone       BrainNetwork, 4 blocos       453,4M params   COMPARTILHADO
        │         saída: embeddings CLIP (256×1664)
        ▼
   diffusion_prior                             259,9M params   COMPARTILHADO
        │
        ▼
   unCLIP SDXL → imagem                        (sempre congelado)
                                               ──────────────
                                                729,3M params
```

Cada pessoa tem um número e um arranjo diferente de voxels, então a ridge é
necessariamente individual. Backbone e prior são o conhecimento compartilhado —
é daí que vem o subtítulo do paper, *Shared-Subject Models*.

## Como o paper treina

Duas fases. Pré-treino em 7 sujeitos (todos menos o alvo), 40 sessões cada:

```bash
accelerate launch Train.py --model_name=final_multisubject_subj01 \
    --multi_subject --subj=1 --batch_size=42 --hidden_dim=4096 \
    --use_prior --blurry_recon --num_epochs=150 --num_sessions=40
```

Depois, fine-tune no sujeito alvo trocando a ridge e **retreinando os 729M**:

```bash
accelerate launch Train.py --model_name=final_subj01_pretrained_1sess_24bs \
    --no-multi_subject --subj=1 --batch_size=24 --hidden_dim=4096 \
    --use_prior --blurry_recon --num_sessions=1 \
    --multisubject_ckpt=../train_logs/final_multisubject_subj01
```

8×A100 de 80 GB, cerca de 1 dia.

## Como este treina

```bash
./scripts/run_ridgeonly_prior.sh
```

Que roda:

```bash
python src/train_ridgeonly.py --ridge_only \
    --multisubject_ckpt=$DATA/train_logs/multisubject_subj01_1024hid_nolow_300ep \
    --subj=1 --num_sessions=1 --batch_size=16 \
    --hidden_dim=1024 --use_prior --prior_scale=30 --no-blurry_recon \
    --num_epochs=150 --embedder_fp16 --new_test
```

1×RTX A4500 de 20 GB, ~45 s por época, ~1h50 no total.

## O que mudou no código

O `train_ridgeonly.py` mantém as 26 flags originais do `Train.ipynb` e acrescenta
três: `--ridge_only`, `--embedder_fp16` (ViT-bigG em fp16, libera ~3,6 GB) e
`--metrics_csv`.

O núcleo é este bloco:

```python
if ridge_only:
    model.backbone.requires_grad_(False)              # 453M congelados
    if use_prior:
        model.diffusion_prior.requires_grad_(False)   # 260M congelados
    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()],
         'weight_decay': 1e-2},                       # otimizador só vê a ridge
    ]
```

Treináveis: **16.102.400 de 729.327.896 — 2,21%**. É exatamente
`15724 × 1024 + 1024`, a matriz da ridge do subj01 mais o viés.

`git diff --stat origin/upstream-base..main` mostra o conjunto todo das mudanças.

## Por que congelar era necessário aqui

A ridge é a **primeira** camada, então o gradiente ainda atravessa backbone e
prior de volta para chegar nela — congelar não elimina o backward. O que
economiza é:

- gradientes dos 713M congelados: ~2,9 GB
- estado do AdamW (dois momentos por parâmetro): ~5,7 GB

Cerca de 8,5 GB, e é o que faz caber em 20 GB. O fine-tune completo foi tentado e
não roda nesta placa:

| Tentativa | Batch | Resultado |
|---|---|---|
| `probe_fulltune` | 16 | CUDA out of memory |
| `probe_b8` | 8 | CUDA out of memory |
| `probe_b4` | 4 | CUDA out of memory |
| ridge-only | 16 | 150 épocas |

## Resultados

> **Revisado no benchmark.** Estes são os números da primeira rodada. O
> [benchmark completo](#benchmark-completo), no fim deste arquivo, refaz tudo com
> o mesmo pipeline e corrige duas coisas: a coluna do artigo não aplicava a
> mistura de 25% blurry, e a linha do ridge-only refinada veio de uma geração
> anterior das reconstruções.

Retrieval top-1 entre 300 candidatos, acaso = 0,33%:

| | ridge-only | modelo do paper |
|---|---|---|
| Forward | 92,8% | 93,9% |
| Backward | 88,1% | 77,6% |

> **Sentido.** No código, *forward* é imagem → cérebro (cada imagem procura o seu cérebro entre
> as 300 previsões) e *backward* é cérebro → imagem. É o oposto do comentário do
> `final_evaluations.py` e do que o artigo define como Image Retrieval; veja
> [a seção do FRR](#três-coisas-que-apareceram-pelo-caminho).

Reconstrução:

| Métrica | ridge-only | paper | nível |
|---|---|---|---|
| PixCorr | 0,182 | 0,202 | baixo |
| SSIM | 0,347 | 0,391 | baixo |
| AlexNet(2) | 0,853 | 0,883 | baixo |
| AlexNet(5) | 0,923 | 0,934 | baixo |
| InceptionV3 | 0,843 | 0,834 | alto |
| CLIP | 0,827 | 0,810 | alto |
| EffNet-B | 0,802 | 0,798 | alto |
| SwAV | 0,459 | 0,458 | alto |

Padrão: perde nas métricas de baixo nível, empata ou ganha nas de alto nível,
treinando 2,21% dos parâmetros.

## Comparação visual

`results/figs/subj01_ridgeonly_1sess_prior_enhanced_comparacao.png` e a versão `_base_`
mostram, lado a lado, a imagem que o participante viu, a reconstrução deste
repositório e a publicada no artigo, para 12 exemplos sorteados com semente fixa.

```bash
python src/make_comparison.py --model subj01_ridgeonly_1sess_prior
python src/make_comparison.py --model subj01_ridgeonly_1sess_prior --kind base --n 18 --seed 3
```

As colunas são exibidas todas na menor resolução entre elas (224×224): as nossas
reconstruções são salvas em 256 e as publicadas em 512, e mostrar cada uma na sua
faria a de menor resolução parecer pior só pela suavização.

O alinhamento dos índices foi verificado, não presumido: PixCorr entre imagem e
reconstrução dá 0,185 nos pares corretos contra 0,031 embaralhando — e 0,220
contra 0,047 para o modelo do artigo.

Olhando as figuras: as reconstruções acertam com frequência o tipo de cena e a
composição geral (animais num campo, avião no céu, ambiente interno), e erram o
conteúdo específico. É o resultado esperado com 1 hora de dados — as métricas de
alto nível ficam em torno de 0,8 justamente por isso.

## Ressalva sobre a comparação

**Não é um ablation limpo.** O modelo do paper difere em mais de uma variável:

| | este | paper |
|---|---|---|
| `hidden_dim` | 1024 | 4096 |
| `blurry_recon` | desligado | ligado |
| checkpoint de partida | `multisubject_..._1024hid_nolow` | `final_multisubject_subj01` |

O `blurry_recon` é justamente o submódulo de baixo nível, então a perda em
PixCorr e SSIM provavelmente vem dele estar desligado, não do congelamento. Para
isolar o efeito do ridge-only faltaria rodar o fine-tune completo na mesma
configuração 1024/nolow — que é o que os probes tentaram e a GPU não permitiu.

## A comparação pareada 4096+blurry: cabe, com duas mudanças

A ressalva acima motivou testar a configuração que torna o ablation limpo —
mesmo `hidden_dim`, mesmo `blurry_recon`, mesmo checkpoint de partida que o
paper, diferindo **só** no que está congelado. O modelo 4096 tem 2.227.294.844
parâmetros (3× o de 1024), dos quais a ridge são 64.409.600 (2,89%).

Do jeito que estava, não cabia — e reduzir o batch não resolvia:

| Batch | Memória do processo no OOM | Onde estourou |
|---|---|---|
| 16 | 18,38 GB | forward do prior |
| 8 | 18,55 GB | forward do prior |
| 4 | 18,52 GB | forward do prior |

Três batches, mesma memória: o gargalo era **estático**, não ativações. A causa é
que o AMP (`mixed_precision="fp16"`) mantém pesos-mestre em fp32 para poder
atualizá-los — mas aqui 97% dos pesos estão congelados e nunca são atualizados.

**`--frozen_fp16`** guarda `backbone` e `diffusion_prior` em fp16 (as camadas de
normalização ficam em fp32, que é a precisão em que o autocast as executa). Os
pesos caem de 8,30 GB para 4,27 GB, e o estático na placa de 15.146 MiB para
11.060 MiB. A flag exige `--ridge_only`: sem ele o backbone treina e perderia a
precisão que o otimizador precisa.

Só isso ainda não bastava: em batch 8 o laço de teste decodificava 60 imagens de
uma vez pelo VAE e pedia 2,87 GiB num único bloco. **`decode_em_blocos`** fatia
esse decode. É métrica sob `no_grad` e o `pixcorr` correlaciona amostra a
amostra, então o resultado é idêntico — verificado: as três épocas de sonda deram
os mesmos números com e sem o fatiamento.

Com as duas mudanças, em batch 8, medindo `torch.cuda.max_memory_allocated`:

| Ponto do laço de teste | Alocado | Pico acumulado |
|---|---|---|
| antes do embedder | 10.706 MiB | 15.824 MiB |
| depois do embedder | 10.943 MiB | 15.824 MiB |
| depois das 3 reps do backbone | 12.047 MiB | 15.824 MiB |
| **depois do prior** | 13.178 MiB | **18.055 MiB** |

Cabe nos 20.470 MiB, com ~1,4 GB de folga contando os ~900 MiB do desktop. O pico
é a chamada do prior sobre 60 amostras no teste; se faltar margem, é ali que se
mexe. Ligar o monitor na iGPU devolve os ~900 MiB.

Medir com `max_memory_allocated`, não com `nvidia-smi`: o alocador reserva
oportunisticamente, então o pico do `nvidia-smi` (19,9 GB) engana.

Custo: ~73 s por época, contra ~44 s do 1024 em batch 16 — as 150 épocas dão
~2 h 40. O batch cai de 16 para 8, então essa configuração não é comparável ao
braço de 1024 em dinâmica de treino; é comparável ao **modelo do paper**.

```bash
BLURRY=1 HIDDEN_DIM=4096 MODEL_NAME=subj01_ridgeonly_1sess_4096blurry \
    scripts/run_ridgeonly_prior.sh
```

## Escalar para 40 sessões: o pré-carregamento estoura a RAM

Com 40 sessões o treino morria em ~1 minuto, em silêncio, sem traceback e sem
erro de CUDA. Não era a GPU: era o **OOM killer do kernel**.

O `Train.py` original pré-carrega a época inteira em memória:

```python
image_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), 3, 224, 224).float()
```

Com 1 sessão são 46 iterações (443 MB, irrelevante). Com 40 sessões são 1.875
iterações: **18,1 GB nesse único tensor**. O processo chegava a 25,8 GB de RSS
numa máquina de 30 GB.

O `.float()` era desperdício: a fonte é o `coco_images_224_float16.hdf5`, a linha
seguinte já cria o tensor em fp16, e o consumo roda sob autocast fp16 — armazenar
em fp32 no meio era uma ida e volta sem função. Guardando em `data_type`, o
tensor cai para 9,0 GB.

Medido: RSS de 25,8 GB para **17,4 GB**, memória disponível de 152 MB para 8,6 GB.

Esse bug é latente no código original e só aparece ao escalar o número de
sessões — com a configuração de 1 sessão do paper ele nunca se manifesta.

```bash
NUM_SESSIONS=40 NUM_EPOCHS=20 CKPT_INTERVAL=2 \
    MODEL_NAME=subj01_ridgeonly_40sess_prior scripts/run_ridgeonly_prior.sh
```

### Resultado do 4096+blurry

150 épocas em ~2 h 40, batch 8. Retrieval top-1 entre 300 candidatos:

| | ridge-only 1024 | ridge-only 4096+blurry | modelo do paper |
|---|---|---|---|
| Forward | 92,8% | **94,3%** | 93,9% |
| Backward | 88,1% | **91,3%** | 77,6% |

Números da última época; melhor fwd ao longo da corrida foi 95,3%, e a média das
dez últimas épocas ficou em 94,3% / 91,4% — ou seja, convergiu e estabilizou.

Esta configuração foi treinada duas vezes (a primeira corrida se perdeu num
reset de máquina antes de ser salva). A repetição serve de medida de variância:

| | corrida 1 | corrida 2 |
|---|---|---|
| Forward | 95,7% | 94,3% |
| Backward | 92,3% | 91,3% |

Cerca de 1 ponto percentual entre corridas, o que vem do `blur_augs` ser
estocástico e do arredondamento em fp16. O ganho sobre o braço 1024 (+1,5 em
forward, +3,2 em backward) é maior que essa variância no backward, e da mesma
ordem no forward — então a leitura segura é que o 4096+blurry ajuda claramente no
backward e provavelmente um pouco no forward.

> **Correção (benchmark).** Com o mesmo código e a mesma semente o treino é
> determinístico: uma terceira corrida repetiu a segunda época por época, em todas
> as colunas do `metrics.csv`. A diferença entre as corridas 1 e 2 veio de uma
> mudança de código entre elas (o conserto do ColorJitter em fp16, `cef48ae`,
> entrou pouco antes da corrida 2), não de ruído. Trocando só a semente, o
> retrieval do 1024 + prior varia ~0,3 ponto; no protocolo das tabelas (média de
> 30 sorteios) o ganho do 4096 + blurry sobre o 1024 é de +2,7 em imagem e +3,9 em
> cérebro, bem acima disso.

O ponto que importa: com `hidden_dim` e `blurry_recon` iguais aos do paper e o
mesmo checkpoint de partida, treinar **só 2,89% dos parâmetros** fica no mesmo
patamar do fine-tune completo em forward e bem à frente em backward. O
congelamento não impede o modelo de aproveitar nem a capacidade maior nem o ramo
de baixo nível.

## Benchmark completo

Em 30/09/2026 os quatro braços ridge-only e os dois modelos publicados no artigo
(1 e 40 sessões) passaram pelo mesmo pipeline, com a mesma semente:
reconstrução, refinamento e as métricas do `final_evaluations.py`. Tabelas em
[BENCHMARK.md](../BENCHMARK.md); galeria de reconstruções lado a lado, curvas de
treino e ruído em `benchmark/index.html`. Para refazer tudo:
`./scripts/run_benchmark.sh` (~50 h numa A4500).

Reconstruções refinadas, com 25% de blurry misturado nos modelos que têm o ramo,
como no artigo:

| | PixCorr | SSIM | Alex(2) | Alex(5) | Incep | CLIP | Eff ↓ | SwAV ↓ | Imagem | Cérebro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ridge 1024 + prior | 0,181 | 0,346 | 85,4% | 92,5% | 83,6% | 82,6% | 0,802 | 0,460 | 92,8% | 88,1% |
| ridge 4096 + blurry | 0,253 | 0,414 | 89,6% | 94,6% | 85,2% | 83,5% | 0,785 | 0,443 | 95,5% | 92,0% |
| ridge 1024 sem prior | 0,111 | 0,310 | 75,0% | 83,8% | 72,8% | 72,6% | 0,889 | 0,525 | 78,5% | 72,1% |
| artigo, 1 sessão | 0,237 | 0,421 | 87,6% | 93,1% | 83,0% | 81,7% | 0,804 | 0,457 | 93,9% | 77,6% |
| ridge 1024 + prior, 40 sessões | 0,277 | 0,383 | 94,5% | 98,2% | 94,7% | 92,3% | 0,645 | 0,364 | 99,9% | 99,5% |
| artigo, 40 sessões | 0,373 | 0,432 | 97,5% | 99,2% | 96,0% | 93,6% | 0,606 | 0,331 | 100,0% | 99,9% |

Rodado aqui, o modelo do artigo fica a até 1 ponto das reconstruções que os
autores publicaram (PixCorr 0,237 contra 0,235 em 1 sessão; 0,373 contra 0,374
em 40), então o pipeline reproduz o artigo.

### O que muda em relação ao que está acima

- **O artigo estava subestimado no baixo nível.** A coluna do artigo em
  [Resultados](#resultados) avaliava as reconstruções publicadas sem a mistura de
  25% blurry e sem as legendas previstas, que não estavam disponíveis. Com a
  mistura, as mesmas reconstruções vão de PixCorr 0,202 para 0,235 e de SSIM
  0,391 para 0,428. O 1024 perde mais no baixo nível do que parecia.
- **A linha refinada do 1024 veio de outra geração de reconstruções.**
  Recalculada sobre o tensor do release: PixCorr 0,181 (antes 0,182), Incep
  83,6% (antes 84,3%). As métricas das reconstruções unCLIP saíram idênticas às
  da tabela antiga, byte a byte.
- **A diferença entre corridas não era ruído** (correção na seção do 4096 +
  blurry). A régua de ruído agora é a troca de semente: no 1024 + prior o
  retrieval varia menos de 0,5 ponto entre as sementes 42, 1 e 2, e reamostrar a
  reconstrução move as identificações 2-way até 0,6 ponto e o PixCorr até 0,006.
  O efeito da semente do treino nas métricas de imagem não foi medido.
- **Treinar a ridge sem o prior não é neutro.** O `SETUP.md` dizia que tirar o
  prior não afetava o retrieval. Com o mesmo script e a mesma semente, mudando só
  `--use_prior`, o retrieval cai de 92,9% / 88,1% para 78,4% / 72,6%, e todas as
  métricas de reconstrução caem junto. O retrieval nem passa pelo prior; o que
  muda é o sinal de treino da ridge, já que a loss do prior supervisiona a saída
  inteira do backbone (256 × 1.664), e não só a projeção contrastiva.

### O que o benchmark mostra

- **Com 1 sessão e a arquitetura do artigo, treinar só a ridge basta.** O 4096 +
  blurry fica à frente do fine-tune completo em 9 das 10 métricas de
  reconstrução e retrieval; perde só no SSIM (0,414 contra 0,421). Nas
  identificações 2-way a vantagem é de 1,5 a 2,2 pontos, acima do ruído de
  amostragem; no retrieval Cér→img (bwd), 92,0% contra 77,6%.
- **O 1024 empata com o artigo no alto nível** (Incep e CLIP a menos de 1 ponto)
  e perde no baixo nível, onde a dimensão maior e o ramo blurry fazem diferença.
- **Com 40 sessões**, o ridge 1024 chega a 99,9% / 99,5% de retrieval e fica de
  1 a 3 pontos do artigo nas identificações 2-way. A distância maior é no baixo
  nível (PixCorr 0,277 contra 0,373): o artigo de 40 sessões é 4096 com blurry e
  treinou 150 épocas; este é 1024 sem blurry e treinou 20.

### Mudanças no código para o benchmark

| Onde | O quê |
|---|---|
| `train_ridgeonly.py` | `--resume` retoma do `last.pth`; `time/epoch_s` no `metrics.csv` |
| `run_ridgeonly_prior.sh` | `RESUME=1`, `SEED`, `PRIOR=0`; falha do python não fica mais mascarada pelo `tee` |
| `recon_inference.py` | `--frozen_ckpt` (backbone e prior do ckpt de partida), `--backbone_fp16` (4096 em 20 GB), parcial a cada 100 imagens |
| `run_recon.sh` | `FROZEN_CKPT`, `BACKBONE_FP16`, `SEED`; pula o que já existe |
| `run_evals.sh` | `published` / `published-base` avaliam as reconstruções dos autores; pula tabela pronta (`FORCE=1` recalcula) |
| `verify_retrieval.py` | caminho do `all_images.pt`, `--frozen_ckpt`, resultado em JSON |
| `download_data.py` | etapa `paper40` |
| `run_benchmark.sh`, `make_benchmark.py` | novos: rodam e montam o benchmark |

As tabelas da primeira rodada estão em `results/tables/pre_benchmark/`.

## FRR: o baseline linear do plano de estágio

O plano de estágio pede, entre as famílias de modelos, "um modelo linear por sujeito, provavelmente
Fractional Ridge Regression (FRR), seguindo Doerig et al.". É o ponto mais simples da escala de
complexidade: sem rede, sem prior, sem reconstrução, e com a mesma entrada, o mesmo treino e o mesmo
teste do ridge-only. Os números completos estão no [BENCHMARK.md](../BENCHMARK.md) e o que falta do
plano, no [ESTAGIO.md](ESTAGIO.md).

### O modelo

Regressão ridge fracionária (Rokem & Kay, 2020) dos 15.724 voxels do nsdgeneral para o embedding
CLIP ViT-bigG/14 achatado (256 × 1.664 = 425.984 números). Na ridge comum o α só tem sentido para um
X e um y; a FRR o troca pela fração γ = |β(α)| / |β_OLS| da norma da solução sem regularização, que
significa o mesmo para qualquer alvo, e acha o α de cada γ, em cada dimensão do alvo, na
decomposição espectral dos voxels. Como em Doerig et al. (2025): 20 frações de 0,05 a 1, validação
cruzada de 5 dobras e **uma fração para cada dimensão** do alvo.

A matriz de coeficientes teria 6,70 bilhões de entradas (27 GB em fp32), então nada é
materializado. A implementação própria (`src/mindeye_ridge/frr.py`) diagonaliza a matriz de Gram dos
voxels em fp64 na CPU e faz as contas por dimensão do alvo em blocos de 8.192 colunas na GPU; a
escolha da fração por validação cruzada reaproveita a mesma decomposição, e as dobras ficam em
disco, para retomar uma corrida interrompida. Foi conferida contra o pacote `fracridge` de
referência (mesmos α e mesmos coeficientes), contra uma ridge resolvida pela equação normal (nos
regimes n < p e n > p) e, nos dados reais, contra uma `sklearn.Ridge` com α achado por bisecção:
correlação ≥ 0,99999 em 40 dimensões sorteadas (`tests/test_frr.py`).

### Condições (as do ridge-only)

| | |
|---|---|
| Sujeito e entrada | subj01, 15.724 voxels do nsdgeneral. O FRR lê os betas em fp32; o treino do MindEye2 os lê em fp16 |
| Treino | as N primeiras sessões, sem nenhuma imagem do teste (conferido): 1 sessão = 688 exibições de 536 imagens; 40 = 27.000 de 9.000 |
| Teste | as mesmas 1.000 imagens × 3 repetições; o voxel de cada imagem é a média das 3 |
| Alvo | o embedding CLIP achatado. De treino, com a imagem em fp16, como no treino do MindEye2; de avaliação, com a imagem em fp32, como no `verify_retrieval.py` e no `final_evaluations.py` |
| Métrica | retrieval top-1 entre 300, 30 sorteios, mesma semente e mesma função: reproduz o `verify_retrieval.py` até o sexto dígito (92,9111% / 88,1222%) |
| Hiperparâmetro | a fração de cada dimensão, por validação cruzada nas exibições de treino, agrupada por imagem: nada do teste entra |

### Resultados

Retrieval top-1 entre 300 (Img→cér: cada imagem acha o seu cérebro; Cér→img: cada cérebro acha a
sua imagem). "Centrado" tira a média de treino da previsão e do alvo antes do cosseno.

| | Img→cér | Cér→img | Img→cér centrado | Cér→img centrado | Cosseno centrado |
|---|---:|---:|---:|---:|---:|
| **1 sessão** | | | | | |
| FRR (grade de Doerig, 0,05 a 1) | 55,0% | 3,1% | 56,2% | 31,7% | 0,117 |
| FRR, uma fração para todas as dimensões | 33,2% | 1,3% | 35,8% | 19,4% | 0,101 |
| FRR, grade estendida (0,001 a 1) | 55,7% | 2,0% | 58,4% | 30,7% | 0,111 |
| ridge 1024 + prior | 92,8% | 88,1% | — | — | — |
| artigo, nossa execução | 93,9% | 77,6% | — | — | — |
| **40 sessões** | | | | | |
| FRR (grade de Doerig, 0,05 a 1) | 96,4% | 37,7% | 96,9% | 86,5% | 0,220 |
| FRR, uma fração para todas as dimensões | 96,3% | 37,7% | 96,9% | 86,4% | 0,220 |
| FRR, grade estendida (0,001 a 1) | 92,4% | 21,3% | 95,3% | 77,1% | 0,216 |
| ridge 1024 + prior | 99,9% | 99,5% | — | — | — |
| artigo, nossa execução | 100,0% | 99,9% | — | — | — |

Custo, numa A4500: com 1 sessão, o ajuste leva 10 s (8 s de validação cruzada), 0,6 GiB de GPU e
3 GiB de RAM; com 40, 21 min (18 min de validação cruzada), 3,2 GiB de GPU e 13,6 GiB de RAM, mais
uns 3 min, uma só vez, para calcular e guardar os embeddings-alvo das 9.000 imagens de treino.
Duas corridas de mesma configuração (a de fração única e a principal, com 40 sessões, escolhem as
mesmas frações) diferem em 0,1 ponto de retrieval: é o ruído numérico da diagonalização em fp64.

### O que o FRR mostra

- **Com pouco dado a regressão linear fica muito atrás; com 40 sessões, quase alcança.** Img→cér:
  55,0% com 1 sessão, contra 92,8% do ridge 1024 + prior e 93,9% do artigo; 96,4% com 40, contra
  99,9% e 100,0%. A distância cai de ~38 para ~3,5 pontos, e o ganho de 1 para 40 sessões é de 41
  pontos no FRR, contra 7 no ridge: o modelo linear depende muito mais dos dados.
- **A fração por dimensão sustenta o resultado com pouco dado.** Com 1 sessão, uma fração única
  para todas as dimensões dá 33,2%, contra 55,0%. Com 40 sessões todas as dimensões já escolhem a
  menor fração da grade, então as duas coincidem.
- **O Cér→img bruto é baixíssimo, e é artefato.** 3,1% e 37,7%, contra 88,1% e 99,5% do ridge. As
  previsões lineares ficam encolhidas em direção à média (o cosseno com a média de treino é 0,98
  com 1 sessão e 0,95 com 40) e, no cosseno bruto, a imagem mais parecida com a média vence para
  quase toda previsão. Sem a média de treino, vai a 31,7% e 86,5%.
- **A previsão acrescenta pouco à média, e o suficiente para o ranking.** O cosseno com o embedding
  verdadeiro é de 0,51 e 0,53, mas o verdadeiro e a média de treino já têm cosseno 0,50 entre si;
  tirando a média dos dois, o cosseno é de 0,12 e 0,22. O R² da validação cruzada é de 0,7% e 1,6%
  (otimista: a fração foi escolhida nas mesmas dobras).

### Três coisas que apareceram pelo caminho

1. **O sentido do retrieval.** No código do MindEye2, `fwd` é imagem → cérebro (cada imagem procura
   o seu cérebro entre as 300 previsões) e `bwd` é cérebro → imagem. Foi conferido com um exemplo
   sintético em que a resposta é conhecida por construção, e é o oposto do comentário do
   `final_evaluations.py` (`fwd: brain, clip`) e do que o artigo define como Image Retrieval
   (cérebro → imagem). Nos modelos contrastivos as duas direções diferem de 0 a 16 pontos; no FRR,
   que encolhe as previsões em direção à média, passam de 50. O benchmark passou a rotular as
   colunas pela direção (Img→cér, Cér→img): a versão anterior descrevia "Imagem" como "dado o
   cérebro, achar a imagem", o contrário do que o código calcula. Os números não mudaram.
2. **A grade de frações.** A de Doerig et al. é a que o plano especifica e a principal. No subj01 o
   ótimo do **erro quadrático** da validação cruzada cai abaixo dela (com 1 sessão 86% das dimensões
   escolhem a menor fração, 0,05, e com 40, todas). Estendê-la até 0,001 dobra o R² da validação
   cruzada com 40 sessões (1,6% → 3,2%), mas **piora o retrieval no teste** (96,4% → 92,4%):
   encolher mais aproxima as previsões da média e o ranking se perde. O erro quadrático não é o
   critério do retrieval. A grade principal foi escolhida pelo que o plano especifica, antes de
   ver o teste; a estendida entra como análise de sensibilidade.
3. **A entrada do embedder CLIP.** O ViT-bigG dá saídas diferentes para a mesma imagem em fp16 e
   em fp32 (cosseno 0,996 entre as duas versões). O treino alimenta o embedder com a imagem em fp16
   e as avaliações, com a imagem em fp32; o FRR segue as duas convenções, como os outros modelos.

### O que mudou no código

| Onde | O quê |
|---|---|
| `frr.py` | novo: a FRR (Rokem & Kay), em GPU e por blocos de alvo; validação cruzada agrupada por imagem, com dobras em disco e retomada |
| `nsd_data.py`, `clip_targets.py` | novos: leitura das exibições do NSD e cache em disco dos embeddings CLIP de treino |
| `embedding_metrics.py` | novo: o retrieval top-1 do benchmark (mesma função e mesma semente do `verify_retrieval.py`) e as similaridades com o alvo, brutas e centradas |
| `run_frr.py`, `run_frr.sh` | novos: rodam o FRR de um sujeito e N sessões e gravam `retrieval.json` e `frr.json` |
| `make_benchmark.py`, `run_benchmark.sh` | o FRR entra na tabela e ganha seção própria; colunas de retrieval rotuladas pela direção |
| `tests/` | novos: 26 testes do FRR (contra o `fracridge`, contra a equação normal, bordas) e 23 da página do benchmark |
