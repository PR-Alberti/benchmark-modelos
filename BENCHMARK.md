# Benchmark ridge-only

MindEye2, sujeito 1 do NSD. Cinco variações do fine-tune só da camada ridge, comparadas com os modelos publicados no artigo (fine-tune completo), com um baseline linear (FRR) que não usa rede nenhuma e com o MindEye1 treinado do zero. Mesmo teste (1.000 imagens), mesmo pipeline de reconstrução e mesmas métricas para todos.

A versão completa, com a galeria de reconstruções, curvas de treino e seletor refinada/unCLIP, está em [results/benchmark/index.html](results/benchmark/index.html) — um arquivo só, abre offline.

Gerado por `src/make_benchmark.py` em 13/11/2026 07:07 (commit `e2c37c7`).

## Modelos

| Modelo | Sessões | hidden_dim | Blurry | Parâmetros treinados | Treino | Tempo (A4500) |
|---|---:|---:|---|---|---|---|
| FRR · 1 sessão | 1 | — | não | 6,70 bi de coeficientes (forma fechada) | 20 frações · CV 5 dobras · uma fração por dimensão | 10 s |
| FRR · 40 sessões | 40 | — | não | 6,70 bi de coeficientes (forma fechada) | 20 frações · CV 5 dobras · uma fração por dimensão | 21 min |
| MindEye1 · 1 sessão | 1 | 4096 | sim | todos, do zero | 240 épocas + 120 do baixo nível · batch 16 · Adam 8 bits | — |
| MindEye1 · 40 sessões | 40 | 4096 | sim | todos, do zero | 240 épocas + 120 do baixo nível · batch 16 · Adam 8 bits | — |
| MindEye2 sem low-level | 1 | 1024 | não | — | 150 épocas · batch 16 | ~1 h 50 (corrida anterior) |
| MindEye2 | 1 | 4096 | sim | — | 150 épocas · batch 8 · congelados em fp16 | — |
| MindEye2 sem low-level nem prior | 1 | 1024 | não | — | 150 épocas · batch 16 | — |
| MindEye2 sem low-level · 40 sessões | 40 | 1024 | não | — | 20 épocas · batch 16 · congelados em fp16 | — |
| MindEye2 · 40 sessões | 40 | 4096 | sim | — | 20 épocas · batch 8 · congelados em fp16 | — |
| Paper · 1 sessão | 1 | 4096 | sim | — | 150 épocas · batch 24 · 8×A100 | — |
| Paper · 40 sessões | 40 | 4096 | sim | — | 150 épocas · batch 24 · 8×A100 | — |

## Resumo

- **1 sessão, mesma arquitetura do artigo.** O MindEye2 fica à frente do fine-tune completo em 9 das 10 métricas de reconstrução e retrieval (perde só no SSIM: 0,414 contra 0,421), treinando 2,89% dos parâmetros numa GPU de 20 GB. Nas identificações 2-way a vantagem é de 1,5 a 2,2 pontos, acima dos até 0,6 ponto que mudam só por reamostrar a reconstrução. No retrieval Cér→img a distância é grande: 92,0% contra 77,6%.
- **Sem low-level contra com low-level.** A dimensão maior (4096 contra 1024) e o ramo blurry pesam sobretudo no baixo nível (PixCorr 0,180 → 0,253; Alex(2) 85,4% → 89,6%). No alto nível o MindEye2 sem low-level já empata com o artigo: Incep e CLIP a 1,3 e 0,7 ponto, Eff e SwAV a 0,001 e 0,003.
- **O prior importa no treino da ridge.** Com o mesmo script e a mesma semente, tirar a loss do prior derruba o retrieval (Img→cér / Cér→img) de 92,8% / 88,1% para 78,3% / 72,1%, e todas as métricas de reconstrução caem junto. O retrieval nem passa pelo prior: o que muda é o sinal de treino, já que a loss do prior supervisiona a saída inteira do backbone (256 × 1.664), e não só a projeção contrastiva.
- **40 sessões.** O MindEye2 sem low-level chega a 99,9% / 99,4% de retrieval e fica de 1 a 3 pontos do artigo nas identificações 2-way (CLIP 92,1% contra 93,6%). A distância maior é no baixo nível (PixCorr 0,277 contra 0,373), onde o artigo tem o ramo blurry e este não.
- **MindEye1, treinado do zero.** Sem pré-treino, com 1 sessão ele quase não recupera (retrieval 30,6% / 23,2%, contra 92,8% / 88,1% do MindEye2 sem low-level) e fica atrás do MindEye2 nas identificações 2-way (CLIP 80,8% contra 83,8%, na tabela unCLIP), embora o PixCorr (0,252) seja maior. Com 40 sessões o retrieval sobe 64 pontos (94,5% / 87,6%) e o PixCorr chega a 0,382, acima do artigo reconstruído aqui (0,373 refinada, 0,324 unCLIP). O retrieval do MindEye1 é no espaço ViT-L/14 dele, não no bigG dos outros, então não é diretamente comparável.
- **Ruído.** Trocar a semente do treino move o retrieval em menos de 0,5 ponto; reamostrar a reconstrução move as identificações 2-way em até 0,6 ponto e o PixCorr em até 0,006. O efeito da semente do treino nas métricas de imagem não foi medido.
- **O pipeline reproduz o artigo.** Reconstruídos aqui, os dois modelos publicados ficam a até 1 ponto dos tensores divulgados pelos autores nas identificações 2-way, e a até 0,007 em PixCorr e SSIM.
- **FRR, o baseline linear.** Uma regressão linear dos voxels para o embedding CLIP, sem rede nenhuma, acha o cérebro certo de cada imagem (Img→cér) em 55,0% dos casos com 1 sessão e 96,4% com 40, contra 92,8% e 99,9% do MindEye2 sem low-level: com poucos dados falta muito, e o ganho de 1 para 40 sessões é de 41 pontos no FRR e 7 no MindEye2 sem low-level. Escolher uma fração para cada dimensão do alvo é o que sustenta o resultado com pouco dado: com uma fração só para todas, o de 1 sessão cai de 55,0% para 33,2%; com 40 sessões as duas coincidem.
- **FRR e a média.** No sentido Cér→img o cosseno bruto dá só 3,1% e 37,7%, artefato de previsões encolhidas em direção à média (hubness); tirando a média de treino sobe para 31,7% e 86,5%. O embedding previsto também acrescenta pouco a ela: o cosseno com o verdadeiro vale ~0,5, quase todo pela média, e centrado fica em 0,12 e 0,22. O R² de validação cruzada é de 0,7% e 1,6% (otimista, a fração foi escolhida nas mesmas dobras), mas o retrieval só precisa do ranking.
- **FRR e a grade de frações.** Com 40 sessões, estender a grade até 0,001 (o ótimo do erro quadrático cai abaixo da de Doerig) leva o R² de validação cruzada de 1,6% para 3,2% e piora o Img→cér de 96,4% para 92,3%: o erro quadrático não é o critério do retrieval.

## Reconstrução e retrieval — refinadas

| Modelo | PixCorr ↑ | SSIM ↑ | Alex(2) ↑ | Alex(5) ↑ | Incep ↑ | CLIP ↑ | Eff ↓ | SwAV ↓ | Img→cér ↑ | Cér→img ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 sessão · ~1 h de fMRI, 750 exibições** | | | | | | | | | | |
| FRR · 1 sessão (embedding do FRR direto no unCLIP) | 0,079 | **0,474** | 51,9% | 52,1% | 52,6% | 51,6% | 0,972 | 0,615 | 55,0% | 3,1% |
| MindEye1 · 1 sessão (sem refinamento: a mesma reconstrução nas duas abas) | 0,252 | 0,343 | 87,0% | 89,5% | 81,2% | 80,8% | 0,810 | 0,465 | 30,6% | 23,2% |
| MindEye2 sem low-level | 0,180 | 0,348 | 85,4% | 92,8% | 84,3% | 82,4% | 0,805 | 0,460 | 92,8% | 88,1% |
| MindEye2 | **0,253** | 0,414 | **89,6%** | **94,6%** | **85,2%** | **83,5%** | **0,785** | **0,443** | **95,5%** | **92,0%** |
| MindEye2 sem low-level nem prior | 0,111 | 0,310 | 74,7% | 84,2% | 74,1% | 72,4% | 0,889 | 0,525 | 78,3% | 72,1% |
| Paper · 1 sessão (nossa execução) | 0,237 | 0,421 | 87,6% | 93,1% | 83,0% | 81,7% | 0,804 | 0,457 | 93,9% | 77,6% |
| ↳ reconstruções publicadas pelos autores | 0,235 | 0,428 | 88,0% | 93,3% | 83,6% | 80,7% | 0,798 | 0,459 | 93,9% | 77,6% |
| **40 sessões · ~40 h de fMRI, 30.000 exibições** | | | | | | | | | | |
| FRR · 40 sessões (embedding do FRR direto no unCLIP) | 0,248 | **0,475** | 60,8% | 63,7% | 63,3% | 61,2% | 0,939 | 0,586 | 96,4% | 37,7% |
| MindEye1 · 40 sessões (sem refinamento: a mesma reconstrução nas duas abas) | **0,382** | 0,347 | 96,5% | 98,0% | 94,5% | 92,9% | 0,632 | 0,356 | 94,5% | 87,6% |
| MindEye2 sem low-level · 40 sessões | 0,277 | 0,384 | 94,8% | 98,4% | 93,8% | 92,1% | 0,644 | 0,362 | 99,9% | 99,4% |
| MindEye2 · 40 sessões | 0,337 | 0,430 | 95,1% | 98,4% | 94,6% | 92,0% | 0,643 | 0,363 | 99,9% | 99,5% |
| Paper · 40 sessões (nossa execução) | 0,373 | 0,432 | **97,5%** | **99,2%** | **96,0%** | **93,6%** | **0,606** | **0,331** | **100,0%** | **99,9%** |
| ↳ reconstruções publicadas pelos autores | 0,374 | 0,439 | 97,8% | 99,1% | 96,1% | 93,6% | 0,609 | 0,338 | 100,0% | 99,9% |

## Reconstrução — unCLIP (sem refinar)

| Modelo | PixCorr ↑ | SSIM ↑ | Alex(2) ↑ | Alex(5) ↑ | Incep ↑ | CLIP ↑ | Eff ↓ | SwAV ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 sessão · ~1 h de fMRI, 750 exibições** | | | | | | | | |
| FRR · 1 sessão (embedding do FRR direto no unCLIP) | 0,103 | **0,446** | 53,4% | 52,4% | 51,2% | 50,6% | 0,982 | 0,631 |
| MindEye1 · 1 sessão (sem refinamento: a mesma reconstrução nas duas abas) | **0,252** | 0,343 | 87,0% | 89,5% | 81,2% | 80,8% | 0,810 | 0,465 |
| MindEye2 sem low-level | 0,176 | 0,287 | 83,2% | 92,7% | 85,8% | 81,6% | 0,813 | 0,472 |
| MindEye2 | 0,214 | 0,319 | **88,1%** | **94,7%** | **87,4%** | **83,8%** | **0,794** | **0,452** |
| MindEye2 sem low-level nem prior | 0,108 | 0,245 | 71,5% | 83,2% | 74,6% | 69,5% | 0,897 | 0,539 |
| Paper · 1 sessão (nossa execução) | 0,201 | 0,337 | 86,8% | 93,9% | 84,1% | 82,4% | 0,808 | 0,454 |
| ↳ reconstruções publicadas pelos autores | — | — | — | — | — | — | — | — |
| **40 sessões · ~40 h de fMRI, 30.000 exibições** | | | | | | | | |
| FRR · 40 sessões (embedding do FRR direto no unCLIP) | 0,278 | **0,451** | 62,4% | 60,3% | 57,2% | 56,5% | 0,963 | 0,605 |
| MindEye1 · 40 sessões (sem refinamento: a mesma reconstrução nas duas abas) | **0,382** | 0,347 | 96,5% | 98,0% | 94,5% | 92,9% | 0,632 | 0,356 |
| MindEye2 sem low-level · 40 sessões | 0,271 | 0,332 | 93,7% | 98,5% | 95,3% | 92,0% | 0,670 | 0,382 |
| MindEye2 · 40 sessões | 0,289 | 0,335 | 94,5% | 98,6% | 95,1% | 92,6% | 0,667 | 0,373 |
| Paper · 40 sessões (nossa execução) | 0,324 | 0,332 | **96,9%** | **99,4%** | **96,7%** | **95,1%** | **0,610** | **0,332** |
| ↳ reconstruções publicadas pelos autores | 0,325 | 0,333 | 97,2% | 99,4% | 96,8% | 94,9% | 0,611 | 0,334 |

## Correlação cerebral (GNet) — refinadas

| Modelo | nsdgeneral ↑ | V1 ↑ | V2 ↑ | V3 ↑ | V4 ↑ | Alto nível ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **1 sessão · ~1 h de fMRI, 750 exibições** | | | | | | |
| FRR · 1 sessão (embedding do FRR direto no unCLIP) | 0,016 | 0,030 | 0,025 | 0,034 | 0,046 | 0,019 |
| MindEye1 · 1 sessão (sem refinamento: a mesma reconstrução nas duas abas) | 0,344 | 0,282 | 0,271 | 0,277 | 0,279 | **0,355** |
| MindEye2 sem low-level | 0,350 | 0,299 | 0,319 | 0,325 | 0,305 | 0,350 |
| MindEye2 | **0,355** | **0,329** | 0,335 | 0,336 | **0,317** | 0,352 |
| MindEye2 sem low-level nem prior | 0,295 | 0,232 | 0,256 | 0,266 | 0,255 | 0,299 |
| Paper · 1 sessão (nossa execução) | 0,350 | 0,320 | **0,339** | **0,344** | 0,316 | 0,347 |
| ↳ reconstruções publicadas pelos autores | 0,347 | 0,318 | 0,337 | 0,341 | 0,316 | 0,345 |
| **40 sessões · ~40 h de fMRI, 30.000 exibições** | | | | | | |
| FRR · 40 sessões (embedding do FRR direto no unCLIP) | 0,065 | 0,102 | 0,097 | 0,100 | 0,095 | 0,056 |
| MindEye1 · 40 sessões (sem refinamento: a mesma reconstrução nas duas abas) | 0,354 | 0,333 | 0,317 | 0,311 | 0,302 | 0,353 |
| MindEye2 sem low-level · 40 sessões | 0,367 | 0,357 | 0,358 | 0,353 | 0,332 | 0,358 |
| MindEye2 · 40 sessões | 0,360 | 0,347 | 0,348 | 0,343 | 0,325 | 0,356 |
| Paper · 40 sessões (nossa execução) | **0,380** | **0,395** | **0,388** | **0,372** | **0,340** | **0,365** |
| ↳ reconstruções publicadas pelos autores | 0,374 | 0,389 | 0,381 | 0,367 | 0,337 | 0,361 |

Negrito: melhor do grupo, sem contar as reconstruções publicadas. Modelos com ramo blurry avaliam 75% refinada + 25% blurry, como no artigo; o MindEye1 não tem refinamento e já usa a borrada no img2img, então a mesma reconstrução entra nas duas tabelas, sem mistura. Cada modelo faz o retrieval no seu espaço CLIP: o MindEye1 no ViT-L/14, os outros no ViT-bigG/14.

Retrieval top-1 entre 300, média de 30 sorteios. **Img→cér** (fwd no código): para cada imagem, achar o seu cérebro entre as 300 previsões. **Cér→img** (bwd): para cada previsão, achar a sua imagem. É o que o código calcula, o oposto do comentário do `final_evaluations.py` e da definição de Image Retrieval no artigo (cérebro → imagem). O FRR só tem retrieval.

## FRR: o baseline linear

Regressão ridge fracionária (Rokem & Kay, 2020) dos 15.724 voxels direto para o embedding CLIP achatado (256 × 1.664 = 425.984 números), seguindo Doerig et al. (2025): validação cruzada de 5 dobras agrupada por imagem e uma fração escolhida para cada dimensão do alvo. Sem rede e sem prior; cosseno, Pearson e cosseno centrado medem o quanto do embedding verdadeiro a regressão recupera. A reconstrução é opcional (`src/treina.py --reconstrucao unclip|refino`): o embedding previsto vai direto para o unCLIP e, no refino, para o SDXL guiado pela legenda que o GIT tira dele. Nenhuma dessas redes viu fMRI, então o que vem do cérebro é só o FRR. Sem ela, só o retrieval se compara com os outros modelos.

| Variante | Img→cér ↑ | Cér→img ↑ | Img→cér centrado ↑ | Cér→img centrado ↑ | Cosseno ↑ | Pearson ↑ | Cosseno centrado ↑ | R² da CV ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 sessão · ~1 h de fMRI, 750 exibições** | | | | | | | | |
| Grade de Doerig: 0,05 a 1 (principal) | 55,0% | 3,1% | 56,2% | 31,7% | 0,506 | 0,490 | 0,117 | 0,007 |
| Uma fração para todas as dimensões | 33,2% | 1,3% | 35,8% | 19,4% | 0,503 | 0,487 | 0,101 | 0,004 |
| Grade estendida: 0,001 a 1 | 55,7% | 2,0% | 58,4% | 30,7% | 0,505 | 0,489 | 0,111 | 0,012 |
| **40 sessões · ~40 h de fMRI, 30.000 exibições** | | | | | | | | |
| Grade de Doerig: 0,05 a 1 (principal) | 96,4% | 37,7% | 96,9% | 86,5% | 0,532 | 0,517 | 0,220 | 0,016 |
| Uma fração para todas as dimensões | 96,3% | 37,7% | 96,9% | 86,4% | 0,532 | 0,517 | 0,220 | 0,016 |
| Grade estendida: 0,001 a 1 | 92,3% | 21,3% | 95,3% | 77,1% | 0,528 | 0,513 | 0,216 | 0,032 |

Centrado: o mesmo retrieval depois de tirar a média de treino da previsão e do alvo. A regressão encolhe as previsões em direção à média e, no cosseno bruto, a imagem mais parecida com a média vence para quase toda previsão (hubness): por isso o Cér→img bruto é tão baixo. A comparação com os outros modelos usa o cosseno bruto, como o artigo; o centrado só separa esse efeito do sinal. Cosseno e Pearson brutos já valem ~0,5 só porque o embedding verdadeiro e a média de treino têm cosseno ~0,5 entre si; o cosseno centrado tira a média dos dois e mede o que a previsão acrescenta a ela. R² da CV: variância do embedding explicada nas dobras de validação, com a fração de cada dimensão; é otimista, porque a fração foi escolhida nessas mesmas dobras. As métricas limpas são as do teste (retrieval, cosseno, Pearson).

| Modelo | Exibições | Imagens | Coeficientes | Validação cruzada | Ajuste + previsão | Pico na GPU | Pico de RAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| FRR · 1 sessão | 688 | 536 | 6,70 bi | 9 s | 10 s | 3,9 GiB | 6,6 GiB |
| FRR · 40 sessões | 27.000 | 9.000 | 6,70 bi | 19 min | 21 min | 3,9 GiB | 24,9 GiB |

## Ruído

O treino é determinístico com a mesma semente, então a régua troca só a semente, na configuração MindEye2 sem low-level.

| Semente do treino | Img→cér ↑ | Cér→img ↑ |
|---|---:|---:|
| 42 | — | — |
| 1 | — | — |
| 2 | — | — |

| Com e sem prior (1024) | Img→cér ↑ | Cér→img ↑ |
|---|---:|---:|
| com prior · semente 42 | — | — |
| sem prior · semente 42 | — | — |

| Semente da reconstrução (refinadas) | 42 | 7 |
|---|---:|---:|
| PixCorr | 0,180 | 0,175 |
| SSIM | 0,348 | 0,344 |
| Alex(2) | 85,4% | 84,9% |
| Alex(5) | 92,8% | 92,6% |
| Incep | 84,3% | 84,0% |
| CLIP | 82,4% | 83,1% |
| Eff | 0,805 | 0,800 |
| SwAV | 0,460 | 0,459 |

## Galeria

Refinadas, os mesmos estímulos para todos os modelos (a página tem mais, e as unCLIP e blurry):

![Galeria de reconstruções refinadas](results/benchmark/galeria_refinadas.jpg)

![Galeria de reconstruções unCLIP](results/benchmark/galeria_unclip.jpg)
