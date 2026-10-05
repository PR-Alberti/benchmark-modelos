# Plano de estágio: o que já está no repositório

O plano (`Plano de estágio.pdf`) pergunta **quanta fMRI é preciso** para extrair
informação confiável de uma tarefa visual: quanto dado específico do sujeito, quanto entre sujeitos
e, para um tempo de aquisição fixo, se vale mais ter imagens únicas ou mais repetições. O NSD serve
de banco de provas, e decodificar o embedding CLIP da imagem vista serve de substituto para a
decodificação de imagética visual. A meta é uma lei de escala: como a quantidade de dados exigida
depende do nível de informação desejado e do modelo.

Esta branch (`estagio`) é onde essa análise acontece. A tabela diz o que já existe; os números
estão no [BENCHMARK.md](BENCHMARK.md), e o que foi feito e por quê, no [EXPERIMENTO.md](EXPERIMENTO.md).

## Dados

| Tarefa do plano | Estado | Onde |
|---|---|---|
| Splits de treino e teste que não mudam entre os experimentos | feito | Teste fixo: as 1.000 imagens compartilhadas do NSD (`new_test`), 3 repetições cada; treino: as primeiras N sessões do sujeito, sem nenhuma imagem do teste (conferido em `run_frr.py`). Leitura em `src/mindeye_ridge/nsd_data.py` |
| Embeddings CLIP de todas as imagens-alvo | feito | `src/mindeye_ridge/clip_targets.py`: ViT-bigG/14, 256 × 1.664 tokens, em cache em disco |
| Pipeline reprodutível de subconjuntos controlados | **pendente** | Hoje o único eixo é o número de sessões (`--num_sessions`). As reduções do plano (imagens únicas, repetições, variedade) pedem sorteios com semente sobre as exibições; o `frr.py` já aceita qualquer subconjunto de exibições (`X` e `groups`) e de alvos (`Y`) |

## Modelos

| Tarefa do plano | Estado | Onde |
|---|---|---|
| Modelo linear por sujeito (FRR, Doerig et al.) | feito | `src/mindeye_ridge/frr.py`, `src/run_frr.py`. Validado contra o pacote `fracridge` e contra uma ridge resolvida pela equação normal (`tests/test_frr.py`) |
| Modelo não linear por sujeito, da literatura | **pendente** | Falta escolher por relevância e reprodutibilidade, e implementar |
| MindEye2 simplificado (sem reconstrução de baixo nível e sem unCLIP), pré-treinado e adaptado ao sujeito | feito | Os ridge-only de 1024 (`scripts/run_ridgeonly_prior.sh`): só a ridge do sujeito é treinada, a partir do pré-treino nos outros 7 sujeitos |
| Quais componentes do MindEye2 ficam mantidos, congelados ou retreinados | feito | [EXPERIMENTO.md](EXPERIMENTO.md): backbone e prior congelados, ridge treinada (2,21% dos parâmetros); o 4096 + blurry mantém o ramo de baixo nível |
| Custo computacional e tempo de treino como dimensão de complexidade | parcial | Tempo de treino dos ridge-only e custo do FRR (tempo, GPU, RAM) no [BENCHMARK.md](BENCHMARK.md). Falta padronizar a medida entre famílias |

## Comparação

| Tarefa do plano | Estado | Observação |
|---|---|---|
| Redução de dados por sessões | parcial | 1 e 40 sessões para o FRR, o ridge 1024 e o MindEye1; as condições de 25, 35 e 50 h do plano não foram rodadas |
| Redução de dados por imagens únicas e por repetições | **pendente** | Depende do pipeline de subconjuntos |
| Redução de variedade (por categoria, aleatória, maximizando distância) | **pendente** | Idem |
| Mesmo teste e mesmo sujeito em todas as condições | feito | O benchmark inteiro usa o subj01 e as mesmas 1.000 imagens |

## Extensões

Espaço multi-sujeito com modelos simples, variação de SNR e ROIs estão **pendentes**. Para ROIs,
as máscaras do NSD já estão baixadas (`brain_region_masks.hdf5`).

## Três coisas que o trabalho do FRR trouxe e que valem para os próximos modelos

- **O sentido do retrieval.** No código do MindEye2, `fwd` é imagem → cérebro e `bwd` é cérebro →
  imagem, o oposto do comentário no `final_evaluations.py` e do que o artigo define como Image
  Retrieval. Nos modelos contrastivos as duas direções diferem de 0 a 16 pontos; num decodificador
  de regressão, que encolhe as previsões em direção à média, passam de 50 (hubness). O benchmark
  rotula as colunas pela direção (Img→cér, Cér→img) e mostra também o retrieval depois de tirar a
  média de treino.
- **O critério de escolha da fração do FRR.** A grade de Doerig et al. (0,05 a 1) é a que o plano
  especifica e a que o benchmark usa. No subj01, com os voxels do nsdgeneral, o ótimo do **erro
  quadrático** da validação cruzada cai abaixo dela, e estender a grade até 0,001 dobra o R² da
  validação cruzada com 40 sessões, mas **piora o retrieval no teste**: encolher mais aproxima as
  previsões da média e o ranking se perde. O erro quadrático não é o critério do retrieval. Ao
  reduzir dados, o critério de seleção (erro quadrático, correlação, retrieval) é um grau de
  liberdade a explicitar e a manter fixo entre as condições; o JSON de cada corrida guarda o
  histograma das frações escolhidas e `pares_fracao_alvo_inalcancaveis`, para ver se a grade
  alcança o ótimo.
- **A entrada do embedder CLIP.** O ViT-bigG dá saídas diferentes para a mesma imagem em fp16 e em
  fp32 (cosseno 0,996 entre as duas versões). O treino do MindEye2 alimenta o embedder com a imagem
  em fp16, e o `verify_retrieval.py` e o `final_evaluations.py` com a imagem em fp32
  (`all_images.pt`). O FRR segue as duas convenções, como os outros modelos: treina com o alvo em
  fp16 e é avaliado contra o alvo em fp32. Um modelo novo que misture as duas muda os números sem
  avisar.
