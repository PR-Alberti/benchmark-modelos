# Perguntas de neurociência que este dataset permite responder

Registro de 14/11/2026. Cinco perguntas que o NSD, com o código deste repositório, consegue testar, para mostrar o
tipo de pergunta que vai além de "qual modelo reconstrói melhor". Nenhuma foi rodada ainda: aqui estão a pergunta, o
desenho e os cuidados. O FRR serve de modelo de trabalho, porque treina em segundos a minutos e permite muitas
condições (ver [GUIA.md](GUIA.md#3-os-modelos)).

Índice: [1. Cansaço](#1-cansaço-e-tempo-de-tarefa) · [2. Erasing](#2-erasing-de-regiões-do-cérebro) ·
[3. Brilho e contraste](#3-augmentation-de-brilho-e-contraste) · [4. Memória e repetição](#4-memória-e-repetição) ·
[5. Categorias nunca vistas](#5-generalização-para-categorias-nunca-vistas) · [Ordem sugerida](#ordem-sugerida)

---

## 1. Cansaço e tempo de tarefa

**Pergunta:** a qualidade da representação no córtex visual cai durante uma sessão de ~1 h, e ao longo dos meses de
coleta?

**Por que dá para testar:** as 1.000 imagens de teste aparecem espalhadas pelas 40 sessões, e cada sessão tem 12
runs. Cada exibição de teste tem um "relógio" próprio: o run dentro da sessão mede o cansaço, e o número da sessão
mede o efeito de longo prazo.

**Como:**
- Treinar o FRR com 40 sessões.
- Avaliar **exibição por exibição**, sem a média das 3 repetições que o teste usa hoje. Métrica: o rank da imagem
  certa entre as 1.000, ou o cosseno com o alvo.
- Ajustar um modelo misto: `score ~ run + sessão + nº da repetição + movimento de cabeça + (1|imagem)`.

**Cuidados:**
- Os betas `_renorm` parecem normalizados por sessão (**não conferido**). Se forem, a deriva de amplitude entre
  sessões some, mas a queda dentro da sessão continua visível.
- O número da repetição se mistura com o tempo, porque as imagens repetidas aparecem mais tarde. Por isso entra como
  covariável.
- O tempo de resposta na tarefa do NSD é uma medida direta de cansaço. Ele está no `responses.tsv` do NSD, que
  **não está baixado** aqui.

## 2. Erasing de regiões do cérebro

**Pergunta:** existe uma dupla dissociação, com o córtex visual inicial sustentando o nível baixo (forma e cor) e o
córtex de alto nível sustentando o conteúdo semântico?

**O que já existe:** `~/mindeyev2/brain_region_masks.hdf5` tem as máscaras `V1`, `V2`, `V3`, `V4`, `early_vis`,
`higher_vis` e `nsd_general` de cada sujeito, nos mesmos 15.724 voxels do sujeito 1.

**Como:** apagar a região de duas formas, que respondem perguntas diferentes.
- **No teste:** apagar a ROI de um modelo já treinado, trocando os voxels pela média. Mede o quanto o modelo
  depende dessa área.
- **No treino:** treinar de novo sem a ROI. Mede se a informação está só ali ou também em outras áreas. A diferença
  entre as duas formas mede a redundância.

**Predição:** apagar `early_vis` derruba PixCorr, SSIM e Alex(2); apagar `higher_vis` derruba CLIP e Inception.

**Versão por categoria:** baixar as ROIs de localizador do NSD (`floc-faces`, `floc-places`, `floc-bodies`) e cruzar
com os rótulos de `~/mindeyev2/semantica/`. Apagar a FFA deveria prejudicar mais as fotos com pessoas do que as
paisagens, e apagar a PPA o contrário.

**Controle obrigatório:** conjuntos aleatórios de voxels do mesmo tamanho, porque as ROIs têm tamanhos muito
diferentes.

## 3. Augmentation de brilho e contraste

**Pergunta:** jitter de brilho e contraste dá algum ganho marginal e reduz a distância de desempenho entre imagens de
alto e de baixo contraste? (Se a distância em mente for entre 1 e 40 sessões, o desenho abaixo serve igual.)

**O problema conceitual:** se o brilho da imagem-alvo muda, o cérebro continua tendo visto a original. Para o ramo
de nível baixo isso equivale a ruído no rótulo. Para o CLIP o efeito tende a ser pequeno, porque ele já é quase
invariante a brilho.

**Em duas etapas:**
1. **Diagnóstico, sem treinar nada:** calcular a luminância média e o contraste RMS de cada imagem de
   `coco_images_224_float16.hdf5` e correlacionar com o erro de decoding. Hipótese, pela curva de resposta a
   contraste de V1: imagens de pouco contraste geram sinal mais fraco e são decodificadas pior, principalmente nas
   métricas de nível baixo.
2. **Intervenção:** se o efeito aparecer, comparar, com 1 sessão (onde falta dado):
   - jitter fotométrico nos alvos (o `kornia` já está instalado);
   - normalizar o contraste do alvo e prever o contraste à parte, como um número;
   - augmentation no espaço dos voxels (mixup e ruído). O MindEye1 já usa mixup, e ele é a linha de base.

**Resultado esperado:** um ganho marginal, se houver, concentrado no regime de poucos dados.

## 4. Memória e repetição

**Pergunta:** a tarefa do NSD é de reconhecimento contínuo (a cada foto, "já vi?"). A representação visual muda
quando a imagem é lembrada?
- **Supressão por repetição:** a 1ª exibição é decodificada melhor ou pior que a 2ª e a 3ª?
- **Memória subsequente:** as exibições cuja imagem foi depois reconhecida já tinham uma representação mais fiel na
  primeira vez?

**Por que interessa:** liga a fidelidade da representação visual à atenção e à profundidade de codificação, um tema
clássico de memória, medido aqui pelo decoding e não pela amplitude do sinal.

**Precisa de:** o `responses.tsv` do NSD (público, **não está baixado**) e o decoding por exibição da ideia 1. O
espaçamento entre as repetições entra como covariável.

## 5. Generalização para categorias nunca vistas

**Pergunta:** o mapeamento do cérebro para o CLIP é composicional? Um modelo que nunca viu animais no treino
decodifica fotos de animais?

**Como:** o `dataset_controlado.py` já filtra por classe semântica (`classes`, regra `exclusiva`). Comparar:
- **(a)** treino sem a classe X;
- **(b)** treino com a classe X e o **mesmo número total de imagens**.

Testar as duas na classe X. A diferença entre (a) e (b) é o custo de nunca ter visto a categoria.

**Cruzando com a ideia 2:** a generalização vem mais do córtex de alto nível (conceitos compartilhados) ou do inicial
(estatísticas de imagem que valem para tudo)?

---

## Ordem sugerida

Da mais barata para a mais cara:
1. **Ideia 3, o diagnóstico:** horas, sem treino novo.
2. **Ideia 2, com as máscaras locais:** algumas execuções de FRR.
3. **Ideia 5:** usa um código que já existe.
4. **Ideias 1 e 4:** exigem o decoding por exibição e o download dos dados comportamentais do NSD.

A 2 e a 5 dão as figuras mais "neurocientíficas" para mostrar o que o dataset permite responder.
