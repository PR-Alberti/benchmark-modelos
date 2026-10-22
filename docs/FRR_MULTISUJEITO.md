# FRR multi-sujeito

Extensão linear do FRR para vários sujeitos, no molde do MindEye2: um pré-treino nos outros 7 sujeitos do
NSD e, no subj01, o ajuste só da ridge dele. Tudo linear e em forma fechada. Código em
`src/mindeye_ridge/multisujeito.py` (o modelo) e `src/run_frr_ms.py` (treino, avaliação e saída).

```
voxels do sujeito s ──Rₛ──▶ k números ──Q──▶ embedding CLIP bigG (256 × 1.664)
   (próprio do sujeito)              (compartilhada)
```

- **Rₛ**: uma ridge por sujeito, dos voxels dele para um espaço de **k = 1.024** dimensões (o `hidden_dim`
  do MindEye2 de 1.024).
- **Q**: a leitura compartilhada, desse espaço para o embedding. É ela que carrega o que vem dos outros sujeitos.

## Os modelos

| | Experimento | Pré-treino da leitura Q | Ajuste no subj01 |
|---|---|---|---|
| A | `repro_subj01_frr_{1,40}sess` e `subj01_frr_20sess` (o FRR de sempre) | — (voxels → CLIP direto) | tudo |
| **B** | `subj01_frr_postoreduzido_{1,40}sess` | só o próprio subj01 (controle de posto reduzido) | só R₁ |
| **C** | `subj01_frr_multisujeito_{1,40}sess` | sujeitos 2–8, todas as sessões | só R₁ |

C contra B mede o efeito dos outros sujeitos; B contra A, o efeito do gargalo de k sozinho. Sem B, um C
melhor que A não diz se o ganho veio do pré-treino ou da mudança de arquitetura.

## Como é ajustado

**Pré-treino**: mínimos quadrados alternados sobre Σₛ wₛ ‖Xₛ Rₛ W − Yₛ‖², com **peso igual por sujeito**
(wₛ = 1/nₛ, o efeito dos lotes balanceados do MindEye2):

1. **início**: as k componentes principais dos alvos (SVD aleatorizado), sem dado de cérebro;
2. **passo R**: com a leitura fixa, cada Rₛ é um FRR dos voxels de s para o alvo reduzido Tₛ = Yₛ Q',
   independente por sujeito;
3. **passo W**: com os Rₛ fixos, W é um FRR das entradas Zₛ = Xₛ Rₛ empilhadas, com os pesos. Só entram
   as somas Z'Z e Z'Y, acumuladas sujeito a sujeito: o resultado é o de um ajuste conjunto, com um sujeito
   por vez na memória;
4. **escala**: W = L Q com Q de linhas ortonormais (o produto Rₛ W não muda); a leitura guardada é Q.

As frações (uma por dimensão do alvo, como no FRR) e a parada saem de uma **validação separada**: 10% das
imagens de cada sujeito do pré-treino, que não entram no ajuste. Fica a iteração de menor perda de
validação (máximo de 10; para quando o ganho relativo cai abaixo de 10⁻³). O erro de validação do passo W
é calculado pela forma quadrática (β'Z'Zβ − 2β'Z'y + ‖y‖²), sem materializar as previsões.

**Ajuste fino**: com Q congelada, R₁ é o FRR comum (5 dobras agrupadas por imagem) dos voxels do subj01
para T₁ = (Y₁ − ȳ₁) Q'; a previsão é ȳ₁ + x R₁ Q. Mesmo treino, teste, alvo e avaliação do `run_frr.py`.

O pré-treino dos outros sujeitos é um só para 1 e 40 sessões do subj01: fica em
`train_logs/_pretreino_frr_ms/<assinatura>/` (a base espectral de cada sujeito, a leitura `Q_melhor.npy` e o
histórico em `estado.json`) e retoma da última iteração. O do controle B fica em `train_logs/<modelo>/pretreino/`.

## Cuidados

- **Vazamento**: as 1.000 imagens de teste do subj01 são as compartilhadas, que os outros sujeitos também
  viram. Antes de qualquer conta, `run_frr_ms.py` confere que nenhuma delas está no treino de um sujeito do
  pré-treino, e para se estiver.
- **O espaço compartilhado não é o do MindEye2**: aqui ele é moldado por erro quadrático até o CLIP, de forma
  linear; lá, pelo backbone não linear e pela perda contrastiva. Nenhuma peça do MindEye2 treinado entra.
- **O encolhimento continua**: é regressão linear, então a previsão segue perto da média de treino e o unCLIP
  deve continuar gerando imagens acinzentadas (como o FRR). A pergunta é quanto melhora, não se fica nítido.
- **Os mínimos alternados não são convexos**: podem parar num mínimo ruim; o histórico da validação mostra.

## Rodar

```bash
setup/bootstrap.sh --multisujeito                       # sujeitos 2-8: 12,1 GB (betas + 258 tars)
cd src
python treina.py subj01_frr_multisujeito_1sess subj01_frr_multisujeito_40sess      # C
python treina.py subj01_frr_postoreduzido_1sess subj01_frr_postoreduzido_40sess    # B
python treina.py subj01_frr_multisujeito_1sess --reconstrucao refino                # + unCLIP, refino e métricas
```

Os hiperparâmetros (`treina.py`, modelo `frr_ms`): `k`, `pretreino` (lista de sujeitos ou `"proprio"`),
`pesos` (`"sujeito"` ou `"exibicao"`), `iteracoes`, `tolerancia`, `validacao`, `folds`, `grid`, `fracs`,
`seed`, `chunk`.

## Resultados (subj01, 1.000 imagens de teste)

Retrieval do embedding (média de 30 sorteios de 300) e métricas das reconstruções unCLIP; entre parênteses,
a refinada. A é o FRR de sempre (`repro_subj01_frr_{1,40}sess` e `subj01_frr_20sess`).

| Sessões | | fwd | bwd | PixCorr | SSIM | AlexNet(5) | Inception | CLIP | Corr. cerebral |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 0,550 | 0,031 | 0,103 (0,079) | 0,446 (0,474) | 0,524 (0,521) | 0,512 (0,526) | 0,506 (0,516) | 0,037 (0,016) |
| 1 | B | 0,411 | 0,096 | **0,192** (0,170) | 0,456 (0,480) | **0,571** (0,592) | **0,542** (0,597) | **0,543** (0,578) | **0,056** (0,048) |
| 1 | C | **0,629** | **0,124** | 0,181 (0,160) | 0,456 (0,477) | 0,554 (0,574) | 0,540 (0,583) | 0,523 (0,566) | 0,045 (0,039) |
| 20 | A | 0,943 | 0,533 | 0,325 (0,301) | 0,459 (0,479) | 0,669 (0,687) | 0,624 (0,673) | 0,614 (0,643) | 0,082 (0,078) |
| 20 | B | 0,945 | 0,547 | 0,323 (0,303) | 0,457 (0,479) | **0,717** (0,718) | **0,661** (0,685) | **0,645** (0,654) | **0,108** (0,099) |
| 20 | C | 0,941 | 0,502 | 0,322 (0,296) | 0,457 (0,478) | 0,659 (0,686) | 0,626 (0,668) | 0,613 (0,641) | 0,077 (0,077) |
| 40 | A | 0,964 | 0,377 | 0,278 (0,248) | 0,451 (0,475) | 0,603 (0,637) | 0,572 (0,633) | 0,565 (0,612) | 0,063 (0,065) |
| 40 | B | **0,970** | **0,439** | **0,304** (0,275) | 0,453 (0,476) | **0,632** (0,662) | **0,589** (0,647) | **0,585** (0,621) | **0,072** (0,071) |
| 40 | C | 0,961 | 0,387 | 0,283 (0,252) | 0,452 (0,476) | 0,610 (0,645) | 0,582 (0,653) | 0,572 (0,614) | 0,063 (0,067) |

- **O gargalo explica o ganho (B contra A)**: com 1 sessão, só o posto reduzido quase dobra o PixCorr e
  sobe o CLIP de 0,506 para 0,543; com 20 e 40 sessões, B ainda é o melhor nas métricas de alto nível.
- **Os outros sujeitos não ajudam a reconstrução (C contra B)**: com 1 sessão, C melhora o retrieval (fwd
  0,41 → 0,63), mas a imagem fica igual ou pior; com 20 e 40, C volta ao nível de A.
- O pré-treino parou na 2ª iteração em todos (R² de validação 0,036 nos sujeitos 2–8). No B de 1 sessão
  (536 imagens) o R² de validação foi negativo e o posto efetivo caiu para 481.
- O bwd de 20 sessões é maior que o de 40 nos três modelos: vem do conjunto de dados (6.708 imagens com 1 a
  3 repetições contra 9.000 com 3), não do modelo.
- Em `subj01_frr_20sess` e `subj01_frr_postoreduzido_40sess`, o unCLIP gerou a imagem 351 toda preta;
  o PixCorr dela conta como 0 (`final_evaluations.py`), em vez de anular a média.

## Recursos (medidos nesta corrida, RTX A4500 de 20 GB)

| | |
|---|---|
| Disco | 12,1 GB de dados dos sujeitos 2–8 + ~49 GB de alvos CLIP deles (~7 GB por sujeito) + 7,1 GB do pré-treino (bases espectrais e leituras) + ~2,3 GB por modelo reconstruído |
| VRAM | ≤ 4 GB nos ajustes; a reconstrução, como a do FRR (cabe nos 20 GB) |
| RAM | pico de 15–27 GB (o carregamento do bigG para os alvos já leva o processo a ~20 GB) |
| Tempo | alvos ~3 min por sujeito (uma vez); bases 1–2 min por sujeito (uma vez); ~2,5 min por iteração; o treino de C com 1 sessão, do zero, levou 43 min; ajuste fino de 1 s (1 sessão) a 8 min (40 sessões); unCLIP + refino + métricas ~3 h 15 por modelo |
