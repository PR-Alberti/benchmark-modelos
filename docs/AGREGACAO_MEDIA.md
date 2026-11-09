# Média das repetições no treino (branch `agregation`)

**Pergunta:** treinar com a média dos betas das repetições de cada imagem única (uma amostra por imagem) muda
o resultado, em comparação com usar cada exibição como uma amostra (o que o benchmark faz no MindEye2)?

## O que roda

20 experimentos, definidos em `src/experimentos.py` (bloco `EPOCAS_MEDIA`), todos com `"agregacao": "media"`,
as N primeiras sessões do subj01, semente 42 e o teste de sempre (1.000 imagens, média das 3 repetições):

| Modelo | Nomes (`<N>` = 1, 5, 10, 20, 40) | Treina |
|---|---|---|
| ME2 sem low-level | `subj01_ridgeonly_<N>sess_prior_media` | Ridge 1024 + prior |
| ME2 com low-level | `subj01_ridgeonly_<N>sess_4096blurry_media` | Ridge 4096 + blurry |
| ME1 com low-level | `subj01_me1_<N>sess_media` | alto nível + low-level |
| ME1 sem low-level | `subj01_me1_<N>sess_media_semlowlevel` | nada: mesmo checkpoint do alto nível, reconstrução sem a imagem borrada |

Os pares de comparação são os modelos sem `_media` (exibições separadas): `subj01_ridgeonly_{1,40}sess_*`
do benchmark, `subj01_ridgeonly_{5,10,20}sess_*` e `subj01_me1_{1,5,10,20,40}sess*`.

## Decisões

- **Épocas do ME2.** A época do MindEye2 acompanha o número de amostras (imagens únicas, no lugar de
  exibições), então as épocas sobem na razão exibições / imagens únicas para manter o número de passos das
  corridas sem média:

  | Sessões | Exibições / imagens únicas | Épocas sem média | Épocas com média |
  |---|---|---|---|
  | 1 | 688 / 536 | 150 | 193 |
  | 5 | 3.410 / 2.267 | 60 | 90 |
  | 10 | 6.803 / 4.073 | 40 | 67 |
  | 20 | 13.509 / 6.708 | 30 | 60 |
  | 40 | 27.000 / 9.000 | 20 | 60 |

- **ME1.** A amostra do MindEye1 já é a imagem (rodízio das repetições), então ficam as 240 épocas.
- **Hiperparâmetros** iguais aos pares sem média: `frozen_fp16`, `ckpt_interval=1` no ME2, `save_every` 10 (1 sessão)
  e 5 (5 sessões) no ME1.
- **Teste** não muda: o contraste é só a agregação no treino.

## Como rodar

A GPU de 20 GB comporta uma fila por vez; rode a do ME1 depois da do ME2. Cada tamanho de sessões é
commitado ao terminar (resultados e site).

```bash
tmux new -s me2media "scripts/run_agregacao_media_me2.sh"
tmux new -s me1media "scripts/run_agregacao_media_me1.sh"      # só depois que a do ME2 acabar
SESSOES="1 5" scripts/run_agregacao_media_me2.sh                 # só alguns tamanhos
```

Pode interromper e rodar de novo: etapas concluídas são puladas e o treino retoma do `last.pth`. A fila para
com menos de 60 GB livres (`MIN_LIVRE_GB`). Logs em `results/logs/agregacao_media_{me2,me1}.log`.

## Tempo estimado (RTX A4500, a partir das corridas sem média)

| Fila | Treino | Reconstrução + métricas | Total |
|---|---|---|---|
| ME2 (10 modelos) | ~58 h | ~33 h | ~90 h |
| ME1 (5 treinos) | ~78 h | ~9 h | ~87 h |

Treino do ME2 por sessões (sem / com low-level): 1,9 / 2,7 h; 3,1 / 4,7 h; 4,0 / 6,2 h; 5,8 / 9,2 h; 8,0 / 12,6 h.
ME1 (alto nível + low-level): 2,9; 8,7; 15; 22; 29 h. O tempo segue o número de passos (ME2) e de imagens únicas
(ME1), não o de sessões. Estimativa: reveja depois do primeiro treino de cada tamanho.

## Antes de começar

- Apagar os restos do teste curto (`smoke_*`): `train_logs/`, `mindeye1/train_logs/smoke_me1`,
  `mindeye1/train_logs/models/smoke_me1_ae` e `results/site/modelos/smoke_*.json`. Se ficarem, entram no site.
- **ME1 low-level:** no teste curto com 2 épocas deu `ZeroDivisionError` no `OneCycleLR`. Provavelmente é o
  número de épocas do teste; confirmar com 3 épocas antes de iniciar a fila do ME1.
- O teste curto (5 sessões) passou para ME2 1024, ME2 4096 + blurry e o alto nível do ME1.

## Estado

Nenhum dos 20 treinos rodou ainda.
