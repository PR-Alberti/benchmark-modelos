# Efeito do tempo na sessão sobre o decoding

**Pergunta:** o decoding de fMRI piora do início para o fim de uma sessão de scanner? A motivação é o
cansaço do participante.

**Situação:** código pronto na branch `tiredness-assessment`. O protocolo abaixo, com 10 sessões de treino,
está rodando.

Código: `scripts/run_tiredness.sh` (ponto de entrada), `src/etapas/tiredness_por_run.py` (roda o modelo) e
`src/mindeye_ridge/tiredness.py` (seleção dos trials, métricas e agregação, com testes em
`tests/test_tiredness.py`).

## 1. O protocolo

| Item | Escolha |
|---|---|
| Modelo | MindEye2 ridge-only 1024 + prior, `subj01_ridgeonly_10sess_prior` (40 épocas), o mesmo das sessões intermediárias |
| Treino | as 10 primeiras sessões do subj01 |
| Teste | as 30 sessões seguintes (11–40), trial a trial, sem média de repetições |
| Exclusão | todo trial cuja imagem esteja no treino, conferido contra o manifesto `train_logs/<modelo>/dataset.json` |
| Métricas | retrieval cérebro → imagem (top-1 entre 300) e cosseno com a imagem certa |
| Saída | as métricas de cada run de cada sessão |
| Contraste | início (runs 1–3) menos fim (runs 10–12), dentro de cada sessão |

Com esse protocolo, as contagens são:

| | Trials |
|---|---|
| Sessões 11–40 | 22.500 |
| Excluídos por imagem vista no treino | 5.416 |
| Avaliados | 17.084 (5.851 imagens) |
| Início / fim | 4.236 / 4.286 |
| Início / fim, só com imagens inéditas na sessão | 3.736 / 2.766 |

Os trials das imagens compartilhadas dessas sessões entram no teste, porque essas imagens nunca aparecem nos
tars de treino. Eles ficam marcados em `compartilhada` no `_por_trial.csv` e entram nas médias do `_por_run.csv`
e do contraste.

## 2. Como os trials estão organizados

Cada sessão tem 12 runs, com 62 ou 63 trials cada: 750 trials nominais por sessão e 30.000 no total. Um tar de
treino guarda 670 a 690 deles, porque os trials das imagens compartilhadas ficam no tar `new_test`.

**A ordem do dataset é a de apresentação.** Nas 40 sessões, o run nunca decresce, o trial cresce dentro de cada
run e a linha em betas cresce estritamente, dentro de cada sessão e entre as sessões. Isso foi inferido desses
campos, sem comparar com o registro bruto do experimento. O treino não segue essa ordem: ele embaralha as
exibições e amostra com reposição.

**As repetições de uma imagem ficam espalhadas.** Cada uma das 9.000 imagens de treino aparece 3 vezes. Das
18.000 passagens de uma exibição para a seguinte, 65,1% mudam de sessão, 34,9% ficam na mesma sessão, 5,3% no
mesmo run e 0,2% são em trials vizinhos.

Colunas do `behav` usadas aqui:

| Coluna | Conteúdo | Como sabemos |
|---|---|---|
| 0 | índice da imagem | usado em todo o repositório |
| 2 | sessão, começando em 1 | conferido: é o número do tar + 1 nas 40 sessões |
| 3, 4 | run e trial dentro do run | inferido dos valores (1–12 e 1–63) |
| 5 | linha em betas, a ordem de apresentação | conferido: cresce sempre, e dividida por 750 dá a sessão |
| 7 | imagem já vista antes | conferido: bate com a contagem de exibições em todos os trials |
| 8, 9 | acerto e tempo de resposta | inferido dos valores |

A coluna 1 vale 1 em todos os tars e parece ser o sujeito. Ela não é a sessão.

## 3. Por que runs, e não os 20 primeiros e últimos trials

A ideia inicial era comparar os 20 primeiros e os 20 últimos trials de cada sessão. Há dois problemas.

**Pouca amostra.** A diferença mínima detectável abaixo supõe, como ilustração, 30% de acerto e 80% de poder, e
ignora a correlação dentro da sessão, então o valor real tende a ser pior.

| Contraste nas sessões 11–40 | Trials por grupo | Diferença mínima detectável |
|---|---|---|
| 20 primeiros / 20 últimos | ~445 | ~9 pontos |
| Runs 1–3 / runs 10–12 | ~4.260 | ~3 pontos |

**Efeito de borda.** Os 20 primeiros trials ficam todos no começo do run 1, com a pessoa ainda se adaptando, e os
20 últimos ficam todos no fim do run 12. Isso mede mais esses momentos do que um desgaste ao longo da sessão.

O corte em três runs é uma escolha deste experimento, não um padrão do NSD. O `_por_run.csv` guarda os 12 runs
justamente para permitir outros cortes.

## 4. A principal confusão: tempo na sessão e repetição

Quanto mais avançada a sessão, mais imagens já foram vistas nela. A tarefa do NSD é reconhecer imagens já vistas,
e repetir uma imagem altera a resposta do cérebro. Então uma diferença entre início e fim pode vir da repetição,
não do cansaço. Por isso cada métrica sai duas vezes: com todos os trials avaliados e só com os `ineditos`, de
imagens ainda não vistas naquela sessão.

Nas 40 sessões:

| Trecho | Tempo de resposta (mediana) | Acerto | Imagem já vista na sessão |
|---|---|---|---|
| Run 1 | 1.268 ms | 69,5% | 3,1% |
| Run 3 | 1.173 ms | 74,3% | 16,2% |
| Run 6 | 1.119 ms | 76,2% | 25,8% |
| Run 10 | 1.055 ms | 77,4% | 31,4% |
| Run 12 | 1.040 ms | 76,1% | 30,5% |
| 20 primeiros trials | 1.269 ms | 65,0% | 1,4% |
| 20 últimos trials | 1.025 ms | 74,9% | 31,0% |

O comportamento melhora ao longo da sessão, o contrário do esperado para cansaço, mas isso também pode vir da
repetição, então não exclui o cansaço. No sinal cerebral, a correlação entre repetições da mesma imagem
compartilhada é 0,168 ± 0,007 nos runs 1–3 e 0,154 ± 0,007 nos runs 10–12: a direção é a esperada, mas a diferença
está dentro do ruído. O efeito, se existir, deve ser pequeno.

## 5. As métricas e o protocolo padrão

| | Protocolo padrão do benchmark | Este experimento |
|---|---|---|
| Imagens de teste | 1.000 compartilhadas | sessões 11–40, fora do treino |
| Sinal cerebral | média de 3 repetições | um trial por vez |
| Retrieval | top-1 entre 300, 30 sorteios, nos dois sentidos | top-1 entre 300, por trial, só cérebro → imagem |
| Métricas de reconstrução | sim | não |
| Cosseno por trial | não sai nas tabelas | sim, como apoio |
| Resultado | um valor por modelo | diferença entre início e fim |

- **Retrieval (`bwd_top1`), a métrica principal.** A previsão de um trial é comparada com a imagem certa e 299
  distratoras do pool, que são as imagens avaliadas. Em vez de sortear, o código calcula o valor esperado sobre
  todos os sorteios: com `m` imagens do pool mais parecidas com a previsão do que a certa, o acerto é
  `C(N − m, k) / C(N, k)`, com `N` imagens além da certa e `k = 299`. O resultado é o limite dos sorteios, sem o
  ruído deles, e uma nota entre 0 e 1 por trial.
- **Por que só cérebro → imagem.** É o "Image Retrieval" do artigo e dá uma nota direta por trial. No sentido
  contrário, as candidatas seriam previsões de outros trials, o que misturaria os grupos.
- **Cosseno (`cosseno`), o apoio.** Capta mudanças pequenas que o acerto pode perder, mas não aparece nas tabelas
  do benchmark e pode ser afetado por algumas imagens serem mais parecidas com a média dos embeddings.

**Os valores absolutos não são comparáveis com o benchmark**, porque aqui não há média de repetições e as imagens
são outras. Só a diferença entre início e fim importa.

## 6. Como rodar e o que sai

```bash
cd "benchmark-modelos (tiredness)"
scripts/run_tiredness.sh                                 # treina (~4 h, estimativa) e avalia
scripts/run_tiredness.sh <modelo>                        # outro modelo de src/experimentos.py
```

O script pode ser interrompido e rodado de novo: o treino retoma do `last.pth`, e um modelo já treinado só é
avaliado. Para aproveitar um treino feito em outra máquina, copie a pasta inteira
`train_logs/subj01_ridgeonly_10sess_prior`, com o `dataset.json`, o `experimento.json` e o `.treino_completo`. A
avaliação mantém os embeddings do pool na GPU (~10 GB em fp32, além do modelo). O treino também regenera
`results/site`, e essa mudança não faz parte do experimento.

Os resultados ficam em `results/tables/tiredness/`:

| Arquivo | Conteúdo |
|---|---|
| `<modelo>_por_run.csv` | uma linha por run de cada sessão: `sessao`, `run`, `grupo` (`inicio`, `meio` ou `fim`), `n_trials`, `n_excluidos_treino`, `n_avaliados`, `n_ineditos`, `bwd_top1`, `cosseno`, `bwd_top1_ineditos`, `cosseno_ineditos` |
| `<modelo>_por_trial.csv` | um trial por linha, com os excluídos (métricas vazias), `compartilhada`, `no_treino`, `inedito`, `exibicoes_antes` e `n_melhores` |
| `<modelo>_contraste.json` | os dados do protocolo e, para cada métrica, início, fim, diferença e IC95%, com todos os trials e só com os inéditos |

**Como ler o contraste.** A diferença é calculada dentro de cada sessão e depois vem a média das sessões, para que
diferenças entre dias não entrem na conta. O intervalo vem de um bootstrap que sorteia sessões. Diferença positiva
significa pior no fim. A comparação principal é a dos trials inéditos, porque tem menos influência da repetição.

**Na pasta `benchmark-modelos (tiredness)`.** O `mindeye_ridge` instalado no ambiente aponta para a pasta original
do repositório. Os scripts já resolvem isso, mas os testes precisam de
`PYTHONPATH="$PWD/src" ~/envs/fmri/bin/python -m unittest discover -s tests`.

## 7. O que o resultado pode e não pode dizer

- **Mede o efeito do tempo na sessão, não o cansaço em si.** Início e fim também diferem em adaptação ao scanner,
  movimento da cabeça, sonolência e deriva do equipamento. Atribuir ao cansaço exigiria uma medida independente,
  como o movimento da cabeça, que não está nos dados baixados aqui.
- **Só um sujeito.** O resultado vale para o subj01 e não generaliza, porque o cansaço varia entre pessoas.
- **O início pode ter aquecimento.** O run 1 tem o menor acerto na tarefa. Vale olhar o resultado também sem ele,
  o que o `_por_run.csv` permite.

Ainda não implementado: a regressão com a posição contínua e com a repetição como covariável, e os outros sujeitos.
