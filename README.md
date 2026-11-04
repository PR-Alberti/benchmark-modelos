# Efeito do tempo na sessão sobre o decoding (branch `tiredness-assessment`)

> **Esta branch é à parte e NÃO deve ser mesclada na `main`.** Ela guarda só o experimento do tempo na sessão.
> O README geral do projeto está na `main` (`git show main:README.md`).

**Pergunta:** o decoding de fMRI piora do início para o fim de uma sessão de scanner (motivação: cansaço)?

## Protocolo

| Item | Escolha |
|---|---|
| Modelo | MindEye2 ridge-only 1024 + prior, `subj01_ridgeonly_10sess_prior` (40 épocas), treinado nas 10 primeiras sessões do subj01 |
| Teste | sessões 11–40, trial a trial, sem média das 3 repetições |
| Exclusão | trials de imagens vistas no treino: 22.500 trials, 5.416 excluídos, **17.084 avaliados** (5.851 imagens) |
| Métricas | retrieval cérebro → imagem (top-1 entre 300), a principal; cosseno com a imagem certa |
| Contraste | runs 1–3 (início) menos runs 10–12 (fim), dentro de cada sessão; média das 30 sessões |
| IC 95% | bootstrap de percentis (10.000 reamostras, semente 42) sorteando as **sessões** com reposição |
| Confusão tratada | cada métrica sai com todos os trials e só com os **inéditos** (imagens ainda não vistas na sessão) |

## Resultados

Diferença = início − fim; positiva significa pior no fim. A variância e o EP são da diferença entre as 30 sessões.

| Métrica | Trials | Início | Fim | Diferença (IC 95%) | Variância | EP | Sessões com dif > 0 |
|---|---|---|---|---|---|---|---|
| Retrieval | todos | 92,3% | 90,8% | **+1,5 pt (0,5 a 2,6)** | 8,5·10⁻⁴ | 0,53 pt | 22/30 |
| Retrieval | inéditos | 92,9% | 91,4% | **+1,5 pt (0,2 a 2,7)** | 1,2·10⁻³ | 0,64 pt | 20/30 |
| Cosseno | todos | 0,0879 | 0,0868 | +0,0011 (0,0001 a 0,0021) | 8·10⁻⁶ | 0,0005 | 20/30 |
| Cosseno | inéditos | 0,0884 | 0,0876 | +0,0008 (−0,0004 a 0,0020) | 1,2·10⁻⁵ | 0,0006 | 16/30 |

Retrieval por run (média das 30 sessões): 93,3 · 92,0 · 91,6 · 90,1 · 91,3 · 91,5 · 90,9 · 89,5 · 91,0 · 90,7 · 90,1 · 91,6 (%).

## Leitura

- O retrieval cai cerca de 1,5 ponto do início para o fim, com IC que exclui zero, com todos os trials e só com os
  inéditos. A repetição de imagens não parece explicá-la.
- Não há queda contínua: a diferença vem em boa parte do run 1 ser o melhor, e o run 12 volta a 91,6%.
- O efeito é pequeno: abaixo da diferença mínima detectável do protocolo (~3 pontos) e metade do DP entre sessões
  (2,9 pontos). Só a média de 30 sessões o torna detectável.
- O cosseno aponta a mesma direção, mas é apoio fraco: com inéditos o IC inclui zero, e sem correção para os 4
  contrastes o intervalo de "todos" não resistiria.

## Limites

- Mede o efeito do tempo na sessão, não o cansaço em si (adaptação ao scanner, movimento da cabeça e sonolência
  também diferem entre início e fim).
- Só o subj01 e uma semente de treino. O IC cobre só a variação entre sessões.
- Os valores absolutos não são comparáveis com o benchmark; só a diferença início − fim importa.

## Onde está tudo

| | |
|---|---|
| Protocolo detalhado, confusões e como ler | [docs/TIREDNESS.md](docs/TIREDNESS.md) |
| Resultados | `results/tables/tiredness/` (`_por_run.csv`, `_por_trial.csv`, `_contraste.json`) |
| Código | `scripts/run_tiredness.sh`, `src/etapas/tiredness_por_run.py`, `src/mindeye_ridge/tiredness.py` |
| Testes | `PYTHONPATH=src ~/envs/fmri/bin/python -m unittest discover -s tests -p "test_tiredness.py"` |
