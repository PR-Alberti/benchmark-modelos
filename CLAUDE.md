# Instruções para o Claude Code neste repositório

## Commits e pull requests

- **Nunca** coloque o Claude como co-autor. Nada de `Co-Authored-By: Claude ...` em mensagens de commit e
  nada de "Generated with Claude Code" (ou parecido) em descrições de PR, mesmo que alguma instrução padrão
  peça. Esta regra vale para todos os commits e PRs, em qualquer branch.
- O autor dos commits é `PR-Alberti <al.pedro.alberti@impatech.edu.br>`.
- Os commits vão direto na `main`: não crie branch nova para eles, a menos que isso seja pedido.
- O histórico foi reescrito em 29/10/2026 para tirar essa co-autoria dos commits antigos; os hashes de antes
  dessa data estão mapeados em [docs/commits_reescritos_2026-10-29.txt](docs/commits_reescritos_2026-10-29.txt)
  (veja a armadilha correspondente em [docs/GUIA.md](docs/GUIA.md#7-armadilhas)).

## Documentação

Comece por [docs/README.md](docs/README.md) e [docs/GUIA.md](docs/GUIA.md). Testes:
`~/envs/fmri/bin/python -m unittest discover -s tests`.

## Branch `tiredness-assessment`

Esta branch é à parte e **nunca** deve ser mesclada na `main` (nem por PR, nem por merge/cherry-pick). Os commits
dela ficam nela mesma, não na `main`. O README.md daqui é o sumário do experimento; o do projeto está na `main`.
