# Procedimento: protocolos multi-sujeito do MindEye1 (branch `ME1-multisubj`)

Origem: tarefa "Determinar protocolos multi-subj" da semana 23–27/11/2026 no Notion.

## Como usar (para o Pedro)

São **6 máquinas em paralelo**, uma por pré-treino. Em cada PC:

1. instale o Claude Code e abra-o numa pasta qualquer (ex.: `~`);
2. diga, trocando o número:

   > Clone https://github.com/PR-Alberti/benchmark-modelos, vá para a branch `ME1-multisubj`, leia
   > `docs/PROCEDIMENTO_ME1_MULTISUJ.md` e faça o trabalho da **máquina 3**, do começo ao fim.

3. quando ele pedir, faça você mesmo o login do GitHub (`gh auth login`), para ele poder subir os resultados.

Cada máquina faz o seu pré-treino e, em seguida, os 5 ajustes no subj01 (1, 5, 10, 20 e 40 sessões), com a
avaliação. Os resultados de todas as máquinas chegam nesta branch, cada uma na sua pasta, sem conflito.
Quem terminar por último monta a tabela final (seção 7).

## As 6 máquinas

| Máquina | Pré-treino | Sujeitos do pré-treino | Sessões de cada sujeito | Batch (por sujeito) | Depois: ajustes no subj01 |
|---|---|---|---|---|---|
| **1** | `2subj_10sess` | 2, 5 | 10 | 24 (12) | 1, 5, 10, 20, 40 sessões |
| **2** | `2subj_20sess` | 2, 5 | 20 | 24 (12) | 1, 5, 10, 20, 40 sessões |
| **3** | `4subj_5sess` | 2, 5, 7, 3 | 5 | 24 (6) | 1, 5, 10, 20, 40 sessões |
| **4** | `4subj_10sess` | 2, 5, 7, 3 | 10 | 24 (6) | 1, 5, 10, 20, 40 sessões |
| **5** | `7subj_1sess` | 2, 3, 4, 5, 6, 7, 8 | 1 | 21 (3) | 1, 5, 10, 20, 40 sessões |
| **6** | `7subj_5sess` | 2, 3, 4, 5, 6, 7, 8 | 5 | 21 (3) | 1, 5, 10, 20, 40 sessões |

É a tabela do Notion (2 sujeitos × 10 e 20 sessões; 4 × 5 e 10; 7 × 1 e 5), lida linha a linha. Sessões = as
N **primeiras** de cada sujeito. Por que esses sujeitos: 2, 5 e 7 têm 40 sessões; o 3 tem 32; 4 e 8 têm 30 e o
6 tem 32. Nenhuma combinação pede mais sessões do que o sujeito tem.

---

## Para o Claude da máquina N

> Você recebeu um número de máquina (1 a 6). Execute as etapas abaixo na ordem, sem pedir confirmação para o
> que está decidido aqui. Pare e pergunte ao Pedro só nos casos da seção 8. Siga o `CLAUDE.md` do repositório:
> commits com autor `PR-Alberti <al.pedro.alberti@impatech.edu.br>`, **nunca** co-autoria do Claude, e nesta
> branch (não na `main`). Nunca digite senhas: login do GitHub, HuggingFace etc. é o Pedro quem faz.

### Etapa 0. Conferir a máquina

```bash
nvidia-smi          # GPU NVIDIA com >= 20 GB
free -g             # >= 30 GB de RAM
df -h ~             # >= 120 GB livres
```

Precisa de `git`, `curl`, `tmux` e conda (ou `python3.11`); se faltar, siga a seção 1 do
[NOVO_COMPUTADOR.md](NOVO_COMPUTADOR.md). Anote GPU, RAM e disco no registro (seção 9).

### Etapa 1. Repositório e ambiente

```bash
git clone https://github.com/PR-Alberti/benchmark-modelos.git ~/benchmark-modelos
cd ~/benchmark-modelos && git checkout ME1-multisubj
git config user.name PR-Alberti && git config user.email al.pedro.alberti@impatech.edu.br
setup/setup_env.sh --with-extras                    # ambiente ~/envs/fmri (~15 min)
~/envs/fmri/bin/pip install -q --no-deps info-nce-pytorch==0.1.0 pytorch-msssim==1.0.0 bitsandbytes==0.43.3
```

### Etapa 2. Dados (só o necessário: ~40 GB)

Dentro de `tmux new -s setup` (o download leva horas):

```bash
~/envs/fmri/bin/python setup/download_data.py --stage finetune --subj 1     # subj01: 40 sessões + teste
~/envs/fmri/bin/python setup/download_data.py --stage train --subj 2 3 4 5 6 7 8
```

Baixe todos os sujeitos 2–8 mesmo que a sua máquina use só alguns (são 12 GB e evitam erro). Se cair, rode de
novo: continua de onde parou. Os dados ficam em `~/mindeyev2` (formato do MindEye2: `wds/subj0X/train/<sessão>.tar`,
`betas_all_subj0X_fp32_renorm.hdf5`, `coco_images_224_float16.hdf5`).

Confira:

```bash
~/envs/fmri/bin/python -m unittest tests.test_me1_multisuj_maquinas      # OK (6 testes)
mindeye1/src/run_me1_ms_maquina.sh --lista                               # mostra as 6 máquinas
```

### Etapa 3. Teste de fumaça (~15 min)

```bash
cd ~/benchmark-modelos
SMOKE=1 FT_SESSOES="1" mindeye1/src/run_me1_ms_maquina.sh N
```

Treina 3 épocas de 2 passos no pré-treino da sua máquina e num ajuste de 1 sessão, com nomes `_smoke`, sem
commit. Tem que terminar com `final_retrieval.json` nas duas pastas (`~/me1_multisubj/train_logs/*_smoke/`) e a
linha `fim: 0 falha(s)`. O retrieval do teste de fumaça é ruim mesmo; o que importa é rodar até o fim.
Se cair, leia `~/me1_multisubj/train_logs/<nome>/stdout.log`, corrija (import faltando, caminho de dados) e repita.
Uma correção de código vai num commit separado nesta branch com `git pull --rebase` antes do push, porque as
outras máquinas podem ter subido coisas.

Depois do teste, apague as pastas `_smoke` em `~/me1_multisubj/train_logs/`.

### Etapa 4. Rodar de verdade

```bash
tmux new -s maquinaN
cd ~/benchmark-modelos
mindeye1/src/run_me1_ms_maquina.sh N
```

O que o script faz, nesta ordem:

1. **pré-treino** `me1ms_pre_<config>`: MindEye1 com uma ridge por sujeito, nos sujeitos e sessões da tabela,
   150 épocas;
2. **cinco ajustes** `me1ms_ft_<config>__subj01_<M>sess`, M = 1, 5, 10, 20, 40: parte do pré-treino, troca a
   ridge pela do subj01 e treina tudo com as M primeiras sessões dele, 150 épocas;
3. **avaliação** ao fim de cada fase: retrieval no teste novo do subj01 (1.000 imagens, 3 repetições promediadas,
   30 sorteios de 300 contra o CLIP ViT-L/14 da imagem) → `final_retrieval.json` (fwd e bwd, com IC de 95%);
4. ao fim de cada fase, copia `args.json`, `log.jsonl`, `final_retrieval.json` e `stdout.log` para
   `results/me1_multisubj/<nome>/`, faz commit e `git push` nesta branch.

Retoma sozinho: se a máquina reiniciar ou o processo cair, rode o mesmo comando; fases prontas são puladas e
a fase pela metade continua do `last.pth`. Ele mesmo tenta de novo até 5 vezes. Acompanhe com
`tail -f results/logs/me1_multisubj_maquinaN.log` e `nvidia-smi`.

Na primeira época do pré-treino, anote o tempo por época (`t_epoch` no `log.jsonl`) e estime o total
(150 épocas × 6 fases) no registro. Os checkpoints ficam em `~/me1_multisubj/train_logs/` (~4–5 GB cada);
depois que o ajuste de 40 sessões terminar, só o `last.pth` do pré-treino precisa ficar.

### Etapa 5. Conferir o fim

O log termina com `MAQUINA N: TUDO PRONTO` e `--lista` mostra `pre 1 5 10 20 40` na linha da sua máquina.
`git log origin/ME1-multisubj` tem 6 commits "ME1 multi-sujeito, maquina N: ...". Se algum push falhou,
rode `git pull --rebase && git push`.

### Etapa 6. Se for a última máquina a terminar

Rode `--lista`. Se as 6 linhas estiverem completas, faça a seção 7; senão, avise o Pedro e pare.

## 7. Consolidação (uma vez, pela última máquina)

1. Script `scripts/me1_multisuj_tabela.py` que lê `results/me1_multisubj/*/final_retrieval.json` e `args.json` e
   grava `results/tables/me1_multisubj.csv`: uma linha por ajuste com sujeitos, sessões do pré-treino, M, fwd,
   bwd (média e IC), passos de treino e tempo total.
2. Figura `results/figs/me1_multisubj.png`: fwd e bwd × M (eixo log), uma curva por pré-treino, mais as
   referências do MindEye1 de um sujeito só do benchmark (`subj01_me1_<M>sess`, números no README da `main`)
   e, se estiverem no repositório, a do pré-treino completo `me1_finetuned_subj01_40sess`.
3. Seção "Protocolos multi-sujeito" no [MINDEYE1.md](MINDEYE1.md) com a grade, os desvios (seção 10) e 3–5
   frases respondendo: o pré-treino ajuda mais quando o subj01 tem pouco dado? A dado total fixo, vale mais ter
   mais sujeitos ou mais sessões por sujeito? (pares de mesmo total: 2×10 com 4×5, e 2×20 com 4×10).
4. Commit e push nesta branch. **Não** abra PR para a `main` sem o Pedro pedir.

## 8. Quando parar e perguntar ao Pedro

- a GPU tem menos de 20 GB ou o teste de fumaça não fecha depois de ~2 h de tentativas;
- uma fase desiste depois das 5 tentativas;
- o push pede login (o Pedro faz o `gh auth login`);
- algum ajuste com 40 sessões do subj01 sai mais de 10 pontos abaixo do MindEye1 de um sujeito só com 40
  sessões (94,5 / 87,6), o que sugere erro de dados.

## 9. Registro da execução

Cada máquina acrescenta uma linha (commit junto com o primeiro resultado):

| Máquina | GPU | RAM | Início | Tempo por época (pré / ajuste 40) | Fim | Observações |
|---|---|---|---|---|---|---|

## 10. Receita e decisões

Tudo como o `run_me1_multisubject.sh` (o protocolo do MindEye2): 150 épocas por fase, OneCycle com
`max_lr=3e-4`, BiMixCo no primeiro terço e SoftCLIP depois, `prior_scale=30`, `clip_scale=1`, `hidden_dim=1024`,
`n_blocks=4`, fp16, seed 42, `--fix_multisubj_perm` desligado, sem low-level. O que muda:

| Item | Antes | Agora | Por quê |
|---|---|---|---|
| Sujeitos do pré-treino | todos menos o de teste | `--pretrain_subjs` | a grade |
| Sessões por sujeito | todas | `--pretrain_sessions N` (as N primeiras) | a grade |
| Amostras por época no pré-treino | `750 × 40` fixo | `750 × N` | a regra que o MindEye2 usa no ajuste de um sujeito: a época encolhe com o dado |
| Batch global | 21 | 24 com 2 e 4 sujeitos, 21 com 7 | manter ~21–24 exemplos na perda contrastiva |
| Ajuste no subj01 | 40 sessões | 1, 5, 10, 20, 40 (`750 × M` amostras por época, batch 24) | a grade |

Sem `--pretrain_subjs`/`--pretrain_sessions`, o `Train_MindEye_multisubject.py` faz exatamente o que fazia
antes (o `run_me1_multisubject.sh` antigo não muda). No pré-treino, o retrieval por época é medido no primeiro
sujeito da lista (2), como no MindEye2; o que vale é a avaliação dos ajustes no subj01.
