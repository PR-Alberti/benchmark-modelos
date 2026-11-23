# Procedimento: protocolos multi-sujeito do MindEye1 (branch `ME1-multisubj`)

> **Para o Claude no computador do laboratório.** Este arquivo é a ordem de serviço completa. Execute as
> etapas na ordem, sem pedir confirmação para o que está decidido aqui. Pergunte ao Pedro só o que estiver
> marcado como **[PERGUNTAR]** ou o que contradisser o que você encontrar. Siga o `CLAUDE.md` do repositório
> (autor `PR-Alberti`, nunca co-autoria do Claude). Os commits desta tarefa vão **nesta branch**, não na `main`.

Origem: tarefa "Determinar protocolos multi-subj" da semana 23–27/11/2026 no Notion.

## 1. Objetivo

Medir como o **pré-treino em outros sujeitos** ajuda o MindEye1 no sujeito de teste (subj01), variando
**quantos sujeitos** e **quantas sessões de cada um** entram no pré-treino, e quanto dado do subj01 entra no
ajuste. Hoje só existe o protocolo "7 sujeitos × todas as sessões" ([MINDEYE1.md](MINDEYE1.md#mindeye1-multi-sujeito)).

## 2. A grade (decidida)

Pré-treinos (marcados com x no Notion), cada um com **as N primeiras sessões de cada sujeito do pré-treino**:

| | 1 sessão | 5 sessões | 10 sessões | 20 sessões | 40 sessões |
|---|---|---|---|---|---|
| 2 sujeitos | | | x | x | |
| 4 sujeitos | | x | x | | |
| 7 sujeitos | x | x | | | |

| Grupo | Sujeitos do pré-treino | Por quê |
|---|---|---|
| 2 sujeitos | **2, 5** | têm 40 sessões |
| 4 sujeitos | **2, 5, 7, 3** | os três com 40 sessões + o 3 (32 sessões; 10 cabem) |
| 7 sujeitos | **2, 3, 4, 5, 6, 7, 8** | todos menos o de teste |

São **6 pré-treinos**. Cada um é seguido de **5 ajustes no subj01**, com 1, 5, 10, 20 e 40 sessões dele →
**30 ajustes**. Total: 36 corridas.

Nomes (pasta em `$ROOT/train_logs/`):

- pré-treino: `me1ms_pre_<S>subj_<N>sess` — ex.: `me1ms_pre_2subj_10sess`;
- ajuste: `me1ms_ft_<S>subj_<N>sess__subj01_<M>sess` — ex.: `me1ms_ft_2subj_10sess__subj01_5sess`.

Referências que já existem e entram nas tabelas para comparar (não treinar de novo, a menos que não estejam na
máquina): `me1_finetuned_subj01_40sess` (7 sujeitos, todas as sessões; o `run_me1_multisubject.sh`) e o MindEye1
de um sujeito só do benchmark (`subj01_me1_<M>sess`).

## 3. Decisões de receita

Tudo igual ao `run_me1_multisubject.sh` atual (150 épocas por fase, `max_lr=3e-4`, BiMixCo no 1º terço e
SoftCLIP depois, `prior_scale=30`, `clip_scale=1`, `hidden_dim=1024`, `n_blocks=4`, fp16, seed 42,
`--fix_multisubj_perm` **desligado**), exceto:

| Item | Hoje | Agora | Por quê |
|---|---|---|---|
| Sujeitos do pré-treino | todos menos o de teste | lista explícita (`--pretrain_subjs 2 5`) | a grade |
| Sessões por sujeito no pré-treino | todas (`NSESSIONS_ALLSUBJ`) | as N primeiras (`--pretrain_sessions N`) | a grade |
| Amostras por época no pré-treino | `750 × 40`, fixo | **`750 × N`** | a mesma regra que o MindEye2 usa no ajuste de um sujeito (`750 × num_sessions`): com pouco dado, a época encolhe junto e o número de passos fica proporcional ao dado |
| Batch global do pré-treino | 21 (3 por sujeito, 7 sujeitos) | **3 por sujeito × 7 = 21; 6 × 4 = 24; 12 × 2 = 24** | manter o batch total perto de 21–24 para a perda contrastiva ter negativos parecidos |
| Ajuste no subj01 | 40 sessões | 1, 5, 10, 20 ou 40 (`--num_sessions M`), batch 24, `750 × M` amostras por época | a grade |
| Sujeito de teste no pré-treino | `subj_list[0]` | idem (o 1º da lista: 2) | só monitora; a avaliação que vale é a do ajuste |

Se 150 épocas com `750 × 1` amostras derem uma corrida curta demais para o OneCycle (menos de ~300 passos no
total), mantenha assim mas anote o número de passos no resultado. **Não** mude a receita sem registrar aqui.

## 4. Mudanças no código

1. `mindeye1/src/Train_MindEye_multisubject.py`:
   - `--pretrain_subjs` (lista de ints; padrão: todos menos `--subj`, que é o comportamento de hoje);
   - `--pretrain_sessions` (int; padrão: `NSESSIONS_ALLSUBJ` de cada sujeito, como hoje). Se um sujeito tiver
     menos sessões que o pedido, **pare com erro** (não pode acontecer na grade acima);
   - `num_samples_per_epoch = 750 * pretrain_sessions` no modo multi-sujeito quando `--pretrain_sessions` for
     dado (sem ele, o `750 * 40` de hoje, para não mudar a corrida antiga);
   - batch: aceite `--batch_size` global como hoje; o script de fora passa 21/24/24.
   - grave tudo no `args.json` (já grava os argumentos).
2. Novo `mindeye1/src/run_me1_ms_protocolos.sh` — **um experimento por PC**:

   ```bash
   mindeye1/src/run_me1_ms_protocolos.sh --lista                 # as 6 configurações e o que já terminou
   mindeye1/src/run_me1_ms_protocolos.sh 2subj_10sess            # pré-treino + os 5 ajustes desse pré-treino
   FT_SESSOES="1 5" mindeye1/src/run_me1_ms_protocolos.sh 2subj_10sess   # só alguns ajustes
   SO_PRE=1 mindeye1/src/run_me1_ms_protocolos.sh 7subj_1sess    # só o pré-treino
   ```

   - mesma estrutura do `run_me1_multisubject.sh`: `ROOT`, `PY`, `MAX_RESTARTS`, fase pulada se tem
     `final_retrieval.json`, retomada do `last.pth`;
   - no fim de cada fase, copia `args.json`, `log.jsonl` e `final_retrieval.json` (sem o `last.pth`) para
     `results/me1_multisubj/<nome>/` no repositório, faz commit **só desses arquivos** e `git pull --rebase &&
     git push` nesta branch. Como cada PC escreve em pastas diferentes, não há conflito;
   - log de uma linha por fase em `results/logs/me1_multisubj_<config>.log`.
3. `tests/test_multisujeito_me1.py` (sem GPU): o parser aceita as opções, a conta de amostras/batch por
   configuração dá os números da seção 3, e o `--lista` mostra as 6.

Rode um **teste de fumaça** antes de distribuir: `--max_iters_per_epoch 2 --num_epochs 3` numa configuração
pequena (`7subj_1sess` + ajuste de 1 sessão), até gravar `final_retrieval.json`. Só depois espalhe pelos PCs.

## 5. Preparar cada PC

```bash
git clone https://github.com/PR-Alberti/benchmark-modelos.git && cd benchmark-modelos
git checkout ME1-multisubj
setup/bootstrap.sh --multisujeito        # ambiente + sujeitos 2–8 (12,1 GB de betas e tars)
```

O `run_me1_multisubject.sh` usa dados no formato do MindEye2 em `$ROOT/data/mindeyev2` e o Python de `PY`
(padrão `~/.conda/envs/mindeye1ms`). Se esse ambiente não existir, use `PY=~/envs/fmri/bin/python` e
`ROOT` apontando para onde o bootstrap pôs os dados (`ROOT/data/mindeyev2` pode ser um link para
`~/mindeyev2`); confirme no teste de fumaça. Cada PC precisa de 20 GB de GPU e ~30 GB de RAM.

Distribuição sugerida (6 PCs, um por pré-treino); com menos PCs, enfileire na ordem
`7subj_1sess, 4subj_5sess, 2subj_10sess, 7subj_5sess, 4subj_10sess, 2subj_20sess` (do mais barato ao mais
caro). Antes de começar, rode cada configuração 1 época e anote o tempo por época aqui, para estimar.

## 6. Avaliação e entregáveis

Avaliação: a que o script já faz — retrieval no teste novo do subj01 (1.000 imagens, 3 repetições
promediadas), sorteio do benchmark (30 × 300), contra o CLIP ViT-L/14 da imagem (`final_retrieval.json`, fwd e
bwd). Reconstrução não entra (o multi-sujeito não tem low-level).

1. Tabela `results/tables/me1_multisubj.csv`: uma linha por ajuste (S sujeitos, N sessões no pré-treino, M sessões
   do subj01, fwd, bwd, passos de treino, tempo).
2. Figura: fwd × M (sessões do subj01), uma curva por pré-treino, mais as duas referências (sem pré-treino e
   7 sujeitos × todas as sessões).
3. Seção nova em [MINDEYE1.md](MINDEYE1.md) ("Protocolos multi-sujeito") com a grade, os desvios e 3–5 frases:
   o pré-treino ajuda mais quando M é pequeno? Mais sujeitos ou mais sessões por sujeito, a dado total fixo
   (compare 2×10 com 4×5, e 2×20 com 4×10 — os pares de mesmo total que a grade permite)?
4. Commits nesta branch, com push. **Não** abra PR para a `main` sem o Pedro pedir.

## 7. Quando parar e perguntar

- o teste de fumaça não fecha;
- uma fase cai mais de `MAX_RESTARTS` vezes;
- o retrieval de algum ajuste com 40 sessões do subj01 sair muito abaixo do `me1_finetuned_subj01_40sess`
  (mais de 10 pontos), o que sugere erro na seleção de sessões ou sujeitos.

## 8. Registro da execução

_(preencher: qual configuração rodou em qual PC, GPU, datas, tempo por época, desvios do plano)_
