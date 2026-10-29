# Revisão do benchmark: a ideia de decompor os modelos e críticas gerais

Revisão feita em 22/10/2026 sobre o commit `7617c12`, no formato de "conselho" do
[claude-council](https://github.com/hex/claude-council): revisores independentes, cada um com uma lente,
respondem sem ver os outros; numa segunda rodada cada um lê as respostas dos outros, confere no código
as afirmações centrais e revisa a própria lista. As lentes:

| Revisor | Lente |
|---|---|
| Arquiteto | neurodecodificação: peças, interfaces, comparabilidade entre modelos |
| Estatístico | desenho experimental, incerteza, apresentação dos números |
| Cético | confounds, vazamentos, narrativas que vão além dos dados |
| Pragmático | operação, custo de GPU, menor passo útil |
| Engenheiro de software | arquitetura do repositório, testes, reprodutibilidade |

**Ressalva de método.** Os cinco revisores são o mesmo modelo (Claude) com papéis diferentes. Concordância
entre eles é sinal fraco: o valor está nos ângulos, não no consenso. O que tem mais peso é o que foi
**conferido no código** por mais de um revisor; essas afirmações estão marcadas com ✅. Nenhum treino nem
reconstrução foi rodado; os testes (176, sem GPU) passaram.

**Caminhos.** Os `arquivo:linha` citados são do commit `7617c12`, antes da reorganização das pastas: as
etapas que estavam em `src/` (`train_ridgeonly.py`, `recon_inference.py`, `enhanced_recon_inference.py`,
`recon_embeddings.py`, `final_evaluations.py`, `verify_retrieval.py`, `run_frr.py`, `run_frr_ms.py`)
estão hoje em `src/etapas/`, e `src/generative_models/` e `src/autoencoder/` em `vendor/`.

Índice: [1. Decompor os modelos em peças](#1-decompor-os-modelos-em-peças) ·
[2. Críticas gerais](#2-críticas-gerais) · [3. O que a segunda rodada corrigiu](#3-o-que-a-segunda-rodada-corrigiu) ·
[4. O que está bem feito](#4-o-que-está-bem-feito) · [5. Prioridades](#5-prioridades) ·
[6. Divergências que ficaram](#6-divergências-que-ficaram) ·
[7. Conferência da revisão](#7-conferência-da-revisão-23102026)

> **Leia a seção 7 junto com as anteriores.** A conferência de 23/10/2026 achou afirmações erradas ou
> desatualizadas nesta revisão (o ruído de semente nas métricas de imagem já pode ser medido, a conta do
> fatorial, o alcance do `ckpt_interval`, a "lei de escala") e revisa a lista de prioridades.

---

## 1. Decompor os modelos em peças

A pergunta: quebrar os modelos do benchmark em peças e testar encaixes entre elas, para entender a
contribuição de cada peça isolada e em conjunto.

**Veredito: vale a pena, numa versão restrita.** "Contribuição isolada" de uma peça não é bem definida
quando as peças são treinadas juntas. O que dá para medir com rigor são **dependências, interações e
pisos**, trocando peças nas fronteiras que já são congeladas.

### As peças

| Peça | Onde vive | Interface | ME2 ridge-only | FRR | FRR-MS | ME1 |
|---|---|---|---|---|---|---|
| Dados e seleção | `dataset_controlado`, `nsd_data.carrega_betas` | betas 15.724 | ✓ | ✓ | ✓ (+ sujeitos 2–8) | ✓ |
| Agregação no treino | `agregacao.seleciona` | (n, 15.724) | exibições | exibições/média | idem | rodízio |
| Agregação no teste | `recon_inference.py:360-373`; `run_frr.py`; `Reconstructions.py:387` | — | **média das saídas** | média dos betas | média dos betas | média dos betas |
| Projeção por sujeito | `models.RidgeRegression`; `multisujeito.passo_R` | (B, h) | treinada | — | R₁ | entrada da MLP |
| Espaço compartilhado | `BrainNetwork` congelado; `Q` | latente | não linear, contrastivo | — | linear, MSE | — |
| Mapeamento até o CLIP | `backbone_linear`; `frr.FracRidge`; `lin1` do ME1 | bigG 256×1.664 / ViT-L 257×768 | ✓ | ✓ direto | Q | ✓ |
| Diffusion prior | `BrainDiffusionPrior`; prior do VD | embedding "plausível" | ✓ | — | — | ✓ |
| Ramo low-level | `blin1/bupsampler`; `Voxel2StableDiffusionModel` | latente 28×28 / 64×64 | só no 4096 | — | — | ✓ |
| Gerador | `decodificador.carrega_unclip`; Versatile Diffusion | imagem | unCLIP | unCLIP | unCLIP | VD + img2img 0,85 |
| Legenda | conversor bigG→ViT-L (257×1.024) + GIT | texto | ✓ | ✓ | ✓ | — |
| Refino e mistura | `enhanced_recon_inference.py`; `final_evaluations.py:171` | imagem 256 | ✓ | ✓ | ✓ | — |

### O que encaixa

- **Já encaixa: o contrato `(n, 256, 1664)` em bigG.** Tudo que grava `<nome>_all_clipvoxels.pt` nessa
  forma segue para unCLIP → refino → métricas (`recon_embeddings.py`). É assim que o FRR já reconstrói.
- **Precisa de adaptador:** ME1 → unCLIP/GIT (não há conversor ViT-L 768 → bigG); bigG → VD (o
  `bigG_to_L` gera 257×**1.024**, o espaço do GIT, não o 257×768 do VD); low-level entre ME1 e ME2
  (latentes de VAEs e resoluções diferentes; trocar em pixels).
- **Não encaixa sem retreino:** a ridge do FRR no lugar da ridge do ME2 (o backbone espera o latente
  dele); R₁ do FRR-MS na entrada do `BrainNetwork` (mesma dimensão, outra base); o prior de um modelo no
  backbone do outro; o `clip_proj` sem o backbone.

### Por que "contribuição isolada" engana aqui

- **Peças co-treinadas.** Tirar a perda do prior derruba o retrieval de 92,8 para 78,3, embora o
  retrieval nem passe pelo prior: ele contribui como **sinal de treino**, não como peça de inferência.
  Toda ablação tem de dizer se a peça saiu do treino, da inferência ou dos dois.
- **ME1 com e sem low-level dá retrieval idêntico por construção.** São treinos separados e o
  retrieval não passa pelo low-level (`experimentos.py:191-195`): é um teste de determinismo. Na imagem,
  tirar o low-level derruba o PixCorr (0,252 → 0,132) e *sobe* o CLIP (80,8% → 81,4%) — a contribuição
  muda de sinal conforme a métrica e depende do `img2img_strength`.
- **Interações fortes, já nos dados.** Ganho do multi-sujeito no FRR (Img→cér): +21,8 pt com 1 sessão,
  −0,5 com 20, −0,9 com 40. Um "efeito do multi-sujeito" sem dizer o regime de dados não significa nada.
- **Fatorial completo é inviável.** Com níveis realistas são ~4.600 células, muitas inexistentes (o
  gerador depende do espaço alvo). O desenho certo é **hierárquico**, com fatoriais pequenos. Shapley
  pode ser calculado a partir de um 2^k completo para apresentar, mas não é desenho.

### Plano, em ordem de custo

1. **Infraestrutura, quase sem GPU.** Gravar o `prior_out` do ME2 (hoje o `_all_clipvoxels.pt` é a saída
   da cabeça contrastiva, `recon_inference.py:375-384`; o `prior_out` não é salvo); `recon_embeddings.py`
   com `--entrada <arquivo>`; métricas por imagem; mistura 75/25 como parâmetro.
2. **Encaixes por symlink, minutos cada.** Um `scripts/monta_encaixe.sh <novo> embedding=<a> blurry=<b>`
   que monta `results/evals/enc_<novo>/` com peças emprestadas e roda `run_evals.sh`. Declarados num
   dicionário `ENCAIXES` em `experimentos.py`, com prefixo `enc_` e nunca sobrescrevendo modelos reais.
3. **Encaixes só de inferência, ~3 h cada.** Cabeça contrastiva → unCLIP sem prior (o prior na
   inferência); legenda verdadeira no refino; FRR → prior do ME2 → unCLIP (atenção: a saída do FRR está
   encolhida para a média, cosseno centrado 0,12, e cai fora da distribuição que o prior viu).
4. **Fatorial 2³ com 1 sessão: prior × hidden_dim × blurry**, ~25 GPU-h. Exige o blurry como cabeça
   separada sobre o backbone 1024 (ver seção 3).
5. **Prior × sessões**: Ridge 1024 sem prior com 40 sessões, mesmas épocas do par com prior, ~10 h.
6. **Curva de sessões do FRR e do FRR-MS (5, 10)**, ~1 h: onde o ganho do multi-sujeito troca de sinal.

**Não fazer:** reescrever os modelos como `nn.Module` encaixáveis (a costura por `.pt` basta); abrir o
ME2 por dentro (ridge do FRR antes do backbone congelado) antes do resto; encaixar o ME1 com os bigG.

---

## 2. Críticas gerais

### 2.1 Comparabilidade entre modelos

- ✅ **Agregação no teste diferente.** O ME2 passa cada repetição pela ridge e pelo backbone e tira a
  média das saídas (`recon_inference.py:360-373`); FRR e ME1 tiram a média dos betas antes. Com o
  backbone não linear, as duas medem quantidades diferentes, e nenhuma tabela registra isso.
- ✅ **Retrieval em espaços e cabeças diferentes.** ME2: `clip_proj`, treinada com perda contrastiva.
  FRR: o vetor de 425.984 números ajustado por erro quadrático. ME1: ViT-L 257×768. O cosseno bruto
  favorece a cabeça contrastiva e pune a FRR por hubness (Cér→img 3,1% bruto, 31,7% centrado).
- ✅ **Mistura 75/25 disparada pelo nome.** `final_evaluations.py:171-173`:
  `"enhanced" in model_name_plus_suffix and all_blurryrecons is not None`. O PixCorr e o SSIM
  "refinados" do 4096 e do artigo vêm em parte da imagem borrada; FRR e 1024 não têm esse ramo; o ME1 usa
  a borrada como init do img2img, com outro peso efetivo.
- **A "1 sessão" do ME2 não é 1 sessão.** Backbone e prior vêm de um pré-treino de 7 sujeitos × 40
  sessões (~280 h de fMRI); o ME1 treina do zero. "~1 h de fMRI" (`BENCHMARK.md:42`) é enquadramento,
  não orçamento de dados. A comparação ME1 × ME2 mede o pré-treino, não a arquitetura.
- **A FRR como "baseline linear".** A ridge do ME2 também é linear, mas seguida de 713 M de parâmetros
  congelados. A comparação é "MSE direto ao CLIP" contra "um sistema pré-treinado inteiro".
- **Tetos e pisos do gerador ausentes.** Sem imagem verdadeira → embedding → gerador (teto) e embedding
  médio/embaralhado → gerador (piso), não dá para separar decodificador de gerador nem saber quanto do
  Incep/CLIP em torno de 0,8 vem do cérebro.

### 2.2 Incerteza e apresentação dos números

- **Orçamento desigual.** 150 épocas com 1 sessão, 20 com 40 sessões e congelados em fp16; batch 16 no
  1024, 8 no 4096, 24 no artigo. "Fica de 1 a 3 pontos do artigo" mistura método e orçamento.
- ✅ **IC de 30 sorteios sobrepostos.** `embedding_metrics.py:49-51`: `std/sqrt(30)` sobre reamostras de
  300 das mesmas 1.000 imagens. Mede a variação do sorteio, não a incerteza sobre imagens ou sementes.
- ✅ **Métricas por imagem calculadas e descartadas.** `two_way_identification` tem `return_avg=False`
  (`final_evaluations.py:274`, `305-309`), mas todas as chamadas usam a média. O CSV é `Metric/Value`,
  sem modelo, semente, commit ou checkpoint. Sem dados por imagem não há bootstrap pareado nem teste.
- **Negrito sem margem.** `rows.py:12-13` marca o máximo (tratando empates); falta uma tolerância
  estatística. SSIM 0,421 contra 0,414 e 100,0% contra 99,9% recebem negrito.
- **"À frente do artigo em 9 das 10 métricas"** (`BENCHMARK.md:27`) conta métricas correlacionadas
  como evidências independentes, e a vantagem (1,5 a 2,2 pt) é comparada só com o ruído de
  reconstrução (0,6 pt). O efeito da semente de treino nas métricas de imagem "não foi medido"
  (`BENCHMARK.md:32`), e as reproduções se afastam dos originais em até 1,24 p.p.
- **Teto.** Com 40 sessões o retrieval entre 300 satura (99,9%). Faltam top-1 entre 1.000, rank médio,
  escala logit.
- **Política de semente única.** `experimentos.py:20-24` diz que, com a semente fixa, "a diferença
  entre duas corridas vem só da configuração". Fixar a semente congela uma realização do ruído; quando
  a configuração muda, o fluxo aleatório muda junto. A política proíbe justamente a medida que falta.
- **Teste usado durante o treino.** O `test_dl` roda a cada época (`train_ridgeonly.py:367-372`). O
  checkpoint é sempre o último (sem seleção pelo teste), mas escolhas como 20 épocas para 40 sessões,
  `prior_scale=30` ou a grade do FRR foram feitas olhando curvas de teste. Não há validação separada.
- **Um sujeito só.** Tudo vale para o subj01; a variância entre sujeitos é desconhecida.

### 2.3 Narrativas que vão além dos dados

- **"Lei de escala"** (`docs/ESTAGIO.md:7`, apresentação): 2 pontos de sessões no ME1 e no ME2, 3 no FRR.
- **"Com pouco dado, o pré-treino é o que mais vale"** (`src/report/conteudo/apresentacao.html:374`):
  ME1 × ME2 difere também em arquitetura, espaço CLIP, gerador, refino e agregação. O contraste limpo é
  FRR-MS × posto reduzido (+21,8 pt com 1 sessão), e ele não é o citado.
- **"Acima dos dois em retrieval"** compara retrieval em espaços CLIP diferentes, o que o próprio
  `BENCHMARK.md:31` diz não ser comparável.
- **"O prior importa"** (`BENCHMARK.md:29`): provavelmente sobrevive (14 pt), mas vale só para 1 sessão,
  hidden 1024 e uma semente.
- **"Treinado de novo, dá o mesmo resultado?"** (apresentação): mesmo código e mesma semente testam
  determinismo, não reprodutibilidade. `docs/REEXECUTAR.md` é honesto sobre isso; o slide, não.

### 2.4 Vazamento

- **Captioner treinado em COCO.** O refino usa a legenda do `microsoft/git-large-coco`
  (`decodificador.py:29`). As 1.000 imagens de teste do NSD são todas `cocoSplit = train2017`
  (`nsd_stim_info_merged.csv`), e provavelmente mais de 90% estão no treino de legendas (split
  Karpathy). **Provável, não verificado imagem a imagem.** O viés não é igual para todos: só os modelos
  com refino o recebem, o que distorce ME1 × ME2. unCLIP e VD vêm de LAION; não auditado.
- **Checkpoint multi-sujeito do ME2.** O `run_frr_ms.py` confere se o shared1000 vaza pelos sujeitos
  2–8; não há verificação equivalente para o `multisubject_subj01_1024hid_nolow_300ep` baixado, base de
  todo o ridge-only.

### 2.5 Proveniência dos números

- **Dois avaliadores de retrieval sem rótulo.** O mesmo checkpoint dá 92,76 no `final_evaluations`
  (`results/tables/repro_subj01_ridgeonly_1sess_prior_all_enhancedrecons.csv`) e 92,91 no
  `verify_retrieval` (`docs/REEXECUTAR.md:57`); o `BENCHMARK.md` mistura os dois (`:46` e `:132`).
- **README com números antigos.** `README.md:186` (78,5/72,1 sem prior) não bate com nenhum dos dois
  avaliadores (78,29 e 78,41).
- ✅ **`fonte()` troca para `repro_` em silêncio.** `report/config.py:64-88` lê `repro_<mid>` assim que
  a tabela existe, com o mesmo id e o mesmo rótulo. A tabela pode mudar sem mudança de código.
- **`BENCHMARK.md` editado à mão depois de gerado.** O cabeçalho diz "commit `1aec3b2`", mas o commit
  `bc85145` mudou linhas de tabela sem regenerar. Rodar o gerador hoje desfaria a edição.
- **Originais substituídos.** O commit `bc85145` apagou 30 tabelas `subj01_*`; elas continuam em
  `git show bc85145^`, mas nada no site aponta para elas. Os dicionários dos originais foram escritos
  depois (`REEXECUTAR.md:112-114`).
- **Conclusões à mão.** O resumo em `src/report/texts.py:18-19` é texto fixo ("revisar se os números
  mudarem"); os adjetivos ("à frente", "empata") não acompanham os números.

### 2.6 Arquitetura do repositório

- **Scripts-notebook.** `recon_inference.py:111`, `final_evaluations.py:93`, `train_ridgeonly.py:223` e
  `enhanced_recon_inference.py:82` injetam argumentos com `globals()[...] = ...`; `final_evaluations.py:535-538`
  apaga variáveis de `globals()`. Esse código não se importa nem se testa: **o caminho científico
  (treino, reconstrução, métricas) não tem nenhum teste**. Restos do cluster original: `/weka/...`
  (`recon_inference.py:50`), `/fsx/...` (`mindeye1/src/train_autoencoder.py:114`).
- **Contratos por sufixo de arquivo.** `_all_enhancedrecons`, `_all_recons`, `_published_`, `_frr.json`
  aparecem literais em ~10 arquivos (`report/config.py`, `report/catalogo.py`, `report/data.py`,
  `report/galeria_site.py`, `make_comparison.py`, `run_recon.sh`). Não há um módulo de layout de
  artefatos, e comportamento muda pelo nome (a mistura 75/25).
- **Registro de modelos em cinco lugares.** `experimentos.py`, `report/config.py` (`MODELOS`),
  `report/catalogo.py`, `report/apresentacao.py` e `run_benchmark.sh`. Os hiperparâmetros do site
  ("150 épocas · batch 16") são texto, não vêm do `experimento.json`.
- **Retomada pela existência do arquivo.** `run_recon.sh:60` e `run_benchmark.sh:36` pulam etapas se o
  `.pt` ou o marcador existir, sem conferir a configuração ou o checkpoint que os gerou. Uma
  reconstrução feita com checkpoint antigo é pulada para sempre.
- **Dois padrões para o mesmo hiperparâmetro.** `ckpt_interval` é 999 no shell
  (`run_ridgeonly_prior.sh:93`) e 5 no argparse (`train_ridgeonly.py:181`). Com 999, o `last.pth` só é
  gravado na época 0 e no fim; os treinos longos sobrescrevem com 10 ou 1 (`experimentos.py:65,68,72`),
  então o risco real é perder até ~2 h no Ridge 1024 de 1 sessão.
- **Três interfaces de execução.** Scripts soltos com variáveis (README:80-109), `run_benchmark.sh`
  (nomes do benchmark original, `set -e`, para na primeira falha) e `run_reexecutar.sh` + `treina.py`
  (o caminho de fato usado hoje).
- **Código vendorizado duplicado.** `mindeye1/src/models.py` e `src/mindeye_ridge/models.py` repetem
  `Clipper`, `BrainNetwork`, `BrainDiffusionPrior` com implementações diferentes; os dois `utils.py`
  repetem ~10 funções. `mindeye1/src` importa `utils` e `models` como módulos de topo, e
  `tests/test_agregacao.py:14` põe esse diretório no `sys.path`: colisão de nomes à espera de acontecer.
  Não unificar (custo alto); isolar o `sys.path` basta.
- **Testes.** Cobrem bem a lógica pura (dataset controlado, agregação, FRR contra o `fracridge`,
  relatório, `treina.py --dry_run`, `common.sh`). `tests/test_site.py:289-300` testa o site commitado
  em vez do gerador. Não há CI.
- **Reprodutibilidade.** Versões fixadas em `setup/setup_env.sh`, mas sem lockfile das transitivas;
  sem sha256 de checkpoints e dados (`restore_ckpt.py`, `download_data.py`).
- **Site e relatórios.** Dois geradores (`make_benchmark.py` e `make_site.py` + `src/report/`, ~3.400
  linhas). `results/site` está no git (149 arquivos, ~14 MB, regerados em 35 commits) e é regenerado ao
  fim de cada etapa (`common.sh:66`), o que suja a árvore e faz os commits de treino saírem como
  `+alteracoes` (daí o remendo `COMMITA=1`). `index_anterior.html` (728 KB) é uma versão antiga commitada.

### 2.7 Operação e quem chega novo

- `docs/NOVO_COMPUTADOR.md:46` manda fazer `git checkout controlled-dataset`, uma branch já mesclada e
  atrás da `main` (sem o `frr_ms`); `:106` e `:126` omitem o `frr_ms`; `:188-189` dão tempos do ME1
  (~5 h e ~20 h) que não batem com os medidos (`REEXECUTAR.md:27-28`: 1 h 40 e 16 h 03).
- Tabelas de disco e tempo repetidas em `README.md`, `SETUP.md` e `NOVO_COMPUTADOR.md`, já divergentes.
- O `--ckpts` vem de um repositório privado: um externo só reproduz os `repro_`. Dizer isso logo no README.

---

## 3. O que a segunda rodada corrigiu

Afirmações da primeira rodada que, conferidas no código, estavam erradas ou exageradas:

| Afirmação | Correção |
|---|---|
| "Determinístico" contradito por 92,8 × 92,9 e 78,3 × 78,4 | São **dois avaliadores** do mesmo checkpoint, não dois treinos. O problema é a falta de rótulo |
| `ckpt_interval` 999 perde horas de treino | Só afeta quem herda o padrão (Ridge 1024, 1 sessão, ~2 h). O problema é haver dois padrões |
| Aliasing hidden × blurry imposto pelo `treina.py:142` | Vem dos checkpoints que existem (só há 4096 com blurry). Liberar a trava não resolve; é preciso treinar o blurry como cabeça separada sobre o 1024 congelado |
| Negrito por máximo estrito | Trata empates; o que falta é margem estatística |
| Vazamento COCO pelo GIT | Rebaixado para "provável, a auditar" |
| `BENCHMARK.md` "não regenerável" | É regenerável por design (`make_benchmark.py:50`); foi editado à mão depois e o gerador desfaria a edição |
| Originais "apagados" | Estão no histórico do git; falta apontar para eles (uma tag `originais-v1`) |
| "Lei de escala com 2 pontos" | Vale para ME1 e ME2; o FRR tem 3 (1, 20, 40), e mais pontos custam minutos |

---

## 4. O que está bem feito

Não mudar:

- `treina.py`: recusa sobrescrever modelos do benchmark e mudar a configuração de um nome existente,
  filtra o ambiente (`treina.py:290-294`), registra o commit com sufixo de árvore suja.
- `paths.py` + `common.sh` como fonte única de caminhos; `confere_dataset` com hash do manifesto.
- A fila tolerante a falhas com marcadores por etapa em `run_reexecutar.sh`.
- Os 176 testes rápidos e sem GPU, incluindo o FRR contra a referência.
- Os sorteios de retrieval iguais para todos os modelos (`seed_everything(42)`): o desenho já é pareado.
- `generative_models` vendorizado sem modificação; versões fixadas com comentários de compatibilidade.
- Métricas pequenas (CSV/JSON) no git, tensores fora.
- A costura por arquivo `.pt` entre etapas: é o que torna os encaixes da seção 1 baratos.

---

## 5. Prioridades

| # | Ação | Custo | Por quê |
|---|---|---|---|
| 1 | **Protocolo de teste explícito:** mistura 75/25 como flag gravada no CSV; uma agregação no teste para todos; retrieval bruto e centrado lado a lado; tabela **unCLIP, sem refino**, como primária entre famílias | baixo | Sem isso as comparações não medem a mesma coisa |
| 2 | **Métricas por imagem em formato longo, com metadados** (modelo, config, commit, semente de treino e de reconstrução) → bootstrap pareado sobre as 1.000 imagens; negrito só quando o IC da diferença exclui o zero | 1 tarde, sem GPU | Base de qualquer inferência |
| 3 | **Controles de teto, piso e legenda** via `recon_embeddings.py` (embedding verdadeiro, médio e embaralhado → gerador; refino com legenda verdadeira e vazia); auditoria do split Karpathy | ~6 GPU-h | Separa decodificador de gerador e mede o vazamento |
| 4 | **Proveniência:** rotular os dois avaliadores; corrigir o README; fonte (`repro_` ou original) como campo explícito no registro; tag dos originais; CI que falha se `BENCHMARK.md` ou o site divergirem do gerador | baixo | Já há números divergentes publicados |
| 5 | **3 sementes nos contrastes de 1 sessão** citados no texto (prior, 1024 × 4096, agregação) e reescrever a justificativa em `experimentos.py:20-24` | ~12 GPU-h | Mede o ruído de treino nas métricas de imagem |
| 6 | **Restringir o escopo das afirmações** do site e da apresentação | baixo | Consequência de 1 a 5 |
| 7 | **Higiene:** um padrão de `ckpt_interval`; `NOVO_COMPUTADOR.md:46`; site fora do git (ou só o HTML) e site regenerado sob demanda; lockfile + sha256; registro único de modelos; aposentar `run_benchmark.sh`; isolar o `sys.path` do ME1 | baixo a médio | Operação e quem chega novo |

Opcionais, depois: FRR nos sujeitos 2, 5 e 7 (barato, testa a generalização); o blurry como cabeça
separada sobre o 1024; o plano de encaixes da seção 1, que se apoia nos itens 1 a 3.

---

## 6. Divergências que ficaram

- **Sementes nas 40 sessões.** Os cinco concordam que a justificativa da semente única está errada e
  que 3 sementes nos contrastes de 1 sessão valem o custo. Estatístico e cético querem também as de 40
  sessões; o pragmático não (8 a 12 h cada, mais ~3 h de reconstrução), e propõe declarar no texto
  "sem réplica; a régua de ruído é a de 1 sessão".
- **`main(args)` nos scripts-notebook.** O engenheiro de software defende um embrulho mecânico, sem
  reescrever o corpo, como pré-requisito de métricas por imagem e flags explícitas. O pragmático acha
  que custa 1 a 2 dias e arrisca a igualdade numérica já conferida, e prefere um teste de fumaça por
  subprocesso com dados falsos (como `scripts/me1_fake_data.py`). As métricas por imagem dá para gravar
  com uma mudança local em `final_evaluations.py`, sem a reestruturação.
- **O que vem primeiro.** O engenheiro de software põe o contrato de artefatos (`artefatos.py` + JSON de
  metadados ao lado de cada saída) em primeiro, por ser pré-requisito de quase tudo. Cético e pragmático
  o veem como risco futuro, não como erro já visível nos números. O caso que já muda números é a
  mistura 75/25 pelo nome, e por isso ele está no item 1 da seção 5.

---

## 7. Conferência da revisão (23/10/2026)

Cada afirmação central das seções 1 a 6 foi conferida no código, nas tabelas de `results/tables/` e, quando
deu, nos dados, já depois da reorganização das pastas (commit `1dc8643`).

### 7.1 Confirmadas

| Afirmação | Evidência |
|---|---|
| Mistura 75/25 disparada pelo nome | `src/etapas/final_evaluations.py:171-173` |
| ME2 tira a média das saídas no teste, FRR e ME1 a dos betas | `src/etapas/recon_inference.py:360-373` |
| `_all_clipvoxels.pt` do ME2 é a saída da cabeça contrastiva; o `prior_out` não é gravado | `src/etapas/recon_inference.py:375-384` |
| IC de 30 sorteios sobrepostos | `src/mindeye_ridge/embedding_metrics.py:49-51` |
| Métricas por imagem calculadas e descartadas | `final_evaluations.py:274`, `305` (`return_avg`) |
| Negrito sem margem estatística | `src/report/rows.py:12-13` |
| Trava blurry ⇒ hidden 4096 | `src/treina.py:141-142`; `scripts/run_ridgeonly_prior.sh:21-36` escolhe o checkpoint pelo `BLURRY` |
| Captioner `microsoft/git-large-coco` | `src/mindeye_ridge/decodificador.py:29-30` |
| **As 1.000 imagens de teste são todas `train2017`** | `nsd_stim_info_merged.csv`: `shared1000` → `{'train2017': 1000}`. O GIT foi afinado no split Karpathy (train2014 + val2014 menos ~10 mil de validação e teste), então a maior parte delas provavelmente esteve no treino de legendas. Segue "provável": não há o `dataset_coco.json` para conferir imagem a imagem |
| Dois avaliadores de retrieval, sem rótulo | `final_evaluations`: 92,76 / 88,10 e 78,29 / 72,08; `verify_retrieval` (`docs/REEXECUTAR.md:51`): 92,91 / 88,12 e 78,41 / 72,59 |
| README com número que não bate com nenhum avaliador | `README.md` (tabela "Combinações testadas"): sem prior 78,5 / 72,1 |
| `fonte()` troca para `repro_` pela existência do arquivo | `src/report/config.py:64-88`. É intencional e está no docstring; o risco é a volta silenciosa ao original quando a tabela da reprodução falta |
| Teste avaliado a cada época, sem validação separada; checkpoint sempre o último | `src/etapas/train_ridgeonly.py:372`, `876`, `1021-1030` |
| ME1 sem low-level: PixCorr 0,252 → 0,132, CLIP 80,8% → 81,4% | `results/tables/repro_subj01_me1_1sess{,_semlowlevel}_all_recons.csv` |
| Ganho do multi-sujeito (C − B, Img→cér): +21,8 pt / −0,4 / −0,9 | `results/tables/subj01_frr_{multisujeito,postoreduzido}_{1,20,40}sess_frr.json` (62,9 − 41,1; 94,1 − 94,5; 96,1 − 97,0) |

### 7.2 Erradas ou desatualizadas

- **"O efeito da semente de treino nas métricas de imagem não foi medido"** (seções 2.2 e 5, e
  `BENCHMARK.md`, Resumo, "Ruído"). Os dados já existem: as reproduções das sementes 1 e 2 do Ridge 1024 +
  prior têm a tabela completa (`results/tables/repro_subj01_ridgeonly_1sess_prior_seed{1,2}_all_enhancedrecons.csv`).
  Com a semente 42, a amplitude entre as 3 sementes, nas refinadas:

  | PixCorr | SSIM | Alex(2) | Alex(5) | Incep | CLIP | Eff | SwAV | Img→cér | Cér→img |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 0,0014 | 0,002 | 0,3 pt | 0,2 pt | 0,5 pt | 0,7 pt | 0,008 | 0,004 | 0,3 pt | 0,4 pt |

  Somada à reamostragem da reconstrução (até 0,6 pt), dá uma régua de ~1 pt nas identificações 2-way.
  São 3 sementes de uma configuração só (o 1024 de 1 sessão), não de cada modelo.
- **"À frente do artigo em 9/10 fica dentro do ruído"** (seções 2.2 e 2.3): exagerada. A vantagem do 4096 +
  blurry sobre o artigo com 1 sessão (Alex(2) +2,0, Alex(5) +1,5, Incep +2,2, CLIP +1,8 pt; PixCorr +0,016;
  Eff −0,019; SwAV −0,014; Img→cér +1,6) fica de ~2,5 a ~10 vezes acima da amplitude entre sementes.
  Somando a reamostragem da reconstrução (até 0,6 pt), o caso de fronteira é o CLIP (+1,8 contra ~1,3). Continua valendo que as 9 métricas são correlacionadas (não são 9
  evidências) e que o artigo é um checkpoint só, de outro regime de treino.
- **"Lei de escala com 2 pontos"** (seção 2.3): injusta. A apresentação diz que o plano *busca* uma lei de
  escala (`src/report/conteudo/apresentacao.html:26`) e lista mais pontos de sessões como próximo passo
  (`:597`); não afirma ter uma. Fica a observação: os dados de hoje são um contraste, não uma curva.
- **"Fatorial completo, ~4.600 células"** (seção 1): erro de conta. 2·4·3·3·2·2·2·2·2 = **2.304** (6.912 com
  1, 20 e 40 sessões). A conclusão (inviável, e muitas células nem existem) não muda.
- **`ckpt_interval` "só afeta o Ridge 1024 de 1 sessão"** (seção 3): errada na outra direção. Os
  experimentos ainda não rodados de MindEye2 (`subj01_me2_pessoa_animal`, `subj01_me2_1000img_3rep_*`,
  `src/experimentos.py:106-134`, de 20 a 60 épocas com 40 sessões) não definem `ckpt_interval` e herdam o
  999 do shell: uma queda perde a corrida inteira. O pragmático estava certo para eles.
- **"`BENCHMARK.md` editado à mão; o gerador desfaria a edição"** (seções 2.5 e 3): sem base. O commit
  `bc85145` mudou o `BENCHMARK.md` junto com o gerador (`src/report/data.py`, `page.py`, `config.py`), na
  mesma direção. O que está defasado é o cabeçalho ("commit `1aec3b2`"), porque o `make_benchmark.py` não
  roda sem os tensores de `results/evals/` e `~/mindeyev2/evals/all_captions.pt` (conferido: falha nesta
  máquina sem escrever nada).
- **"Colisão de nomes à espera de acontecer"** (seção 2.6): risco baixo. O `mindeye_ridge` importa
  `utils` e `models` qualificados (`from mindeye_ridge import utils`); só os scripts do MindEye1 usam o nome
  solto. Com a reorganização, só `tests/test_agregacao.py` ainda põe `mindeye1/src` no `sys.path`.
- **"Retrieval do ridge-only acima do ME1 não é comparável"** (seção 2.3): exagerada. Top-1 entre 300 é a
  mesma tarefa, com o mesmo acaso, para os dois; o que muda é o espaço em que se mede a distância. A
  ressalva certa é "a mesma tarefa em espaços diferentes", não "incomparável".
- **FRR → prior do ME2 com "cosseno centrado 0,12"** (seção 1): é o de 1 sessão; com 40 sessões é 0,22.

### 7.3 Sugestões revistas

| Sugestão original (seção 5) | Revisão |
|---|---|
| 1. Uma agregação no teste para todos | Não forçar uma só: a média das saídas é o protocolo do artigo do MindEye2 e trocá-la tira a comparação com ele. Registrar a agregação como coluna e rodar a outra como variante |
| 1. Tabela unCLIP como primária | Mantida para ME2 × FRR (tira o refino e o GIT). Para o MindEye1 não resolve: o gerador dele é outro (Versatile Diffusion), e a comparação de imagem com ele segue sendo de pipeline inteiro |
| 2. Métricas por imagem + bootstrap pareado | Mantida, e é a de maior retorno |
| 3. Controles de teto, piso e legenda | Mantida; o custo está mais perto de 6–8 GPU-h (o teto com refino é uma reconstrução completa, ~3 h) |
| 4. Proveniência | Mantida. Acrescentar: o resumo do `BENCHMARK.md` ("não foi medido") desatualizou, e o `make_benchmark.py` deveria calcular a régua de semente das tabelas que já existem |
| 5. 3 sementes nos contrastes de 1 sessão | **Mais barata do que dito.** O 1024 + prior já tem 3 sementes com métricas de imagem. Faltam só as do 1024 sem prior (2 × ~4 h com a reconstrução) e as do 4096 + blurry (2 × ~6 h), se quiser a régua de cada configuração |
| Fatorial 2³ com o blurry como cabeça nova sobre o 1024 | Trocar por **4096 com e sem a perda blurry, a partir do mesmo checkpoint**. Separa o blurry da dimensão sem treinar uma peça nova (que seria treinada só com os dados do subj01, enquanto a do 4096 veio do pré-treino). Hoje o script escolhe o checkpoint pelo `BLURRY` (`run_ridgeonly_prior.sh:21-36`), então isso pede uma mudança pequena: aceitar o 4096 com `--no-blurry_recon` |
| 7. Site fora do git | **Retirada.** O site fica no git. Se o diff a cada etapa incomodar, a alternativa é regenerar só sob demanda, não tirar o site |
| 7. `ckpt_interval` | **Sobe de prioridade**: definir `ckpt_interval` (1 ou 5) como padrão do mindeye2 no `treina.py` antes de rodar os experimentos de 40 sessões |
| 7. `NOVO_COMPUTADOR.md`, `pyproject.toml`, aposentar `run_benchmark.sh`, isolar o `sys.path` | **Feito** na reorganização (commits `2b09312`, `308f186`): branch, tempos e `frr_ms` corrigidos; `pip install -e .`; o `scripts/README.md` marca o `run_benchmark.sh` como o lote original |
| 7. Registro único de modelos, lockfile + sha256 | Seguem pendentes |

### 7.4 Prioridades, depois da conferência

1. **`ckpt_interval` padrão no mindeye2** (uma linha no `treina.py`). Protege os experimentos que ainda vão rodar.
2. **Métricas por imagem** em formato longo, com metadados, e a régua de semente calculada das tabelas que existem. Corrigir o "não foi medido" do `BENCHMARK.md`.
3. **Mistura 75/25 e agregação como parâmetros explícitos**, gravados no CSV. Tabela unCLIP como primária entre ME2 e FRR.
4. **Controles de teto, piso e legenda**, ~6–8 GPU-h, sem treino.
5. **Rótulo dos dois avaliadores** e o número do README.
6. **4096 com e sem a perda blurry** a partir do mesmo checkpoint, no lugar do fatorial 2³.
7. Restringir "o pré-treino é o que mais vale" (`apresentacao.html:374`), que é a narrativa realmente confundida.
