# Procedimento: migrar o benchmark para imagética (branch `imagery-migrate`)

> **Para o Claude no computador do laboratório.** Este arquivo é a ordem de serviço completa. Execute as
> etapas na ordem, sem pedir confirmação para o que está decidido aqui. Pergunte ao Pedro só o que estiver
> marcado como **[PERGUNTAR]** ou o que contradisser o que você encontrar na máquina. Siga o `CLAUDE.md` do
> repositório (autor `PR-Alberti`, nunca co-autoria do Claude). Os commits desta tarefa vão **nesta branch**,
> não na `main`.

Origem: tarefa "Migrar pro imagery (nsd imagery e avid)" da semana 23–27/11/2026 no Notion.

## 1. Objetivo

Treinar o **MindEye1** e o **MindEye2** com **1, 5, 10, 20 e 40 sessões** do subj01 do NSD, exatamente nos
padrões do benchmark, e avaliar cada modelo, com todas as métricas, em dois conjuntos de imagética:

1. **NSD-Imagery** (mesmos sujeitos e scanner do NSD): o modelo é aplicado direto.
2. **AVID** (outro scanner, outros sujeitos): o modelo do NSD é a base congelada e só a camada de entrada é
   ajustada por sujeito do AVID.

A pergunta: quanto do desempenho de percepção (teste de 1.000 imagens do NSD) sobrevive na imagética, e como
isso escala com as sessões de treino.

## 2. Decisões já tomadas (não reabrir)

| Item | Decisão |
|---|---|
| Treino | **de novo, nesta branch** (não reaproveitar checkpoints de outra máquina) |
| Sessões | 1, 5, 10, 20, 40 (as N primeiras sessões do subj01, como em `src/experimentos.py`) |
| MindEye1 | a receita do benchmark (`subj01_me1_<N>sess`, com e sem low-level, como o `scripts/run_me1_sessoes.sh`) |
| MindEye2 | **as duas variantes**: com low-level (`..._4096blurry`, o "MindEye2" do site) e sem low-level (`..._prior`) → 10 modelos |
| Total | 5 MindEye1 + 10 MindEye2 = **15 treinos** |
| NSD-Imagery | aplicação direta, sem nenhum ajuste (mesmo sujeito, mesma máscara `nsdgeneral`) |
| AVID | **ajuste por sujeito**: backbone do NSD congelado; treina só uma camada nova voxels AVID → entrada do modelo |
| Métricas | todas as do benchmark (`src/etapas/final_evaluations.py`) + as semânticas/de legenda que o repositório já calcula |

## 3. Preparar a máquina

Siga o [NOVO_COMPUTADOR.md](NOVO_COMPUTADOR.md) pelo **caminho A** (`setup/bootstrap.sh`, ~130 GB; é o único
que traz MindEye1, reconstrução e métricas). Reserve **≥ 600 GB** de disco no total: 200 GB do setup, ~17 GB
por modelo MindEye1 treinado (5 × 17 = 85 GB), os MindEye2, o NSD-Imagery e a parte do AVID (seção 6).

```bash
git clone https://github.com/PR-Alberti/benchmark-modelos.git && cd benchmark-modelos
git checkout imagery-migrate
setup/bootstrap.sh --check && setup/bootstrap.sh
~/envs/fmri/bin/python -m unittest discover -s tests      # tem que dar OK antes de seguir
```

Rode tudo dentro de `tmux`. Registre no fim deste arquivo (seção 9) a máquina, a GPU e a data.

## 4. Etapa 1: os 15 treinos no NSD (percepção)

Antes, confira em `src/experimentos.py` os nomes exatos dos experimentos de 1/5/10/20/40 sessões para as três
famílias (MindEye1, MindEye2 com low-level, MindEye2 sem low-level). Os de 1 e 40 já existem
(`subj01_me1_{1,40}sess`, `subj01_ridgeonly_{1,40}sess_4096blurry`, `subj01_ridgeonly_{1,40}sess_prior`); os
intermediários vêm de `EPOCAS_POR_SESSOES`. Se algum nome faltar, crie a entrada **igual** às vizinhas, só
trocando o número de sessões (e as épocas pela regra de `EPOCAS_POR_SESSOES`).

```bash
# MindEye1, 5/10/20 (com e sem low-level) e depois 1 e 40
scripts/run_me1_sessoes.sh                         # 5, 10, 20
SESSOES="1 40" scripts/run_me1_sessoes.sh          # se o script aceitar 1 e 40; senão: scripts/run_me1_benchmark.sh
# MindEye2, com e sem low-level, 5/10/20 e depois 1 e 40
scripts/run_sessoes_intermediarias.sh              # SESSOES="2 5 10 20" por padrão: use SESSOES="5 10 20"
SESSOES="1 40" scripts/run_sessoes_intermediarias.sh   # se recusar, use scripts/run_reexecutar.sh para as pontas
```

Ordem: do menor para o maior (1 → 40), para ter resultados cedo. Cada script retoma de onde parou.
Tempo estimado numa A4500: MindEye1 ~1h40 (1 sessão) a ~16 h + 13 h de low-level (40); MindEye2 ~2 h a ~8 h.
No total, **~5–6 dias de GPU**. Se houver mais de um PC, divida por família (um PC por família).

**Critério de pronto:** os 15 modelos aparecem no site (`results/site/index.html`) com retrieval e métricas no
teste de 1.000 imagens, e os números de 1 e 40 sessões ficam a ±2 pontos percentuais dos que já estão no
README da `main` (é a checagem de que o treino reproduziu). Se não ficarem, **pare e avise o Pedro**.

## 5. Etapa 2: NSD-Imagery

### 5.1 Dados

NSD-Imagery: Kneeland et al., CVPR 2025, [arXiv 2506.06898](https://arxiv.org/abs/2506.06898). Uma sessão
extra de 7T nos sujeitos do NSD, com condições **visão** e **imagética** (e atenção, que não entra) sobre três
conjuntos de estímulos: **simples** (6 formas geométricas), **complexos** (5 cenas do shared1000 + 1 obra de
arte) e **conceituais** (6 palavras; na visão eles ficam de fora). Betas GLMsingle no mesmo formato do NSD
(`betas_fithrf_GLMdenoise_RR`, 1,8 mm).

1. O acesso é pelo termo do NSD (o Pedro já aceitou). Os arquivos ficam no bucket público
   `s3://natural-scenes-dataset/`. Ache a pasta do imagery com
   `aws s3 ls --no-sign-request s3://natural-scenes-dataset/` (e no manual do NSD). **Não foi verificado** o
   caminho exato: descubra, anote aqui e baixe só o do **subj01** (betas da sessão de imagery, o arquivo de
   design/condições e os estímulos).
2. Confira se o artigo tem código de pré-processamento para o formato do MindEye (procure no GitHub do
   MedARC-AI e do Reese Kneeland por "imagery"). Se existir, use-o e cite o commit aqui.

### 5.2 Montar o teste no formato do benchmark

Escreva `src/mindeye_ridge/nsd_imagery.py` (com teste em `tests/`) que entregue, para o subj01:

- os voxels do `nsdgeneral` na **mesma ordem e normalização** do `betas_all_subj01_fp32_renorm.hdf5` (leia em
  `nsd_data.py`/`setup/` como o "renorm" foi feito e aplique igual, por run/sessão);
- uma tabela de ensaios: estímulo, conjunto (simples/complexo/conceitual), condição (visão/imagética),
  repetição;
- a imagem alvo de cada estímulo em 224×224 (para os conceituais, a imagem de referência do artigo; se não
  houver, eles ficam só nas métricas de legenda/semânticas);
- a **média das repetições** por estímulo e condição (como o teste do NSD faz com as 3 repetições).

Sanidade obrigatória: nos 5 estímulos complexos tirados do shared1000, o embedding previsto pelo modelo de 40
sessões na condição **visão** tem que acertar o retrieval bem acima do acaso. Se não acertar, a normalização
ou a ordem dos voxels está errada: corrija antes de seguir.

### 5.3 Avaliar os 15 modelos

Para cada modelo e cada (conjunto × condição):

1. embedding previsto (alto nível) e, quando houver, latente do low-level;
2. reconstrução com o mesmo pipeline do modelo (MindEye2: `recon_inference` + refinamento; MindEye1:
   Versatile Diffusion com/sem img2img), 1 amostra por estímulo (o benchmark usa 1);
3. métricas: PixCorr, SSIM, AlexNet(2), AlexNet(5), Inception, CLIP, EffNet-B, SwAV, correlação cerebral
   (`final_evaluations.py`) e as de legenda/semânticas que o repositório já tem para o modelo;
4. **identificação 2-a-2** dentro de cada conjunto (o protocolo do artigo, já que são só 6 estímulos por
   conjunto: o retrieval entre 300 do benchmark não se aplica); retrieval top-1 entre os 18 como extra.

Saídas: `results/imagery/nsd_imagery/<modelo>/` (reconstruções, embeddings) e
`results/tables/imagery_nsd_<modelo>.csv`, uma linha por (conjunto, condição), mais uma tabela resumo
`results/tables/imagery_nsd_resumo.csv` com as curvas por sessões.

## 6. Etapa 3: AVID

### 6.1 Dados

AVID: *Auditory-induced Visual Mental Imagery Dataset* — Shen, Li, Zhang, Li (2026), ScienceDB,
DOI [10.57760/sciencedb.33666](https://doi.org/10.57760/sciencedb.33666), CC BY 4.0, página
<https://www.scidb.cn/en/detail?dataSetId=4f116696b67f4158ad42ebb8de54c406>.

- 5T (uMR Jupiter), 2 mm isotrópico, TR 1,5 s. 9 participantes; **sub-09 não tem dados** → 8 sujeitos.
- Tarefa: o participante **ouve** uma descrição falada e **imagina** a cena; 12 s por ensaio. Não há condição
  de percepção visual.
- **Cenas complexas**: 800 estímulos × 2 repetições = 1.600 ensaios (runs 1–64), todos os 8 sujeitos.
- **Formas simples**: 400 estímulos × 2 = 800 ensaios (runs 65–96), só 7 sujeitos (sub-02 e sub-03 não fizeram).
- Mesmos estímulos para todos; ordem diferente por sujeito. `source_data/used_stimuli.tsv` tem índice,
  legendas (chinês e inglês), áudio e **imagem correspondente** de cada estímulo.

**Baixe só o necessário** (o total é 775 GB):

| Pasta | Para quê |
|---|---|
| `derivatives/pp_betas_vol/sub-0{1..8}/` | betas GLMsingle em volume (NIfTI 5D: `[raw, zscore] × (run·ensaio) × X × Y × Z`); use o `zscore` |
| `source_data/` | estímulos (imagens + áudio), `used_stimuli.tsv`, ordem e onsets por sujeito (`experiment_scripts/`) |
| `derivatives/freesurfer_results/sub-0{1..8}/` | para montar a máscara de córtex visual |

Se o download do ScienceDB pedir login, **[PERGUNTAR]**: o Pedro faz o login (nunca digite senha). Prefira o
download por pasta/arquivo, não o pacote inteiro.

### 6.2 Máscara e dados por sujeito

Não há `nsdgeneral` no AVID. Monte uma máscara equivalente por sujeito a partir do `aparc+aseg` do FreeSurfer
(lateraloccipital, lingual, cuneus, pericalcarine, fusiform, inferiortemporal, parahippocampal, superiorparietal,
inferiorparietal; as duas hemisférios), reamostrada para o espaço funcional. Anote quantos voxels ficaram por
sujeito. Depois: ordene os ensaios pelo `experiment_scripts/`, associe cada ensaio ao índice do estímulo e média
das 2 repetições só no teste.

### 6.3 Divisão treino/teste

Mesma para todos os sujeitos (os estímulos são os mesmos): com `seed=42`, sorteie **100 das 800 cenas** para
teste; as outras 700 (1.400 ensaios, repetições separadas) são o treino do ajuste. Nas formas simples, 50 das
400 para teste. Grave a lista em `results/imagery/avid/split.json` e não mude depois.

### 6.4 O ajuste por sujeito

Para cada um dos 15 modelos do NSD e cada sujeito do AVID:

- **MindEye2** (as duas variantes): troque a ridge do subj01 por uma ridge nova (voxels AVID → `hidden_dim` do
  modelo, 1.024 ou 4.096); **congele todo o resto** (backbone, prior, low-level) e treine só a ridge com a
  mesma perda do treino do modelo, com as imagens correspondentes do AVID como alvo.
- **MindEye1**: o equivalente é a primeira camada linear do MLP (voxels → 4.096) do alto nível e a primeira do
  `Voxel2StableDiffusionModel` do low-level; troque e treine só essas, resto congelado.
- Hiperparâmetros: os da fase de ajuste do modelo original; épocas até a perda de validação (10% do treino,
  por estímulo) parar de cair, no máximo 150. Registre a curva.

São 15 modelos × 8 sujeitos = 120 ajustes por tarefa; cada um é pequeno (só uma camada). Faça primeiro as
**cenas complexas**; as formas simples vêm depois, se sobrar tempo.

### 6.5 Avaliação

Mesmas métricas da seção 5.3 nas 100 cenas de teste, com a imagem correspondente como alvo. Como são 100
cenas, o retrieval é top-1 entre as 100 (e 2-a-2). Também: como as legendas em inglês existem, calcule as
métricas de legenda/semânticas contra elas. Saídas em `results/imagery/avid/<modelo>/sub-0X/` e
`results/tables/imagery_avid_resumo.csv` (média e desvio entre sujeitos, por sessões).

## 7. Entregáveis

1. Os 15 modelos treinados e avaliados no teste do NSD (no site).
2. `results/tables/imagery_nsd_resumo.csv` e `results/tables/imagery_avid_resumo.csv`.
3. Uma aba ou seção "Imagética" no site (`src/report/`), com: as curvas por sessões (percepção NSD × visão
   NSD-Imagery × imagética NSD-Imagery × imagética AVID), por métrica, e uma galeria pequena de reconstruções.
4. Uma seção "Resultados" no fim deste arquivo com a tabela principal e 3–5 frases do que se vê.
5. Commits nesta branch, um por etapa concluída, com push. **Não** abra PR para a `main` sem o Pedro pedir.

## 8. Quando parar e perguntar

- os números de 1/40 sessões não reproduzem (seção 4);
- o retrieval de sanidade do NSD-Imagery fica no acaso (seção 5.2) e você não acha a causa em ~2 h;
- o download do AVID ou do NSD-Imagery pede login, ou o caminho no S3 não existe;
- o disco vai passar de 90%.

## 9. Registro da execução

_(preencher: máquina, GPU, datas, caminhos dos dados, desvios do plano)_
