# Procedimento: modelos auditivos no NSD (branch `auditive-on-NSD`)

> **Para o Claude no computador do laboratório.** Este arquivo é a ordem de serviço completa. Execute as
> etapas na ordem, sem pedir confirmação para o que está decidido aqui. Pergunte ao Pedro só o que estiver
> marcado como **[PERGUNTAR]** ou o que contradisser o que você encontrar. Siga o `CLAUDE.md` do repositório
> (autor `PR-Alberti`, nunca co-autoria do Claude). Os commits desta tarefa vão **nesta branch**, não na `main`.

Origem: tarefa "Testar um modelo de auditivo no NSD" da semana 23–27/11/2026 no Notion.

## 1. Objetivo

Pegar dois decodificadores feitos para fMRI **auditivo** e aplicar a **arquitetura** deles ao NSD (visual, subj01),
com os dados e a avaliação do benchmark, para ver se ideias de lá melhoram a decodificação visual. Primeiro o
modelo A; o B só começa quando o A tiver resultado.

| | Artigo | No original | No NSD |
|---|---|---|---|
| **A** | Tang, LeBel, Jain, Huth. *Semantic reconstruction of continuous language from non-invasive brain recordings*. Nat. Neurosci. 26:858 (2023). [PMC11304553](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304553/), código [HuthLab/semantic-decoding](https://github.com/HuthLab/semantic-decoding) | fMRI de histórias faladas → texto contínuo | fMRI de uma imagem → **legenda** (texto) |
| **B** | Liu, Du, Chen, He. *Reverse the auditory processing pathway: coarse-to-fine audio reconstruction from fMRI* (2024). [arXiv 2405.18726](https://arxiv.org/abs/2405.18726); sem código publicado | fMRI → som (CLAP → AudioMAE → AudioLDM2) | fMRI → **imagem**, com o mesmo esquema grosso-para-fino |

## 2. Decisões já tomadas (não reabrir)

- A saída é **visual/textual**, não áudio: usamos a arquitetura dos artigos com alvos do NSD.
- Sujeito: subj01. Sessões: **1, 5, 10, 20 e 40** (as N primeiras, como `src/experimentos.py`), começando por
  1 e 40. Teste: as 1.000 imagens do teste novo, 3 repetições promediadas (o mesmo dos outros modelos).
- Os dados de entrada são os do benchmark (`betas_all_subj01_fp32_renorm.hdf5`, 15.724 voxels do
  `nsdgeneral`), lidos por `src/mindeye_ridge/nsd_data.py`. Nada de dataset novo.
- **Vazamento**: as legendas e imagens das 1.000 de teste não podem entrar em nada que seja treinado (nem no
  ajuste do modelo de linguagem do A). Escreva uma checagem que pare o script se entrarem.
- Cada modelo vira um modelo do `src/treina.py` (`"modelo": "tang"` e `"modelo": "coarse2fine"`) com entradas em
  `src/experimentos.py` (`subj01_tang_<N>sess`, `subj01_c2f_<N>sess`), para aparecer no site como os outros.

## 3. Preparar a máquina

[NOVO_COMPUTADOR.md](NOVO_COMPUTADOR.md), **caminho B + complemento de reconstrução** (dados das 40 sessões,
unCLIP, SDXL do refino e redes das métricas; ~85 GB). Não precisa do MindEye1.

```bash
git clone https://github.com/PR-Alberti/benchmark-modelos.git && cd benchmark-modelos
git checkout auditive-on-NSD
setup/setup_env.sh --with-extras
~/envs/fmri/bin/python setup/download_data.py --stage finetune recon enhanced evals --subj 1
~/envs/fmri/bin/python setup/prefetch_weights.py
~/envs/fmri/bin/python -m unittest discover -s tests       # OK antes de seguir
```

Precisa também das **legendas COCO** das imagens do NSD (5 por imagem). O repositório já baixa as anotações do
COCO para a análise semântica (`semantica.tabela_fracoes`); use as `captions_train2017/val2017` do mesmo lugar
e o `nsd_stim_info` para ligar o índice do NSD ao id do COCO. Pacotes novos (instale no `~/envs/fmri` sem
mexer no torch, `--no-deps` quando der, e registre as versões no [SETUP.md](SETUP.md)):
`transformers` (já tem), `nltk`, `jiwer`, `bert-score`, `pycocoevalcap`.

## 4. Modelo A: decodificador semântico (Tang 2023) adaptado

### 4.1 Como o original funciona

Decodificador **gerativo** bayesiano: um modelo de linguagem (GPT-1 ajustado) propõe continuações de palavras
(prior); um **modelo de encoding** (ridge, das features do LM para cada voxel) prevê o BOLD de cada candidato e
a verossimilhança é a de um gaussiano com covariância do ruído estimada por bootstrap; uma **busca em feixe**
(largura 200) mantém as melhores sequências. Features: camada 9 do GPT-1 (768-d), com 4 atrasos FIR; ~10.000
voxels escolhidos pelo desempenho do encoding. Métricas: WER, BLEU-1, METEOR, BERTScore, identificação.

### 4.2 O que muda no NSD

| Peça | Original | Aqui | Por quê |
|---|---|---|---|
| Unidade | palavra no tempo, com modelo de taxa de palavras | **uma legenda por imagem** | o beta do NSD é um por ensaio, sem série temporal |
| Atrasos FIR | 4 | **nenhum** | o GLMsingle já tirou a resposta hemodinâmica |
| Features do encoding | GPT-1, camada 9, por palavra | GPT-1, camada 9, **média dos tokens da legenda** (768-d) | uma resposta por legenda |
| Alvo do encoding | BOLD de cada TR | beta de cada exibição; as 5 legendas da imagem → **média das 5 features** | as 5 descrevem a mesma imagem |
| LM do prior | GPT-1 ajustado em texto narrativo | GPT-1 **ajustado nas legendas do COCO** (train2017 sem nenhuma imagem do NSD) | o domínio é legenda |
| Voxels | ~10 mil melhores do encoding | os 15.724 do `nsdgeneral`, depois seleção dos melhores por validação (5 dobras por imagem) | mesma ideia |
| Busca | feixe 200, palavra a palavra, sem fim fixo | feixe 200, palavra a palavra até `<eos>` ou 20 palavras; o escore de um prefixo é a verossimilhança das features do prefixo | aproximação para legenda curta |

Use o código do HuthLab como base (copie para `src/tang/`, cite o commit, marque cada mudança com
`# benchmark-modelos`, como foi feito no `mindeye1/`). Mantenha a regularização do ridge e o bootstrap da
covariância do original; a escolha do alpha é por validação cruzada agrupada por imagem (5 dobras), como o FRR.

### 4.3 Avaliação

Nas 1.000 imagens de teste (média das 3 repetições):

1. **Texto**: WER, BLEU-1, METEOR, BERTScore (contra as 5 legendas; reporte o máximo e a média) e CIDEr. Linha de
   base nula: legendas sorteadas do treino (como o "null" do artigo). Teto: concordância entre as legendas
   humanas (uma contra as outras 4).
2. **Identificação**: retrieval top-1 entre 300 (30 sorteios, como o benchmark) usando a verossimilhança do
   encoding de cada legenda verdadeira.
3. **Imagem** (para entrar nas métricas de imagem do site): gere uma imagem a partir da legenda decodificada com o
   SDXL que o repositório já baixa (texto → imagem, 1 amostra, seed 42) e passe pelo `final_evaluations.py`.
   Isso deixa o A comparável aos modelos visuais, mas fica marcado no site como "via texto".
4. Compare com as legendas previstas que o MindEye2 já gera (GIT): mesmas métricas de texto.

**Critério de pronto do A**: 1 e 40 sessões treinados e avaliados; BERTScore acima do nulo com intervalo de 95%
(bootstrap nas 1.000 imagens). Se com 40 sessões não passar do nulo, **pare e avise o Pedro** antes de fazer o B.

## 5. Modelo B: grosso-para-fino (Liu 2024) adaptado

### 5.1 Como o original funciona

1. **Grosso**: ridge (PyFastL2LiR) dos voxels para o embedding CLAP do áudio (512-d) → um "token semântico".
2. **Fino**: um decodificador Transformer recebe o fMRI + o token semântico e prevê o latente do **AudioMAE**
   (768-d por patch). Perdas: L2 no latente + L2 perceptual nas camadas do decodificador do AudioMAE + L2 no
   mel reconstruído.
3. **Gerador**: difusão latente iniciada do AudioLDM2, condicionada no latente previsto, + vocoder HiFi-GAN.
   AdamW; lr 1e-6 nos decodificadores e 1e-4 no LDM; uma A100 de 80 GB.

### 5.2 O análogo visual

| Original | Aqui |
|---|---|
| CLAP do áudio (512) | **CLIP ViT-L/14 da imagem, vetor global (768)**, por ridge — use o FRR do repositório (`src/mindeye_ridge/frr.py`) em vez do PyFastL2LiR, que é o mesmo modelo linear com escolha de regularização melhor |
| AudioMAE (768 por patch) | **MAE ViT-L/16 da imagem** (`facebook/vit-mae-large`, 196 × 1.024), alvo do decodificador fino |
| Transformer: fMRI + token → latente | igual: os voxels entram por uma linear para tokens (16 × 1.024), concatenados com o token CLIP; 4 camadas de decodificador cruzando com 196 queries aprendidas |
| perdas: latente, perceptual (decoder do AudioMAE), mel | L2 no latente MAE + L2 perceptual nas camadas do decoder do MAE + L2 nos pixels reconstruídos pelo decoder do MAE |
| AudioLDM2 + HiFi-GAN | **unCLIP do repositório** condicionado no CLIP previsto, partindo (img2img, força 0,85) da imagem que o decoder do MAE reconstrói do latente previsto; depois o refino SDXL como o FRR (`--reconstrucao refino`) |
| ajuste do LDM com lr 1e-4 | **não**: o gerador fica congelado (20 GB de GPU não comportam). Registre como desvio |

Hiperparâmetros: AdamW, lr 1e-6 como o artigo; se em 10 épocas a perda de validação quase não se mexer, use
1e-4 e registre. Validação: 10% das imagens de treino (por imagem). Épocas até a validação parar de cair, máx. 100.
Batch o maior que couber em 20 GB.

### 5.3 Avaliação

O mesmo do FRR: retrieval do embedding CLIP (30 × 300), reconstruções unCLIP e refinadas, `final_evaluations.py`
completo (PixCorr, SSIM, AlexNet(2/5), Inception, CLIP, EffNet-B, SwAV, correlação cerebral) e as métricas
semânticas do repositório. Ablações obrigatórias, cada uma um modelo no `experimentos.py`:

- `subj01_c2f_<N>sess_sogrosso`: só o CLIP previsto no unCLIP, sem o latente MAE (isola o ganho do fino);
- `subj01_c2f_<N>sess_semtoken`: o Transformer sem o token semântico (isola o condicionamento do artigo).

## 6. Entregáveis

1. `src/tang/` e `src/coarse2fine/` (ou nomes equivalentes), com testes em `tests/` que rodem sem GPU.
2. Modelos A e B, 1/5/10/20/40 sessões, no site, ao lado do FRR, MindEye1 e MindEye2.
3. Uma página `docs/AUDITIVO.md` no estilo do [FRR_MULTISUJEITO.md](FRR_MULTISUJEITO.md): o que é cada modelo,
   como foi adaptado, todos os desvios, tabela de resultados e 3–5 frases de leitura.
4. Commits nesta branch, um por etapa, com push. **Não** abra PR para a `main` sem o Pedro pedir.

## 7. Quando parar e perguntar

- a checagem de vazamento falhar e não for óbvio o porquê;
- o A não passar do nulo com 40 sessões (seção 4.3);
- o B não couber em 20 GB nem com batch 1 e checkpoint de gradiente;
- o código do HuthLab exigir dados ou pesos que não estão públicos.

## 8. Registro da execução

_(preencher: máquina, GPU, datas, commits de código externo usados, desvios do plano)_
