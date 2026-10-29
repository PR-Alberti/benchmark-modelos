# Guia para quem vai mexer no código

Para quem programa bem mas não é da área de neurociência ou de modelos generativos. Em ordem: o problema
em uma página, as palavras que aparecem no código, o caminho de um dado do início ao fim, onde fica cada
coisa, e receitas para as mudanças mais comuns. Os detalhes de cada modelo ficam nos outros documentos
(índice em [docs/README.md](README.md)).

Índice: [1. O problema](#1-o-problema) · [2. Glossário](#2-glossário) · [3. Os modelos](#3-os-modelos) ·
[4. O pipeline](#4-o-pipeline) · [5. Onde fica cada coisa](#5-onde-fica-cada-coisa) ·
[6. Receitas](#6-receitas) · [7. Armadilhas](#7-armadilhas)

---

## 1. O problema

Uma pessoa deitada num aparelho de ressonância magnética olha fotos. O aparelho mede a atividade de
milhares de pontinhos do cérebro (voxels) enquanto cada foto aparece. A pergunta do projeto: **a partir só
dessa medida, dá para dizer qual foto a pessoa viu, e até desenhá-la de novo?** E, principalmente,
**quantos dados de cérebro são precisos** para isso funcionar (o plano está em [ESTAGIO.md](ESTAGIO.md)).

Como "desenhar a foto" direto do cérebro é difícil, todos os modelos daqui fazem em dois passos:

```
cérebro (15.724 números)  ──[o modelo, treinado aqui]──▶  embedding CLIP  ──[gerador pronto]──▶  imagem
```

1. **O modelo** aprende a converter a atividade do cérebro num *embedding CLIP*: uma lista grande de
   números que descreve o conteúdo de uma imagem ("um cachorro na grama"). É a parte treinada neste
   repositório, e é o que se compara.
2. **Um gerador de imagens pronto** (o mesmo tipo de modelo que gera imagens a partir de texto) transforma
   esse embedding numa imagem. Ninguém treina o gerador aqui: ele vem de terceiros, em `vendor/`.

O modelo é avaliado de dois jeitos:
- **Retrieval** (só o passo 1): dados 300 embeddings previstos e as 300 imagens certas, quantas vezes o
  previsto fica mais perto da sua imagem do que das outras 299. É a métrica principal, barata e sem gerador.
- **Reconstrução** (os dois passos): a imagem gerada é comparada com a vista por métricas de pixel
  (PixCorr, SSIM) e de conteúdo (redes que reconhecem objetos: AlexNet, Inception, CLIP, EfficientNet, SwAV).

Os dados são o **NSD** (Natural Scenes Dataset), público: 8 pessoas, 40 sessões cada, ~750 fotos por
sessão. Tudo aqui usa a pessoa 1 (`subj01`). As 1.000 fotos que todas as pessoas viram são o **teste**,
sempre o mesmo para todos os modelos; o treino são as outras fotos das primeiras N sessões.

## 2. Glossário

| Termo | No código | O que é |
|---|---|---|
| sujeito | `subj01`, `--subj=1` | a pessoa que fez o exame. Tudo aqui é o sujeito 1; os 2–8 só entram no pré-treino |
| sessão | `sessoes`, `NUM_SESSIONS` | ~1 h no aparelho, ~750 exibições. 1 sessão = pouco dado; 40 = tudo |
| voxel | `15724`, `nsdgeneral` | um "pixel 3D" do cérebro. Usamos os 15.724 da região visual (`nsdgeneral`) |
| beta | `betas_all_subj01_fp32_renorm.hdf5` | a resposta de cada voxel a uma exibição: um vetor de 15.724 números por foto vista |
| exibição / repetição | `exibicoes`, `repeticoes` | cada foto aparece até 3 vezes; cada vez é uma exibição com o seu próprio beta |
| agregação | `agregacao` | como as repetições de uma foto viram amostras de treino: separadas, em média, sorteadas... |
| shared1000 / teste | `new_test`, `all_images.pt` | as 1.000 fotos de teste, as mesmas para todos |
| embedding CLIP | `clipvoxels`, `clip_targets` | descrição numérica de uma imagem, feita pela rede CLIP. Dois tamanhos aqui: **ViT-bigG** (256 × 1.664 números, MindEye2 e FRR) e **ViT-L/14** (257 × 768, MindEye1) |
| alvo | `ClipTargets`, `D_ALVO` | o embedding CLIP da foto verdadeira: o que o modelo tenta prever |
| ridge | `RidgeRegression`, `frr.py` | regressão linear com regularização. É a camada que lê os voxels |
| FRR | `frr`, `run_frr.py` | *fractional ridge regression*: uma ridge com a força da regularização escolhida por validação cruzada, uma para cada número do alvo |
| backbone | `BrainNetwork` | a rede neural do MindEye2 depois da ridge (MLP grande), pré-treinada e congelada aqui |
| diffusion prior | `prior`, `BrainDiffusionPrior` | rede que "ajeita" o embedding previsto para parecer um embedding de imagem de verdade |
| blurry / low-level | `blurry_recon`, `lowlevel` | ramo que prevê uma versão borrada da imagem (cores e formas, sem detalhe) |
| unCLIP | `carrega_unclip`, `unclip6.yaml` | o gerador: do embedding bigG para uma imagem (SDXL unCLIP, da Stability AI) |
| refino | `enhanced`, `enhanced_recon_inference.py` | segunda passada de geração (SDXL) guiada por uma legenda, para a imagem ficar mais nítida |
| legenda prevista | `predcaptions`, GIT | texto gerado a partir do embedding previsto (rede GIT), usado no refino |
| Versatile Diffusion | `VD`, `vd_compat.py` | o gerador do MindEye1 (do embedding ViT-L para imagem) |
| fwd / bwd | `FwdRetrieval`, `BwdRetrieval` | os dois sentidos do retrieval: imagem → cérebro e cérebro → imagem |
| hubness | "centrado" | efeito em que poucas imagens "parecem" com tudo; o retrieval *centrado* tira a média antes de comparar |
| checkpoint | `last.pth`, `train_logs/` | os pesos salvos de um modelo treinado |
| pré-treino multi-sujeito | `multisubject_subj01_1024hid_nolow_300ep` | o backbone e o prior do MindEye2 já treinados nos sujeitos 2–8, baixados prontos |

## 3. Os modelos

| Modelo | Em uma frase | Treina aqui | Código |
|---|---|---|---|
| **FRR** | uma regressão linear dos voxels direto para o embedding bigG. Sem rede, segundos a minutos | tudo | `src/etapas/run_frr.py`, `src/mindeye_ridge/frr.py` |
| **FRR multi-sujeito** | o FRR com um "funil" linear compartilhado, pré-ajustado nos sujeitos 2–8 | tudo | `src/etapas/run_frr_ms.py`, `src/mindeye_ridge/multisujeito.py` |
| **MindEye2 ridge-only** | a rede do MindEye2 pré-treinada nos outros sujeitos; aqui só a camada de entrada (ridge) do sujeito 1 treina | só a ridge | `src/etapas/train_ridgeonly.py` |
| **MindEye1** | a versão anterior do MindEye, treinada do zero, com outro CLIP (ViT-L) e outro gerador | tudo | `mindeye1/src/Train_MindEye.py` |
| MindEye2 do artigo | os modelos publicados pelos autores, baixados prontos, como referência | nada | — |

## 4. O pipeline

Do dado bruto ao site, com o arquivo que cada etapa grava (o nome do modelo é `<m>`):

```
 dados (~/mindeyev2: betas .hdf5, imagens .hdf5, tars do webdataset)
   │
   │  src/treina.py <m>            monta o dataset (train_logs/<m>/dataset.json) e chama o script do modelo
   ▼
 TREINO   scripts/run_frr.sh → etapas/run_frr.py          (FRR: já grava o embedding previsto do teste)
          scripts/run_ridgeonly_prior.sh → etapas/train_ridgeonly.py      (MindEye2: train_logs/<m>/last.pth)
          scripts/me1_run.sh train → mindeye1/src/Train_MindEye.py        (MindEye1)
   │
   ▼
 EMBEDDING DO TESTE        results/evals/<m>/<m>_all_clipvoxels.pt    (1.000 × 256 × 1.664, bigG)
   │      + retrieval      results/tables/<m>_frr.json  ou  results/evals/<m>/<m>_retrieval.json
   │
   │  MindEye2: scripts/run_recon.sh → etapas/recon_inference.py   (passa pelo prior e pelo unCLIP)
   │  FRR:      scripts/run_recon_embeddings.sh → etapas/recon_embeddings.py   (direto no unCLIP)
   ▼
 IMAGEM GERADA             results/evals/<m>/<m>_all_recons.pt  (+ _all_predcaptions.pt, _all_blurryrecons.pt)
   │
   │  etapas/enhanced_recon_inference.py     (refino com a legenda)
   ▼
 IMAGEM REFINADA           results/evals/<m>/<m>_all_enhancedrecons.pt
   │
   │  scripts/run_evals.sh <m> base|enhanced → etapas/final_evaluations.py
   ▼
 MÉTRICAS                  results/tables/<m>_all_recons.csv, <m>_all_enhancedrecons.csv   (no git)
   │
   │  src/make_site.py (chamado sozinho ao fim de cada script)  ·  src/make_benchmark.py
   ▼
 SITE E BENCHMARK          results/site/index.html, apresentacao.html  ·  BENCHMARK.md, results/benchmark/
```

O que importa para mexer:

- **As etapas conversam só por arquivos** em `results/evals/<m>/` com nomes fixos (`<m>_all_<coisa>.pt`).
  Qualquer coisa que grave `<m>_all_clipvoxels.pt` com a forma (n, 256, 1664) pode seguir para a
  reconstrução: é assim que o FRR usa o mesmo gerador do MindEye2.
- `results/evals/` e `train_logs/` são grandes e ficam **fora do git**; `results/tables/` (as métricas) e o
  site ficam **no git**.
- Cada etapa é retomável: rodar o mesmo comando de novo pula o que já terminou.

## 5. Onde fica cada coisa

```
benchmark-modelos/
├── README.md  BENCHMARK.md       o que é e os números (BENCHMARK.md é gerado)
├── pyproject.toml                instala src/ no ambiente (pip install -e .)
├── docs/                         documentação; comece pelo docs/README.md
├── setup/                        preparar uma máquina: bootstrap.sh faz tudo
├── scripts/                      os comandos de cada etapa (bash); common.sh tem os caminhos
├── src/
│   ├── treina.py                 ponto de entrada: treina um experimento
│   ├── experimentos.py           a lista de experimentos (um dicionário cada)
│   ├── make_site.py  make_benchmark.py  make_comparison.py    geram o site, o BENCHMARK.md, figuras
│   ├── etapas/                   as etapas do pipeline (treino, reconstrução, métricas)
│   ├── mindeye_ridge/            a biblioteca: dados, FRR, métricas, dataset controlado
│   └── report/                   o código do site e do BENCHMARK.md
├── vendor/                       código de terceiros (geradores): não editar
├── mindeye1/                     o MindEye1, com ajustes marcados "# benchmark-modelos"
├── notebooks/                    treino.ipynb: o uso, com exemplos
├── tests/                        176 testes, sem GPU, ~20 s
└── results/                      tabelas e site (no git); tensores e logs (fora)
```

Cada pasta tem um `README.md` curto com o papel de cada arquivo. O que dá para mexer sem medo e o que
pede cuidado:

| Pode mexer | Com cuidado | Não mexa |
|---|---|---|
| `src/mindeye_ridge/` (tem testes), `src/report/`, `src/experimentos.py`, `docs/`, `notebooks/` | `src/etapas/` e `scripts/`: não têm teste automático e só rodam de verdade com GPU e dados | `vendor/`; `src/mindeye_ridge/models.py`, `utils.py` e `modeling_git.py` vieram do MindEye2 e os checkpoints dependem dos nomes das camadas |

## 6. Receitas

**Rodar os testes** (não precisam de GPU nem dos dados):

```bash
~/envs/fmri/bin/python -m unittest discover -s tests
```

**Ver o que existe e o que um experimento faria**, sem treinar nada:

```bash
~/envs/fmri/bin/python src/treina.py --lista
~/envs/fmri/bin/python src/treina.py <nome> --so_dataset   # quantas imagens, exibições e repetições
~/envs/fmri/bin/python src/treina.py <nome> --dry_run      # o comando que rodaria
```

**Criar um experimento novo.** Um experimento é um dicionário: o modelo, os hiperparâmetros e o recorte
dos dados. Acrescente em `EXPERIMENTOS` (`src/experimentos.py`) ou rode de um notebook:

```python
import treina
treina.roda("subj01_frr_so_animais", {
    "modelo": "frr",                                       # frr | frr_ms | mindeye2 | mindeye1
    "hiper": {},                                           # vazio: os padrões (treina.py, MODELOS)
    "dataset": {"sessoes": 40, "classes": {"animal": ["sup_animal"]}, "minimo": 10},
    "rotulo": "FRR · só animais",                          # o nome que aparece no site
})
```

Os campos possíveis estão no README (seção "Combinações possíveis"). O FRR leva de segundos a ~20 min;
os MindEye, horas. O modelo novo aparece no site sozinho ao terminar.

**Ver o resultado de um modelo:**

```bash
~/envs/fmri/bin/python -c "from mindeye_ridge import resultados; print(resultados.retrieval('<nome>'))"
```

ou abra `results/site/index.html` no navegador.

**Mudar o site.** O texto fixo está em `src/report/conteudo/*.html`, o estilo e os gráficos em
`src/report/assets/`, e a montagem em `src/report/site.py` e `catalogo.py` (que descobre os modelos em
disco). Regere com `~/envs/fmri/bin/python src/make_site.py` e abra o HTML. Os testes de
`tests/test_site.py` cobrem a montagem.

**Mudar uma etapa** (`src/etapas/`). Elas vieram de notebooks e rodam de cima para baixo, com os
argumentos virando variáveis globais. Antes e depois de mexer, rode a etapa com `--help`: isso importa
tudo e pega erro de import ou de caminho sem precisar de GPU:

```bash
cd src && PYTHONPATH=$PWD ~/envs/fmri/bin/python etapas/<etapa>.py --help
```

Mudança de verdade só se confirma rodando na máquina de treino, num nome de modelo novo (nunca num do
benchmark).

**Acrescentar um modelo novo** (o caminho mais curto é imitar o FRR):
1. a etapa em `src/etapas/<modelo>.py`, gravando `results/evals/<m>/<m>_all_clipvoxels.pt` (n, 256, 1664) e o retrieval;
2. o script `scripts/run_<modelo>.sh`, que faz `source common.sh` e chama a etapa;
3. a entrada em `MODELOS` (`src/treina.py`): o script, as variáveis de ambiente que ele lê e os hiperparâmetros
   padrão (`tests/test_treina.py` confere que as duas listas batem);
4. com `reconstrucao="run_recon_embeddings.sh"` nessa entrada, como no FRR, `--reconstrucao unclip|refino`
   já gera imagens e métricas pelo mesmo caminho.

## 7. Armadilhas

- **O nome do modelo é a chave de tudo.** Pastas, arquivos e tabelas saem de `<m>`. O `treina.py` recusa um
  nome que já tem modelo treinado e um nome cuja configuração mudou: para refazer algo, use outro nome.
- **Comportamento escondido no nome do arquivo.** `final_evaluations.py` mistura 25% da imagem borrada na
  refinada quando o nome tem `enhanced` e existe `_all_blurryrecons.pt`; o site lê `repro_<m>` no lugar de
  `<m>` quando a tabela dele existe (`report/config.py`, `fonte()`).
- **Dois espaços CLIP.** O MindEye1 prevê ViT-L, os outros bigG: o retrieval dele não se compara direto com o
  dos outros, nem a saída dele passa pelo unCLIP.
- **"1 sessão" do MindEye2 não é pouco dado.** O backbone e o prior foram pré-treinados em ~280 h dos outros
  sujeitos; só a ridge vê a 1 sessão do sujeito 1.
- **Semente fixa.** Todo experimento usa a semente 42 (`src/experimentos.py`): uma corrida por configuração.
  Diferenças de ~1 ponto entre modelos podem ser ruído (veja [REVISAO.md](REVISAO.md)).
- **Caminhos relativos no MindEye1.** Os scripts de `mindeye1/src/` rodam de dentro dessa pasta e usam
  `../train_logs` e `../../results`: não mova arquivos de lá.
- **O site é regenerado ao fim de cada etapa** e está no git, então a árvore fica com alterações depois de um
  treino. É esperado.
- **Commits sem co-autoria de IA.** Nenhum commit ou PR leva o Claude como co-autor (`Co-Authored-By`) nem
  "Generated with Claude Code"; a regra está no [CLAUDE.md](../CLAUDE.md), que o Claude Code lê sozinho.
- **Hashes de antes de 29/10/2026 mudaram.** O histórico foi reescrito para tirar essa co-autoria: o código de
  cada commit é o mesmo, mas todos os hashes são outros
  ([commits_reescritos_2026-10-29.txt](commits_reescritos_2026-10-29.txt) tem o antigo e o novo). O
  `experimento.json` de cada treino guarda o hash do código: numa máquina com `train_logs/` daquela época, rode
  `python scripts/remapeia_commits.py train_logs` antes de regenerar o site. Um clone antigo se alinha com
  `git fetch && git reset --hard origin/<branch>` (sem alterações locais pendentes).
