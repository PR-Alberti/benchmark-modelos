"""Gera final_evaluations.py a partir do notebook.

Mesmo procedimento dos outros geradores: converte os code cells para .py e
aplica os patches minimos para rodar por linha de comando sobre modelos
treinados com --no-blurry_recon.
"""
import json, re, sys

nb = json.load(open('final_evaluations.ipynb'))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

# Cells 32-35 sao visualizacoes UMAP que carregam all_backbones.pt e
# all_prior_out.pt -- arquivos que o recon_inference.py nao gera. Nao produzem
# nenhuma metrica da tabela, entao ficam de fora.
DROP = {32, 33, 34, 35}

parts = []
for i, c in enumerate(cells):
    if i in DROP:
        continue
    lines = [l for l in c['source'] if not l.lstrip().startswith(('%', '!'))]
    src = ''.join(lines)
    if src.strip():
        parts.append(src)
code = '\n\n'.join(parts)


def patch(old, new):
    global code
    if code.count(old) != 1:
        sys.exit(f"padrao encontrado {code.count(old)}x (esperado 1):\n{old[:200]}")
    code = code.replace(old, new, 1)


# --- dados baixados vem de $DATA/evals, nao de src/evals --------------------
# all_images.pt, all_captions.pt e all_git_generated_captions.pt sao baixados
# pelo download_data.py para $DATA/evals/. O caminho relativo do notebook so
# funcionava com uma copia dentro de src/evals/, que some a cada limpeza.
for _nome in ("all_images", "all_captions", "all_git_generated_captions"):
    patch(f'torch.load(f"evals/{_nome}.pt")',
          'torch.load(f"{data_path}/evals/%s.pt")' % _nome)

# --- backend nao-interativo ------------------------------------------------
patch("import matplotlib.pyplot as plt",
      "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt")

# --- o sufixo deve refletir o arquivo realmente avaliado --------------------
# O original fixa "_all_enhancedrecons", o que rotula errado uma avaliacao
# feita sobre as recons base. O sufixo tambem controla o blend low-level
# logo abaixo ("if 'enhanced' in model_name_plus_suffix").
patch('model_name_plus_suffix = f"{model_name}_all_enhancedrecons"',
      'model_name_plus_suffix = os.path.splitext(os.path.basename(all_recons_path))[0]')

# --- blurryrecons e opcional (modelos --no-blurry_recon nao geram) ----------
patch('all_blurryrecons = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons.pt")',
      '_blurry_path = f"evals/{model_name}/{model_name}_all_blurryrecons.pt"\n'
      'all_blurryrecons = torch.load(_blurry_path) if os.path.exists(_blurry_path) else None\n'
      'if all_blurryrecons is None:\n'
      '    print("sem all_blurryrecons.pt: modelo treinado com --no-blurry_recon, "\n'
      '          "o blend low-level de 25% nao sera aplicado")')

patch('''if all_blurryrecons.shape[-1] != imsize:
    all_blurryrecons = transforms.Resize((imsize,imsize))(all_blurryrecons).float()
    
if "enhanced" in model_name_plus_suffix:
    print("weighted averaging to improve low-level evals")
    all_recons = all_recons*.75 + all_blurryrecons*.25''',
      '''if all_blurryrecons is not None and all_blurryrecons.shape[-1] != imsize:
    all_blurryrecons = transforms.Resize((imsize,imsize))(all_blurryrecons).float()

if "enhanced" in model_name_plus_suffix and all_blurryrecons is not None:
    print("weighted averaging to improve low-level evals")
    all_recons = all_recons*.75 + all_blurryrecons*.25''')

# --- salvar a tabela completa, nao so a coluna de valores ------------------
patch('''df["Value"].to_csv(f'tables/{model_name_plus_suffix}.csv', sep='\\t', index=False)''',
      '''df.to_csv(f'tables/{model_name_plus_suffix}.csv', sep='\\t', index=False)''')

# --- two_way_identification em lotes ---------------------------------------
# O original empilha as 1000 imagens de uma vez e passa tudo num unico batch
# para a GPU (~6.7 GB numa alocacao so), estourando a VRAM. O resultado e
# identico processando em lotes, pois a rede e aplicada imagem a imagem.
patch('''    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))''',
'''    def _in_batches(items, bs=64):
        outs = []
        for i in range(0, len(items), bs):
            batch = torch.stack([preprocess(x) for x in items[i:i+bs]], dim=0).to(device)
            out = model(batch)
            if feature_layer is None:
                outs.append(out.float().flatten(1).cpu())
            else:
                outs.append({k: v.float().flatten(1).cpu() for k, v in out.items()})
            del batch, out
        if feature_layer is None:
            return torch.cat(outs, dim=0)
        return {k: torch.cat([o[k] for o in outs], dim=0) for k in outs[0]}

    preds = _in_batches(all_recons)
    reals = _in_batches(all_images)''')

# como _in_batches ja retorna achatado e na CPU, o pos-processamento muda
patch('''    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()''',
'''    if feature_layer is None:
        preds = preds.numpy()
        reals = reals.numpy()
    else:
        preds = preds[feature_layer].numpy()
        reals = reals[feature_layer].numpy()''')

# --- predcaptions opcional -------------------------------------------------
# Permite avaliar reconstrucoes publicadas por terceiros, para as quais nao
# temos as legendas geradas. Sem elas, so as metricas de caption ficam de fora.
patch('all_predcaptions = torch.load(f"evals/{model_name}/{model_name}_all_predcaptions.pt")',
      '''_cap_path = f"evals/{model_name}/{model_name}_all_predcaptions.pt"
if os.path.exists(_cap_path):
    all_predcaptions = torch.load(_cap_path)
else:
    all_predcaptions = np.array([""] * len(all_recons))
    print("sem all_predcaptions.pt: metricas de caption serao puladas", flush=True)''')

# --- liberar a GPU antes do GNet -------------------------------------------
# Ao chegar no GNet o script ja acumulou na GPU o ViT-bigG, AlexNet, Inception,
# EfficientNet, SwAV e os tensores de reconstrucao -- ~18 GB. Nenhum desses e
# usado depois deste ponto, entao sao liberados antes de carregar o GNet.
patch("GNet = GNet8_Encoder(device=device,subject=subj,model_path=f\"{cache_dir}/gnet_multisubject.pt\")",
      '''for _nome in ["clip_img_embedder", "alex_model", "inception_model",
                      "eff_model", "swav_model", "model", "preds", "reals"]:
    if _nome in globals():
        del globals()[_nome]
import gc; gc.collect(); torch.cuda.empty_cache()
_rss_gpu = torch.cuda.memory_allocated() // 2**20
print(f"GPU liberada antes do GNet: {_rss_gpu} MB em uso", flush=True)
GNet = GNet8_Encoder(device=device,subject=subj,model_path=f"{cache_dir}/gnet_multisubject.pt")''')

# --- caminho relativo do brain_region_masks --------------------------------
# O notebook abre "brain_region_masks.hdf5" sem prefixo, assumindo que o cwd e
# o diretorio dos dados. Rodando de src/ o arquivo nao e encontrado e o script
# morre depois de ja ter calculado quase todas as metricas.
patch('with h5py.File("brain_region_masks.hdf5", "r") as file:',
      'with h5py.File(f"{data_path}/brain_region_masks.hdf5", "r") as file:')

# --- liberar a copia em CPU do ViT-bigG apos mover para a GPU ---------------
# Mesmo padrao do recon_inference: o modelo e construido em fp32 na CPU e a
# copia fica retida sem uma coleta explicita (~7 GB).
patch("clip_img_embedder.to(device)",
      "clip_img_embedder.to(device)\nimport gc; gc.collect()")

# --- sem plt.show() --------------------------------------------------------
code = re.sub(r'^plt\.show\(\)$', '', code, flags=re.M)

open('final_evaluations.py', 'w').write(code)
print(f"final_evaluations.py gerado ({len(code.splitlines())} linhas, {len(DROP)} cells UMAP descartados)")
