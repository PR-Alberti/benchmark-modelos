"""Gera Train_ridgeonly.py a partir de Train.ipynb.

Converte os code cells para .py (equivalente ao `jupyter nbconvert --to python`
usado em accel.slurm) e aplica os patches minimos para o experimento
"fine-tune apenas da ridge a partir do checkpoint multi-sujeito".
"""
import json, re, sys

nb = json.load(open('Train.ipynb'))
parts = []
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    lines = [l for l in c['source'] if not l.lstrip().startswith(('%', '!'))]
    src = ''.join(lines)
    if src.strip():
        parts.append(src)
code = '\n\n'.join(parts)


def patch(old, new, count=1):
    global code
    assert code.count(old) >= count, f"padrao nao encontrado:\n{old[:200]}"
    code = code.replace(old, new, count)


# --- 1) backend nao-interativo (sem display no servidor) -------------------
patch("import matplotlib.pyplot as plt",
      "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt")

# --- 2) novos argumentos ---------------------------------------------------
patch('''parser.add_argument(
    "--max_lr",type=float,default=3e-4,
)''',
      '''parser.add_argument(
    "--max_lr",type=float,default=3e-4,
)
parser.add_argument(
    "--ridge_only",action=argparse.BooleanOptionalAction,default=False,
    help="congela backbone (e prior) e treina SOMENTE a camada ridge do sujeito",
)
parser.add_argument(
    "--metrics_csv",type=str,default=None,
    help="caminho para gravar as metricas por epoca em CSV",
)
parser.add_argument(
    "--embedder_fp16",action=argparse.BooleanOptionalAction,default=False,
    help="guarda os pesos do ViT-bigG em fp16 (~3.6 GB a menos de VRAM)",
)''')

# --- 3) congela tudo menos a ridge e monta o optimizer de acordo ------------
patch('''opt_grouped_parameters = [
    {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
if use_prior:
    opt_grouped_parameters.extend([
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ])''',
      '''if ridge_only:
    # congela backbone (e prior, se houver): so a ridge do sujeito e treinada
    model.backbone.requires_grad_(False)
    if use_prior:
        model.diffusion_prior.requires_grad_(False)
    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
    ]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"RIDGE-ONLY: {n_train:,} params treinaveis de {n_total:,} ({100*n_train/n_total:.2f}%)")
else:
    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    if use_prior:
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ])''')

# --- 4) o ckpt multi-sujeito e carregado ANTES do freeze/optimizer ----------
#     (no notebook ele e carregado depois; a ordem nao muda o resultado, mas
#      manter o carregamento antes do accelerator.prepare e o que importa)

# --- 4b) embedder CLIP em fp16 (opcional) ----------------------------------
# O ViT-bigG tem ~1.8B params; em fp32 ocupa ~7.2 GB so de pesos. O autocast
# do loop ja faz as matmuls em fp16, entao guardar os pesos em fp16 libera
# ~3.6 GB de VRAM sem mudar de forma relevante os alvos de CLIP.
patch("""clip_img_embedder.to(device)""",
      """clip_img_embedder.to(device)
if embedder_fp16:
    clip_img_embedder = clip_img_embedder.half()
    print("clip_img_embedder convertido para fp16 (economia ~3.6 GB de VRAM)")""")

# --- 5) grava as metricas por epoca em CSV ---------------------------------
patch("            progress_bar.set_postfix(**logs)",
      '''            if metrics_csv is not None:
                import csv as _csv
                _new = not os.path.exists(metrics_csv)
                with open(metrics_csv, 'a', newline='') as _f:
                    _w = _csv.DictWriter(_f, fieldnames=['epoch'] + list(logs.keys()))
                    if _new: _w.writeheader()
                    _w.writerow({'epoch': epoch, **logs})
            print(f"epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in logs.items()
                                                if isinstance(v, (int, float))))

            progress_bar.set_postfix(**logs)''')

# --- 6) sem plt.show() no final -------------------------------------------
code = re.sub(r'^plt\.(plot|show)\(.*\)$', '', code, flags=re.M)

with open('Train_ridgeonly.py', 'w') as f:
    f.write(code)
print("Train_ridgeonly.py gerado com", len(code.splitlines()), "linhas")
