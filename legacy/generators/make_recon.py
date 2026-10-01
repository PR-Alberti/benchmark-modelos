"""Gera recon_inference.py e enhanced_recon_inference.py a partir dos notebooks.

Mesmo procedimento do make_ridgeonly.py: converte os code cells para .py e
aplica os patches minimos para rodar por linha de comando.
"""
import json, re, sys


def convert(nb_path, out_path, patches):
    nb = json.load(open(nb_path))
    parts = []
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        lines = [l for l in c['source'] if not l.lstrip().startswith(('%', '!'))]
        src = ''.join(lines)
        if src.strip():
            parts.append(src)
    code = '\n\n'.join(parts)

    for old, new in patches:
        if code.count(old) < 1:
            sys.exit(f"[{nb_path}] padrao nao encontrado:\n{old[:200]}")
        code = code.replace(old, new, 1)

    # backend nao-interativo
    if 'import matplotlib.pyplot as plt' in code:
        code = code.replace("import matplotlib.pyplot as plt",
                            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt", 1)
    code = re.sub(r'^plt\.show\(\)$', '', code, flags=re.M)

    open(out_path, 'w').write(code)
    print(f"{out_path} gerado ({len(code.splitlines())} linhas)")


# --- recon_inference -------------------------------------------------------
# O notebook assume blurry_recon=True e use_prior=True implicitamente em alguns
# pontos; os argumentos ja existem no argparser, entao basta rodar por CLI.
RSS_HELPER = """import os as _os
def _rss(tag):
    with open(f"/proc/{_os.getpid()}/status") as _f:
        for _l in _f:
            if _l.startswith("VmRSS"):
                import torch as _t
                _v = _t.cuda.memory_allocated()//2**20 if _t.cuda.is_available() else 0
                print(f"[RSS] {int(_l.split()[1])//1024:6d} MB ram | {_v:5d} MB vram  {tag}", flush=True)
                return
"""

convert('recon_inference.ipynb', 'recon_inference.py', patches=[
    # vstack a cada imagem recopia o tensor inteiro (O(n^2)) e guarda tudo em
    # 768x768, dando ~14 GB de pico de RAM em 1000 recons -- o processo morria
    # por volta de 83%. Como o script salva em 256x256 de qualquer forma, o
    # redimensionamento vai para dentro do laco e a acumulacao vira uma lista
    # com um unico cat: pico de ~1,5 GB. Mesmo remedio ja usado no
    # enhanced_recon_inference.py. Redimensionar por imagem ou no lote inteiro
    # da o mesmo resultado: a interpolacao e por imagem nos eixos espaciais.
    ("""            if all_recons is None:
                all_recons = samples.cpu()
            else:
                all_recons = torch.vstack((all_recons, samples.cpu()))""",
     "            _recon_chunks.append(\n"
     "                transforms.Resize((256, 256))(samples).float().cpu())"),
    ("all_recons = None", "all_recons = None\n_recon_chunks = []"),
    ("""# resize outputs before saving
imsize = 256
all_recons = transforms.Resize((imsize,imsize))(all_recons).float()""",
     "all_recons = torch.cat(_recon_chunks, dim=0)\ndel _recon_chunks\n"
     "import gc; gc.collect()\n_rss('depois de juntar as recons')\n\n"
     "# resize outputs before saving\n"
     "imsize = 256  # all_recons ja saiu do laco em 256x256 float32"),
    # O notebook constroi o BrainNetwork sem repassar blurry_recon nem n_blocks,
    # caindo nos defaults da classe (blurry_recon=True). Isso quebra o
    # carregamento de qualquer modelo treinado com --no-blurry_recon, mesmo
    # passando a flag na linha de comando. O Train.ipynb repassa os dois.
    ("""model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, 
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim) """,
     """model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim,
                          blurry_recon=blurry_recon) """),
    # instrumentacao de memoria: mede RSS nos pontos de carga pesados
    ("import utils", "import utils\n" + RSS_HELPER),
    # O clip_img_embedder e construido mas NUNCA usado no recon_inference (nem
    # aqui nem no notebook original da MedARC): so aparece na construcao. Em fp16
    # ele ocupava 3,6 GB de VRAM parado, o que faz o decode do VAE estourar numa
    # placa de 20 GB quando o desktop tambem roda nela. Fica na CPU.
    # RECON_KEEP_EMBEDDER=1 restaura o comportamento antigo, se algum dia
    # o script passar a usa-lo.
    ("clip_img_embedder.to(device)",
     "_rss('antes de decidir o que fazer com o CLIP bigG')\n"
     "if os.environ.get('RECON_KEEP_EMBEDDER'):\n"
     "    clip_img_embedder = clip_img_embedder.half()\n"
     "    clip_img_embedder.to(device)  # 3,6 GB de VRAM\n"
     "else:\n"
     "    # nunca usado neste script (nem no notebook original): descartar libera\n"
     "    # 3,6 GB de VRAM e ~3,5 GB de RAM. Manter so na CPU nao serve: troca o\n"
     "    # estouro da placa pelo estouro da memoria no meio do laco.\n"
     "    clip_img_embedder = None\n"
     "    print('CLIP bigG descartado (nao e usado neste script)')\n"
     "import gc; gc.collect()\n_rss('depois do CLIP bigG + gc')"),
    ("print(\"ckpt loaded!\")", "print(\"ckpt loaded!\")\n_rss('depois do nosso ckpt')"),
    ("ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'",
     "_rss('antes do unCLIP')\nckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'"),
    # Cada modelo grande e instanciado em fp32 na CPU e so depois movido para a
    # GPU. Sem um gc.collect() explicito a copia da CPU fica retida (o ViT-bigG
    # sozinho segura 7 GB), e o pico soma ate estourar os 30 GB de RAM.
    ("clip_text_model.eval().requires_grad_(False)",
     "clip_text_model.eval().requires_grad_(False)\nimport gc; gc.collect()\n_rss('depois do GIT')"),
    ("clip_convert.to(device)",
     "clip_convert.to(device)\nimport gc; gc.collect()"),
    # O conditioner do unclip6 instancia um SEGUNDO ViT-bigG-14 (~10 GB fp32),
    # duplicando o embedder que ja temos carregado. Como o checkpoint do unCLIP
    # sobrescreve esses pesos, nao ha motivo para ler o pre-treinado do disco.
    # Somado a isso, construir o engine em fp16 corta a alocacao pela metade.
    ("""diffusion_engine = DiffusionEngine(network_config=network_config,""",
     """conditioner_config['params']['emb_models'][0]['params']['version'] = None
_prev_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float16)
diffusion_engine = DiffusionEngine(network_config=network_config,"""),
    ("diffusion_engine.eval().requires_grad_(False)\ndiffusion_engine.to(device)",
     "torch.set_default_dtype(_prev_dtype)\n"
     "diffusion_engine.eval().requires_grad_(False)\ndiffusion_engine.to(device)\n"
     "# o config traz disable_first_stage_autocast=True, entao o VAE roda fora do\n"
     "# autocast e recebe latentes fp32; alem disso decode em fp16 no SDXL gera\n"
     "# artefatos. Mantem o first stage em fp32 -- custa ~0.3 GB.\n"
     "diffusion_engine.first_stage_model.float()\n"
     "import gc; gc.collect()\n_rss('depois de construir o unCLIP fp16 na GPU')"),
    # No notebook, "plotting" so e definido quando is_interactive() e True.
    # Rodando como script a variavel nunca existe e o loop quebra com NameError
    # na primeira imagem. (Com plotting=True o proprio codigo aborta de
    # proposito num "err" mais adiante, entao False e o valor correto aqui.)
    ("if utils.is_interactive(): plotting=True",
     "plotting = utils.is_interactive()"),
    # O checkpoint do unCLIP tem 18 GB. Sem liberar explicitamente, ele fica
    # residente na RAM durante todo o loop de geracao -- em maquinas com 30 GB
    # isso leva o processo a ser morto pelo OOM killer (sem traceback).
    ("""ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])""",
     """# mmap=True mapeia o arquivo em disco em vez de materializar os 18 GB na
# RAM. Sem isso o pico passa de 30 GB (o resto do pipeline ja ocupa ~12 GB) e
# o processo morre por SIGKILL do OOM killer, sem traceback.
try:
    ckpt = torch.load(ckpt_path, map_location='cpu', mmap=True)
except (RuntimeError, ValueError, TypeError):  # ckpt em formato antigo
    ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])
del ckpt
import gc; gc.collect()"""),
])

# --- enhanced_recon_inference ---------------------------------------------
# Este tem dois caminhos hardcoded para o cluster da Stability.
ENH_ARGS = '''parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
)
parser.add_argument(
    "--cache_dir", type=str, default=os.getcwd(),
)'''

convert('enhanced_recon_inference.ipynb', 'enhanced_recon_inference.py', patches=[
    # all_images.pt e dado baixado (fica em $DATA/evals/), nao saida do modelo.
    # O caminho relativo so funcionava quando havia uma copia dentro de src/evals/,
    # que some a cada limpeza do repositorio.
    ("""all_images = torch.load(f"evals/all_images.pt")""",
     """all_images = torch.load(f"{data_path}/evals/all_images.pt")"""),
    ('''parser.add_argument(
    "--seed",type=int,default=42,
)''', ENH_ARGS),
    # caminho hardcoded do cluster da Stability -> f-string com o cache_dir local
    ("base_ckpt_path = '/weka/proj-fmri/paulscotti/stable-research/zavychromaxl_v30.safetensors'",
     "base_ckpt_path = f'{cache_dir}/zavychromaxl_v30.safetensors'"),
    # modelos treinados com --no-blurry_recon nao geram all_blurryrecons.pt;
    # o arquivo so e consumido dentro dos blocos "if plotting:"
    ('all_blurryrecons = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons.pt")',
     '_blurry_path = f"evals/{model_name}/{model_name}_all_blurryrecons.pt"\n'
     'all_blurryrecons = torch.load(_blurry_path) if os.path.exists(_blurry_path) else None'),
    ('all_blurryrecons = transforms.Resize((768,768))(all_blurryrecons).float()',
     'if all_blurryrecons is not None:\n'
     '    all_blurryrecons = transforms.Resize((768,768))(all_blurryrecons).float()'),
    ('print(all_images.shape, all_recons.shape, all_clipvoxels.shape, all_blurryrecons.shape, all_predcaptions.shape)',
     'print(all_images.shape, all_recons.shape, all_clipvoxels.shape,\n'
     '      all_blurryrecons.shape if all_blurryrecons is not None else None, all_predcaptions.shape)'),
    # --- mesmas correcoes do recon_inference, aplicadas preventivamente ---
    # "plotting" so e definido sob is_interactive(); em modo script da NameError
    ("if utils.is_interactive(): plotting=True",
     "plotting = utils.is_interactive()"),
    # o refinador tambem e construido em fp32 na CPU e movido para a GPU sem
    # coleta. Em fp16 cabe folgado; o first stage fica em fp32 porque o config
    # desabilita autocast nele e decode de VAE em fp16 gera artefatos.
    ("base_engine = DiffusionEngine(network_config=network_config,",
     "_prev_dtype = torch.get_default_dtype()\n"
     "torch.set_default_dtype(torch.float16)\n"
     "base_engine = DiffusionEngine(network_config=network_config,"),
    ("base_engine.eval().requires_grad_(False)\nbase_engine.to(device)",
     "torch.set_default_dtype(_prev_dtype)\n"
     "base_engine.eval().requires_grad_(False)\nbase_engine.to(device)\n"
     "base_engine.first_stage_model.float()\n"
     "import gc; gc.collect()"),
    ("base_text_embedder1.to(device)",
     "base_text_embedder1.to(device)\nimport gc; gc.collect()"),
    ("base_text_embedder2.to(device)",
     "base_text_embedder2.to(device)\nimport gc; gc.collect()"),
    # O loop acumulava em 768x768 com torch.vstack a cada imagem: o tensor final
    # tem 7 GB e cada vstack realoca tudo de novo (7 GB novos + 7 GB antigos).
    # Como a linha seguinte reduz para 256x256 no fim, basta reduzir na hora de
    # acumular -- 9x menos memoria -- e juntar com um unico torch.cat.
    ("""        samples = samples.cpu()[None]
        if all_enhancedrecons is None:
            all_enhancedrecons = samples
        else:
            all_enhancedrecons = torch.vstack((all_enhancedrecons, samples))
            
all_enhancedrecons = transforms.Resize((256,256))(all_enhancedrecons).float()""",
     """        samples = transforms.Resize((256,256))(samples).float().cpu()[None]
        _enhanced_chunks.append(samples)

all_enhancedrecons = torch.cat(_enhanced_chunks, dim=0)
del _enhanced_chunks"""),
    ("all_enhancedrecons = None", "all_enhancedrecons = None\n_enhanced_chunks = []"),
    # all_recons e salvo em 256x256 (0.79 GB) mas o script faz upscale das 1000
    # imagens para 768x768 de uma vez, ocupando 7.08 GB de RAM -- 6.3 GB a mais
    # que o necessario. O refinador processa uma imagem por vez, entao o resize
    # cabe dentro do loop.
    ("all_recons = transforms.Resize((768,768))(all_recons).float()",
     "all_recons = all_recons.float()  # resize para 768 feito por imagem no loop"),
    ("        image = all_recons[[img_idx]]",
     "        image = transforms.Resize((768,768))(all_recons[[img_idx]]).float()"),
    # instrumentacao de memoria, como no recon_inference
    ("import utils", """import utils
import os as _os
def _rss(tag):
    with open(f"/proc/{_os.getpid()}/status") as _f:
        for _l in _f:
            if _l.startswith("VmRSS"):
                import torch as _t
                _v = _t.cuda.memory_allocated()//2**20 if _t.cuda.is_available() else 0
                print(f"[RSS] {int(_l.split()[1])//1024:6d} MB ram | {_v:5d} MB vram  {tag}", flush=True)
                return"""),
    ("base_engine.first_stage_model.float()",
     "base_engine.first_stage_model.float()\n_rss('depois do ZavyChromaXL')"),
    ("base_text_embedder2.to(device)\nimport gc; gc.collect()",
     "base_text_embedder2.to(device)\nimport gc; gc.collect()\n_rss('depois dos text embedders')"),
    # Salvamento parcial a cada 100 imagens. O loop leva ~2h e salva tudo so no
    # fim: qualquer interrupcao (queda de energia, OOM) perdia a execucao
    # inteira. Com isto, uma nova execucao retoma de onde parou.
    ("""all_enhancedrecons = None
_enhanced_chunks = []
for img_idx in tqdm(range(len(all_recons))):""",
     """all_enhancedrecons = None
_partial_path = f"evals/{model_name}/{model_name}_enhanced_partial.pt"
if os.path.exists(_partial_path):
    _enhanced_chunks = list(torch.load(_partial_path))
    print(f"retomando de {len(_enhanced_chunks)} imagens ja refinadas", flush=True)
else:
    _enhanced_chunks = []
_start_idx = len(_enhanced_chunks)
for img_idx in tqdm(range(len(all_recons))):
    if img_idx < _start_idx:
        continue"""),
    ("""        _enhanced_chunks.append(samples)""",
     """        _enhanced_chunks.append(samples)
        if len(_enhanced_chunks) % 100 == 0:
            torch.save(_enhanced_chunks, _partial_path + ".tmp")
            os.replace(_partial_path + ".tmp", _partial_path)  # troca atomica"""),
    ("""all_enhancedrecons = torch.cat(_enhanced_chunks, dim=0)
del _enhanced_chunks""",
     """all_enhancedrecons = torch.cat(_enhanced_chunks, dim=0)
del _enhanced_chunks
if os.path.exists(_partial_path):
    os.remove(_partial_path)"""),
])
