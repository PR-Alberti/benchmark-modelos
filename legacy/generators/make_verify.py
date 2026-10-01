"""Gera verify_retrieval.py: calcula o retrieval de um modelo SEM rodar difusao.

Reaproveita todo o pipeline de dados do recon_inference.py (mesmo carregamento de
betas, mesmo test set, mesmo embedder) e substitui o loop de geracao por um loop
que so extrai clip_voxels. Em seguida aplica o protocolo de retrieval identico ao
do final_evaluations.py: 300-way, 30 sorteios, media.

Serve para comparar modelos no mesmo sujeito e com o mesmo codigo, sem pagar as
~2h de difusao.
"""
src = open('recon_inference.py').read()

CORTE = "# setup text caption networks"
assert CORTE in src, "marcador de corte nao encontrado"
cabeca = src.split(CORTE)[0]

CAUDA = '''
# ============================================================================
# clip_voxels (forward puro, sem difusao)
# ============================================================================
minibatch_size = 16
all_clipvoxels = None
model.to(device); model.eval().requires_grad_(False)

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    for batch in tqdm(range(0, len(np.unique(test_images_idx)), minibatch_size)):
        uniq_imgs = np.unique(test_images_idx)[batch:batch+minibatch_size]
        voxel = None
        for uniq_img in uniq_imgs:
            locs = np.where(test_images_idx == uniq_img)[0]
            if len(locs) == 1:   locs = locs.repeat(3)
            elif len(locs) == 2: locs = locs.repeat(2)[:3]
            assert len(locs) == 3
            voxel = test_voxels[None, locs] if voxel is None else torch.vstack((voxel, test_voxels[None, locs]))
        voxel = voxel.to(device)

        for rep in range(3):   # media das 3 repeticoes, como no recon_inference
            voxel_ridge = model.ridge(voxel[:, [rep]], 0)
            _, clip_voxels0, _ = model.backbone(voxel_ridge)
            clip_voxels = clip_voxels0 if rep == 0 else clip_voxels + clip_voxels0
        clip_voxels /= 3

        all_clipvoxels = clip_voxels.cpu() if all_clipvoxels is None \\
                         else torch.vstack((all_clipvoxels, clip_voxels.cpu()))

print("all_clipvoxels", all_clipvoxels.shape, flush=True)
os.makedirs(f"evals/{model_name}", exist_ok=True)
torch.save(all_clipvoxels, f"evals/{model_name}/{model_name}_all_clipvoxels.pt")

# libera o modelo antes de calcular o retrieval
del model
import gc; gc.collect(); torch.cuda.empty_cache()

# ============================================================================
# retrieval -- protocolo identico ao do final_evaluations.py
# ============================================================================
all_images = torch.load("evals/all_images.pt")
from scipy import stats

fwds, bwds = [], []
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    np.random.seed(seed)
    for _ in tqdm(range(30)):
        samps = np.random.choice(np.arange(len(all_images)), size=300, replace=False)
        emb  = clip_img_embedder(all_images[samps].to(device)).float()
        emb_ = all_clipvoxels[samps].to(device).float()
        emb  = nn.functional.normalize(emb.reshape(len(emb), -1), dim=-1)
        emb_ = nn.functional.normalize(emb_.reshape(len(emb_), -1), dim=-1)
        labels = torch.arange(len(emb)).to(device)
        fwds = np.append(fwds, utils.topk(utils.batchwise_cosine_similarity(emb_, emb), labels, k=1).item())
        bwds = np.append(bwds, utils.topk(utils.batchwise_cosine_similarity(emb, emb_), labels, k=1).item())

for nome, arr in (("fwd (cerebro->imagem, 'Image Retrieval')", fwds),
                  ("bwd (imagem->cerebro, 'Brain Retrieval')", bwds)):
    m = np.mean(arr); sd = np.std(arr) / np.sqrt(len(arr))
    ci = stats.norm.interval(0.95, loc=m, scale=sd)
    print(f"RESULTADO {model_name} | {nome}: {m:.4f}  IC95% [{ci[0]:.4f}, {ci[1]:.4f}]", flush=True)
'''

open('verify_retrieval.py', 'w').write(cabeca + CAUDA)
print(f"verify_retrieval.py gerado ({len((cabeca+CAUDA).splitlines())} linhas)")
