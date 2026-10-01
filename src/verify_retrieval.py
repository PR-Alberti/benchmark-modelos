import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
from mindeye_ridge import paths
paths.add_vendored_to_path()
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
from mindeye_ridge import utils
from mindeye_ridge.models import *

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:",device)

# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "final_subj01_pretrained_40sess_24bs"
    print("model_name:", model_name)

    # other variables can be specified in the following string:
    jupyter_args = f"--data_path=/weka/proj-medarc/shared/mindeyev2_dataset \
                    --cache_dir=/weka/proj-medarc/shared/mindeyev2_dataset \
                    --model_name={model_name} --subj=1 \
                    --hidden_dim=4096 --n_blocks=4 --new_test"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default=str(paths.DATA),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default=str(paths.DATA),
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=2048,
)
parser.add_argument(
    "--new_test",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--frozen_ckpt",type=str,default=None,
    help="ckpt de onde vem tudo menos a ridge (veja o mesmo argumento no recon_inference.py)",
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# seed all random functions
utils.seed_everything(seed)

# make output directory
os.makedirs(paths.EVALS, exist_ok=True)
os.makedirs(f"{paths.EVALS}/{model_name}",exist_ok=True)

voxels = {}
# Load hdf5 data for betas
f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to("cpu")
num_voxels = betas[0].shape[-1]
voxels[f'subj0{subj}'] = betas
print(f"num_voxels for subj0{subj}: {num_voxels}")

if not new_test: # using old test set from before full dataset released (used in original MindEye paper)
    if subj==3:
        num_test=2113
    elif subj==4:
        num_test=1985
    elif subj==6:
        num_test=2113
    elif subj==8:
        num_test=1985
    else:
        num_test=2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
else: # using larger test set from after full dataset released
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
    
print(test_url)
def my_split_by_node(urls): return urls
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")

# Prep images but don't load them all to memory
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

# Prep test voxels and indices of test images
test_images_idx = []
test_voxels_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
    test_voxels_idx = np.append(test_images_idx, behav[:,0,5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:,0,0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)

assert (test_i+1) * num_test == len(test_voxels) == len(test_images_idx)
print(test_i, len(test_voxels), len(test_images_idx), len(np.unique(test_images_idx)))

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
utils.log_memory('antes de mover CLIP bigG para GPU')
clip_img_embedder = clip_img_embedder.half()  # 10 GB -> 5 GB de VRAM
clip_img_embedder.to(device)
import gc; gc.collect()
utils.log_memory('depois de mover CLIP bigG fp16 + gc')
clip_seq_dim = 256
clip_emb_dim = 1664

if blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    
from mindeye_ridge.models import MindEyeModule, RidgeRegression
model = MindEyeModule()

model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)

from diffusers.models.vae import Decoder
from mindeye_ridge.models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim,
                          blurry_recon=blurry_recon) 
utils.count_params(model.ridge)
utils.count_params(model.backbone)
utils.count_params(model)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb"
    )

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
model.to(device)

utils.count_params(model.diffusion_prior)
utils.count_params(model)

# Load pretrained model ckpt
tag='last'
outdir = str(paths.TRAIN_LOGS / model_name)
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
try:
    checkpoint = torch.load(outdir+f'/{tag}.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    # o treinado sem prior nao tem diffusion_prior no ckpt; o retrieval nao usa
    # o prior, mas o modelo aqui e montado com ele e o load e estrito
    if frozen_ckpt is not None:
        _base = torch.load(frozen_ckpt, map_location='cpu', mmap=True)['model_state_dict']
        state_dict.update({k: v for k, v in _base.items() if k.startswith(('backbone.', 'diffusion_prior.'))})
        del _base
        print(f"backbone e diffusion_prior carregados de {frozen_ckpt}")
    model.load_state_dict(state_dict, strict=True)
    del checkpoint
except: # probably ckpt is saved using deepspeed format
    import deepspeed
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
print("ckpt loaded!")
utils.log_memory('depois do nosso ckpt')


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

        all_clipvoxels = clip_voxels.cpu() if all_clipvoxels is None \
                         else torch.vstack((all_clipvoxels, clip_voxels.cpu()))

print("all_clipvoxels", all_clipvoxels.shape, flush=True)
os.makedirs(f"{paths.EVALS}/{model_name}", exist_ok=True)
torch.save(all_clipvoxels, f"{paths.EVALS}/{model_name}/{model_name}_all_clipvoxels.pt")

# libera o modelo antes de calcular o retrieval
del model
import gc; gc.collect(); torch.cuda.empty_cache()

# ============================================================================
# retrieval -- protocolo identico ao do final_evaluations.py
# ============================================================================
all_images = torch.load(f"{data_path}/evals/all_images.pt")
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

resultado = {}
for chave, nome, arr in (("fwd", "fwd (imagem->cerebro: cada imagem acha o seu cerebro)", fwds),
                         ("bwd", "bwd (cerebro->imagem: cada cerebro acha a sua imagem)", bwds)):
    m = np.mean(arr); sd = np.std(arr) / np.sqrt(len(arr))
    ci = stats.norm.interval(0.95, loc=m, scale=sd)
    print(f"RESULTADO {model_name} | {nome}: {m:.4f}  IC95% [{ci[0]:.4f}, {ci[1]:.4f}]", flush=True)
    resultado[chave] = {"media": float(m), "ic95": [float(ci[0]), float(ci[1])]}
with open(f"{paths.EVALS}/{model_name}/{model_name}_retrieval.json", "w") as f:
    json.dump(resultado, f, indent=1)
