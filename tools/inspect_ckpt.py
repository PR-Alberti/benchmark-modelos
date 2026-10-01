"""Inspeciona o checkpoint multi-sujeito para descobrir com que flags ele foi treinado."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindeye_ridge import paths

p = sys.argv[1] if len(sys.argv) > 1 else \
    str(paths.DATA / "train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth")
ck = torch.load(p, map_location='cpu')
sd = ck['model_state_dict']
print("epoch salvo:", ck.get('epoch'))
pref = {}
for k, v in sd.items():
    pref.setdefault(k.split('.')[0], [0, 0])
    pref[k.split('.')[0]][0] += 1
    pref[k.split('.')[0]][1] += v.numel()
for k, (n, tot) in pref.items():
    print(f"  {k:20s} {n:4d} tensores  {tot/1e6:9.2f}M params")
print()
print("ridge linears:", sorted(k for k in sd if k.startswith('ridge.')))
print("hidden_dim (ridge out):", sd['ridge.linears.0.weight'].shape[0] if 'ridge.linears.0.weight' in sd else '?')
print("num_voxels por linear:", {k: tuple(sd[k].shape) for k in sd if k.startswith('ridge.linears') and k.endswith('weight')})
print()
print("blurry_recon treinado?", any('blin1' in k or 'bupsampler' in k or 'b_maps_projector' in k for k in sd))
print("diffusion_prior presente?", any(k.startswith('diffusion_prior') for k in sd))
nb = len({k.split('.')[2] for k in sd if k.startswith('backbone.mixer_blocks1.')})
print("n_blocks:", nb)
