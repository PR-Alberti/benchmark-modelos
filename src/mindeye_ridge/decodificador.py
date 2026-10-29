"""Do embedding CLIP ViT-bigG (256 x 1664) para a imagem e a legenda: o unCLIP e o GIT do MindEye2.

Nenhuma destas redes viu fMRI: o unCLIP (SDXL ajustado para gerar a partir do embedding de imagem do
bigG), o conversor bigG -> CLIP ViT-L e o GIT (legendas do COCO) foram treinados so com imagens e
texto. Por isso servem a qualquer modelo que preveja o embedding bigG, nao so ao MindEye2: o
recon_inference.py as usa sobre a saida do prior, e com --from_embeddings sobre um embedding pronto
(o do FRR).

A ordem de construcao importa: a inicializacao das redes consome o gerador aleatorio antes de os pesos
serem carregados, e a amostragem que vem depois depende disso. Chame na ordem do recon_inference.py:
carrega_git, carrega_conversor, carrega_unclip.
"""
import gc

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from . import paths, utils

CLIP_SEQ, CLIP_DIM = 256, 1664                 # bigG: tokens x dimensao
CLIP_TEXT_SEQ, CLIP_TEXT_DIM = 257, 1024       # ViT-L, a entrada do GIT


def carrega_git(device):
    """O GIT (microsoft/git-large-coco) que gera a legenda a partir do embedding ViT-L: (processor, modelo)."""
    from transformers import AutoProcessor
    from .modeling_git import GitForCausalLMClipEmb
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    clip_text_model.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
    clip_text_model.eval().requires_grad_(False)
    gc.collect()
    utils.log_memory('depois do GIT')
    return processor, clip_text_model


class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(CLIP_SEQ, CLIP_TEXT_SEQ)
        self.linear2 = nn.Linear(CLIP_DIM, CLIP_TEXT_DIM)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0,2,1))
        return x


def carrega_conversor(cache_dir, device):
    """Conversor do embedding bigG para o ViT-L que o GIT le (bigG_to_L_epoch8.pth)."""
    clip_convert = CLIPConverter()
    state_dict = torch.load(f"{cache_dir}/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
    clip_convert.load_state_dict(state_dict, strict=True)
    clip_convert.to(device)
    gc.collect() # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
    del state_dict
    return clip_convert


def carrega_unclip(cache_dir, device):
    """O unCLIP (unclip6_epoch0_step110000.ckpt) pronto para amostrar: (diffusion_engine, vector_suffix)."""
    paths.add_vendored_to_path()
    from generative_models.sgm.models.diffusion import DiffusionEngine

    config = OmegaConf.load(str(paths.VENDOR / "generative_models/configs/unclip6.yaml"))
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]

    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sampler_config['params']['num_steps'] = 38

    conditioner_config['params']['emb_models'][0]['params']['version'] = None
    _prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    diffusion_engine = DiffusionEngine(network_config=network_config,
                           denoiser_config=denoiser_config,
                           first_stage_config=first_stage_config,
                           conditioner_config=conditioner_config,
                           sampler_config=sampler_config,
                           scale_factor=scale_factor,
                           disable_first_stage_autocast=disable_first_stage_autocast)
    # set to inference
    torch.set_default_dtype(_prev_dtype)
    diffusion_engine.eval().requires_grad_(False)
    diffusion_engine.to(device)
    # o config traz disable_first_stage_autocast=True, entao o VAE roda fora do
    # autocast e recebe latentes fp32; alem disso decode em fp16 no SDXL gera
    # artefatos. Mantem o first stage em fp32 -- custa ~0.3 GB.
    diffusion_engine.first_stage_model.float()
    gc.collect()
    utils.log_memory('depois de construir o unCLIP fp16 na GPU')

    utils.log_memory('antes do unCLIP')
    ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
    # mmap=True mapeia o arquivo em disco em vez de materializar os 18 GB na
    # RAM. Sem isso o pico passa de 30 GB (o resto do pipeline ja ocupa ~12 GB) e
    # o processo morre por SIGKILL do OOM killer, sem traceback.
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', mmap=True)
    except (RuntimeError, ValueError, TypeError):  # ckpt em formato antigo
        ckpt = torch.load(ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])
    del ckpt
    gc.collect()

    batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
          "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
          "crop_coords_top_left": torch.zeros(1, 2).to(device)}
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    print("vector_suffix", vector_suffix.shape)
    return diffusion_engine, vector_suffix
