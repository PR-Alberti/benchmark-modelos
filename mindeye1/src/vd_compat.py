"""Contorna dois defeitos do diffusers 0.23 (o do ambiente do MindEye2) no Versatile Diffusion.

1. O UNetFlatConditionModel do text_unet passa a get_down_block/get_up_block argumentos que elas
nao aceitam (transformer_layers_per_block, attention_type, attention_head_dim, resolution_idx;
corrigido no diffusers 0.24). Aqui eles sao descartados, e os blocos ficam com os padroes.

Para o MindEye1 isso nao muda nada: o text_unet so alimenta o ramo de texto dos
DualTransformer2DModel, e o MindEye1 da peso zero a esse ramo (text_image_ratio = .0 no
Reconstructions.py). O image_unet, que gera a imagem, e um UNet2DConditionModel comum e nao passa
por aqui.

2. Os blocos da UNet passam encoder_attention_mask ao DualTransformer2DModel, que nao aceita esse
argumento. O MindEye1 nao usa mascara (o valor e sempre None), entao ele e descartado.

Importe antes do from_pretrained.
"""
import inspect

import diffusers.pipelines.versatile_diffusion.modeling_text_unet as _text_unet
from diffusers.models.dual_transformer_2d import DualTransformer2DModel


def _so_o_que_aceita(fabrica):
    aceitos = set(inspect.signature(fabrica).parameters)

    def chama(*args, **kwargs):
        return fabrica(*args, **{k: v for k, v in kwargs.items() if k in aceitos})
    return chama


if not getattr(_text_unet, "_benchmark_modelos", False):
    _text_unet.get_down_block = _so_o_que_aceita(_text_unet.get_down_block)
    _text_unet.get_up_block = _so_o_que_aceita(_text_unet.get_up_block)
    _text_unet._benchmark_modelos = True

    _forward_dual = DualTransformer2DModel.forward

    def _forward_sem_mascara(self, *args, encoder_attention_mask=None, **kwargs):
        assert encoder_attention_mask is None, "o MindEye1 nao usa mascara de atencao"
        return _forward_dual(self, *args, **kwargs)

    DualTransformer2DModel.forward = _forward_sem_mascara
