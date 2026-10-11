import torch
import timm

def get_vit(ckpt_path=None, n_class: int = None):
    return timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True).eval()


def get_vit_canonizer():
    raise NotImplementedError


def get_vit_composite():
    raise NotImplementedError
