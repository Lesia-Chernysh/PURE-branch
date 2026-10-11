import torch
import timm

def get_vgg16(ckpt_path=None, n_class: int = None):
    return timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True).eval()


def get_vgg16_canonizer():
    raise NotImplementedError


def get_vgg16_composite():
    raise NotImplementedError
