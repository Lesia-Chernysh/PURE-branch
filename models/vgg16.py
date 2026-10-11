import torch
import timm
from utils.lrp_canonizers import PUREResNetCanonizer
from zennit.composites import EpsilonPlusFlat

def get_vgg16(ckpt_path=None, n_class: int = None):
    return timm.create_model("vgg16.tv_in1k", pretrained=True).to(device).eval()


def get_vgg16_canonizer():
    raise NotImplementedError


def get_vgg16_composite():
    raise NotImplementedError
