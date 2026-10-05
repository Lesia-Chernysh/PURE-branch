import torch
from utils.lrp_canonizers import PUREResNetCanonizer
from torchvision.models import resnet50, ResNet50_Weights
weights = ResNet50_Weights.IMAGENET1K_V2


def get_resnet50(ckpt_path=None, n_class: int = None):
    resnet = resnet50(weights=weights)
    if n_class is not None:
        if n_class != len(weights.meta['categories']):
            print("Mismatch in numbers of classes")
    return resnet


def get_resnet50_canonizer():
    return PUREResNetCanonizer()