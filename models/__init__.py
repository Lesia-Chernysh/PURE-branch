import torch
from models.timm_resnet import get_resnet_timm, get_resnet50_timm, get_resnet34_timm, get_resnet101_timm, \
    get_resnet_canonizer
from models.resnet50_transformers import get_resnet50, get_resnet50_canonizer

MODELS = {
    "resnet50_timm": get_resnet50_timm,
    "resnet34_timm": get_resnet34_timm,
    "resnet101_timm": get_resnet101_timm,
    "resnet50_transformers": get_resnet50
}

CANONIZERS = {
    "resnet50_timm": get_resnet_canonizer,
    "resnet34_timm": get_resnet_canonizer,
    "resnet101_timm": get_resnet_canonizer,
    "resnet50_transformers": get_resnet50_canonizer
}

COMPOSITES = {
    "resnet50_transformers": get_resnet50_composite,
}

def get_canonizer(model_name):
    assert model_name in list(CANONIZERS.keys()), f"No canonizer for model '{model_name}' available"
    return [CANONIZERS[model_name]()]

def get_composite(model_name):
    assert model_name in list(COMPOSITES.keys()), f"No composite for model '{model_name}' available"
    return [COMPOSITES[model_name]()]
    

def get_fn_model_loader(model_name: str) -> torch.nn.Module:
    if model_name in MODELS:
        fn_model_loader = MODELS[model_name]
        return fn_model_loader
    else:
        raise KeyError(f"Model {model_name} not available")
