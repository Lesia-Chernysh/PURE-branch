import os
import re

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from crp.helper import get_layer_names
from typing import List
from transformers import AutoProcessor

# for colorize
import html
import matplotlib
from matplotlib.colors import Normalize, to_hex
from IPython.display import HTML

def colorize(tokens:list[str], weights:list[float], target:str=None):
    """returns a html heatmap representation of the tokens colored by weights.
        tokens and weights should match in lenghth
        target is optionally the token the weights were calculated for
    """
    template = '<span style="color: black; background-color: {}">{}</span>'
    colors = map(to_hex, matplotlib.colormaps['bwr'](Normalize(vmin=-1, vmax=1)(weights)))
    colored_output = ''
    for token, color in zip(tokens, colors):
        colored_output += template.format(color, '&nbsp'+html.escape(token)+'&nbsp')
    if target:
        colored_output += template.format('lime', '&nbsp'+html.escape(target)+'&nbsp')
    return HTML(colored_output)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as stream:
        try:
            config_text = os.path.expandvars(stream.read())
            unresolved = sorted(set(re.findall(r"\$\{([^}]+)\}", config_text)))
            if unresolved:
                names = ", ".join(unresolved)
                raise EnvironmentError(
                    f"Unresolved environment variable(s) in {config_path}: {names}. "
                    "Set them before running the command or use an absolute path."
                )
            config = yaml.safe_load(config_text) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
        config["wandb_id"] = os.path.basename(config_path)[:-5]
    return config


def get_layer_names_model(model: torch.nn.Module, model_name: str) -> List[str]:
    """
    Get layer names of a model.
    :param model:   model
    :param model_name:  model name (e.g. vgg16)
    :return:
    """
    if model_name == "resnet50_transformers":
        layer_names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Conv2d)
            and name.endswith(".conv2")
        ]
    elif "resnet" in model_name:
        layer_names = get_layer_names(model, [InspectionLayer])
    elif model_name == "vgg16":
        layer_names = [
            name 
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Conv2d)
            and "head" not in name
        ]
    elif model_name == "vit":
        layer_names = [
            name
            for name, module in model.named_modules()
            if type(module).__name__ == "Mlp" and name.endswith("mlp")
        ]
    elif model_name == "vilt":
        selected_names = [
            "model.vilt.encoder.layer.1.output.dense",
            "model.vilt.encoder.layer.2.output.dense",
            "model.vilt.encoder.layer.3.output.dense",
            "model.vilt.encoder.layer.4.output.dense",
            "model.vilt.encoder.layer.5.output.dense",
            "model.vilt.encoder.layer.6.output.dense",
            "model.vilt.encoder.layer.7.output.dense",
            "model.vilt.encoder.layer.8.output.dense",
            "model.vilt.encoder.layer.9.output.dense",
            "model.vilt.encoder.layer.10.output.dense",
            "model.vilt.encoder.layer.11.output.dense",
        ]
        
        layer_names = []
        
        for name, layer in model.named_modules():
          if name in selected_names:
            layer_names.append(name)

    #elif model_name == "qwen":
    #    layer_names = [
    #        
    #    ]
    else:
        raise NotImplementedError
    return layer_names


def validate_layer_name(layer_name: str, layer_names: List[str]) -> str:
    """Validate a requested layer and return it with a useful error on mismatch."""
    if layer_name not in layer_names:
        available = ", ".join(layer_names)
        raise ValueError(
            f"Unknown layer {layer_name!r}. Available inspection layers: {available}"
        )
    if layer_names.index(layer_name) == 0:
        raise ValueError(
            f"Layer {layer_name!r} has no preceding inspection layer for PURE."
        )
    return layer_name


def pad_neuron_references(tensors: List[torch.Tensor]):
    """Pad ragged per-neuron reference tensors without losing the neuron axis."""
    if not tensors:
        raise ValueError("No per-neuron tensors were collected.")
    lengths = torch.tensor([tensor.shape[0] for tensor in tensors], dtype=torch.long)
    if torch.any(lengths == 0):
        empty = torch.where(lengths == 0)[0].tolist()
        raise ValueError(f"No valid reference samples for neurons: {empty}")
    return pad_sequence(tensors, batch_first=True, padding_value=0.0), lengths


def reference_lengths(tensors, tensor_key: str) -> torch.Tensor:
    """Read valid reference lengths, with compatibility for old dense artifacts."""
    key = f"{tensor_key}_lengths"
    tensor = tensors[tensor_key]
    if key in tensors:
        lengths = tensors[key].to(dtype=torch.long)
    elif "reference_lengths" in tensors:
        lengths = tensors["reference_lengths"].to(dtype=torch.long)
    else:
        lengths = torch.full((tensor.shape[0],), tensor.shape[1], dtype=torch.long)
    if lengths.shape != (tensor.shape[0],):
        raise ValueError(
            f"{key} must have shape ({tensor.shape[0]},), got {tuple(lengths.shape)}"
        )
    if torch.any(lengths < 0) or torch.any(lengths > tensor.shape[1]):
        raise ValueError(f"Invalid reference lengths for {tensor_key!r}.")
    return lengths


class InspectionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_pil_list, processor: AutoProcessor = None):
        self.data_pil_list=data_pil_list
        self.processor=processor
        self.length=len(data_pil_list)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        #for key in self.keys:
        sample=self.data_pil_list[idx]
        if self.processor:
            sample=self.processor(images=self.data_pil_list[idx], return_tensors="pt")
            return sample['pixel_values'].squeeze()
        
        return sample.squeeze()
        
            
