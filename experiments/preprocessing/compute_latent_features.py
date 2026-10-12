import argparse
import os

import numpy as np
import torch as torch
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import load_maximization
from crp.visualization import FeatureVisualization
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from zennit.core import Composite

from datasets import get_dataset
from experiments import preprocessing
from models import get_fn_model_loader
from utils.helper import get_layer_names_model, pad_neuron_references, validate_layer_name

import yaml



def get_args():
    parser = argparse.ArgumentParser(description='Compute relevances and activations')
    parser.add_argument('--config_file', type=str,
                        default="configs/imagenet/resnet101_timm.yaml")
    parser.add_argument('--split', type=str, default=None,
                        help="Process one split instead of all splits listed in the config.")
    parser.add_argument('--layer_name', type=str, default=None)
    return parser.parse_args()


def main(model_name,
         ckpt_path,
         dataset_name,
         data_path,
         split,
         batch_size,
         layer_name,
         crp_split=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(dataset_name)(data_path=data_path, preprocessing=True, split=split)

    print("Dataset loaded")

    model = get_fn_model_loader(model_name)(ckpt_path=ckpt_path, n_class=dataset.num_classes)
    model = model.to(device)
    model.eval()
    print("Model loaded")

    layer_names = get_layer_names_model(model, model_name)

    crp_split = crp_split or split
    fname = f"{model_name}_{dataset_name}_{crp_split}"
             
    # TODO: I changed max_target from max to sum and abs_norm from False to True
    if model_name == "vit":
        # import CRP from my branch 'CRP'
        from crp.concepts import TransformerChannelConcept
        from crp.visualization import PredictionFeatureVisualization

        concept = TransformerChannelConcept()
        attribution = CondAttribution(concept, model)
        layer_map = {layer: concept for layer in layer_names}

        predicted_targets = torch.load("vit_predicted_targets.pt")
        
        fv = PredictionFeatureVisualization(concept, attribution, dataset, layer_map,
                                  preprocess_fn=None,
                                  path=f"crp_files/{fname}",
                                  device=device,
                                  max_target="sum", abs_norm=True,
                                  predicted_targets=predicted_targets)
    elif model_name == "vilt":
        from crp.concepts import TransformerChannelConcept
        from crp.visualization import ViLTFeatureVisualization
        from lxt.explicit.models.vilt import ViltForQuestionAnswering, attnlrp as lrp_rule_composite
        from models.vilt import ViltCRPWrapper
        from transformers import ViltProcessor
        
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        concept = TransformerChannelConcept()

        crp_model = ViltCRPWrapper(model)
        attribution = CondAttribution(crp_model, device, no_param_grad=True)
        layer_map = {layer: concept for layer in layer_names}

        fv = ViLTFeatureVisualization(
            attribution,
            dataset,
            layer_map,
            processor,
            path=f"crp_files/{fname}",
            device=device,
            max_target="sum", abs_norm=True,
        )
        
    else:
        attribution = CondAttribution(model)
        concept = ChannelConcept()
        layer_map = {layer: concept for layer in layer_names}
        fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                                  path=f"crp_files/{fname}", max_target="sum", abs_norm=True)

    grad_composite = Composite()

    layer_name = validate_layer_name(layer_name or layer_names[-1], layer_names)
    prev_layer = layer_names[layer_names.index(layer_name) - 1]

    d_c_sorted, a, rf_c_sorted = load_maximization(fv.ActMax.PATH, layer_name)

    d_c_sorted = d_c_sorted[:, :]
    num_neurons = d_c_sorted.shape[1]

    max_activations = []
    mean_activations = []
    cond_relevances = []
    
    for i, neuron in enumerate(tqdm(range(num_neurons))):
        max_activations.append([])
        mean_activations.append([])
        cond_relevances.append([])
    
        most_act_sample_ids = d_c_sorted[:, neuron]
    
        out_of_range = most_act_sample_ids >= len(dataset)
        if np.any(out_of_range):
            invalid = most_act_sample_ids[out_of_range][:10].tolist()
            raise ValueError(
                f"CRP references for neuron {neuron} contain dataset indices "
                f"outside the current {split!r} split (size {len(dataset)}), "
                f"e.g. {invalid}."
            )
    
        # Remove padding/invalid reference IDs.
        most_act_sample_ids = most_act_sample_ids[
            most_act_sample_ids >= 0
        ]
    
        dataset_subset = torch.utils.data.Subset(
            dataset,
            most_act_sample_ids.tolist(),
        )
    
        dataloader = DataLoader(
            dataset_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
        )
    
        for x, _ in dataloader:
            x = x.to(device).requires_grad_()
    
            conditions = [
                {layer_name: [int(neuron)]}
            ]
    
            attr = attribution(
                x,
                conditions,
                grad_composite,
                record_layer=layer_names,
                start_layer=layer_name,
                init_rel=lambda act: act.clamp(min=0),
            )
    
            # Conditional relevance arriving at the preceding layer.
            lower_gradient = (
                attr.relevances[prev_layer]
                .detach()
                .cpu()
            )
            lower_activation = (
                attr.activations[prev_layer]
                .detach()
                .cpu()
            )
    
            lower_relevance = lower_gradient * lower_activation
    
            conditional_relevance = concept.attribute(
                lower_relevance,
                abs_norm=True,
            )
    
            cond_relevances[i].append(
                conditional_relevance.detach().cpu()
            )
    
            activation = (
                attr.activations[layer_name]
                .detach()
                .cpu()
                .clamp(min=0)
            )
    
            if model_name == "vit":
                # timm ViT MLP: [batch, tokens, channels]
                if activation.ndim != 3:
                    raise ValueError(
                        "Expected ViT activation [B, tokens, channels], "
                        f"got {tuple(activation.shape)}"
                    )
    
                max_activations[i].append(
                    activation.amax(dim=1)
                )
                mean_activations[i].append(
                    activation.mean(dim=1)
                )
    
            elif model_name == "vilt":
                if activation.ndim != 3:
                    raise ValueError(
                        "Expected ViLT activation [B, sequence, channels], "
                        f"got {tuple(activation.shape)}"
                    )
            
                # Number of positions occupied by the padded text sequence.
                text_length = inputs["input_ids"].shape[1]
            
                # ViLT concatenates text tokens followed by image tokens.
                image_activation = activation[:, text_length:, :]
            
                if image_activation.shape[1] == 0:
                    raise RuntimeError(
                        f"No image tokens found: activation shape={tuple(activation.shape)}, "
                        f"text length={text_length}"
                    )
            
                # Position zero of the image sequence is ViLT's image CLS token.
                # Exclude it when measuring spatial image-patch activations.
                image_patch_activation = image_activation[:, 1:, :]
            
                if image_patch_activation.shape[1] == 0:
                    raise RuntimeError("ViLT activation contains no image-patch tokens.")
            
                max_activations[i].append(
                    image_patch_activation.amax(dim=1)
                )
                mean_activations[i].append(
                    image_patch_activation.mean(dim=1)
                )
                
            else:
                # CNN: [batch, channels, height, width]
                if activation.ndim != 4:
                    raise ValueError(
                        "Expected CNN activation [B, channels, H, W], "
                        f"got {tuple(activation.shape)}"
                    )
    
                max_activations[i].append(
                    activation.amax(dim=(2, 3))
                )
                mean_activations[i].append(
                    activation.mean(dim=(2, 3))
                )
    
        if not max_activations[i]:
            raise RuntimeError(
                f"No valid reference images found for neuron {neuron}."
            )
    
        # Join all dataloader batches belonging to this neuron.
        max_activations[i] = torch.cat(
            max_activations[i],
            dim=0,
        )
        mean_activations[i] = torch.cat(
            mean_activations[i],
            dim=0,
        )
        cond_relevances[i] = torch.cat(
            cond_relevances[i],
            dim=0,
        )
    
    
    # These operations must be outside the neuron loop.
    max_activations, max_act_lengths = pad_neuron_references(
        max_activations
    )
    mean_activations, mean_act_lengths = pad_neuron_references(
        mean_activations
    )
    cond_relevances, cond_rel_lengths = pad_neuron_references(
        cond_relevances
    )
    
    if not torch.equal(max_act_lengths, cond_rel_lengths):
        raise RuntimeError(
            "Maximum activations and conditional relevances have "
            "different reference counts."
        )
    
    if not torch.equal(mean_act_lengths, cond_rel_lengths):
        raise RuntimeError(
            "Mean activations and conditional relevances have "
            "different reference counts."
        )
    
    path = f"results/global_features/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)
    
    save_file(
        {
            "max_act": max_activations.contiguous(),
            "mean_act": mean_activations.contiguous(),
            "cond_rel": cond_relevances.contiguous(),
            "reference_lengths": cond_rel_lengths.contiguous(),
        },
        f"{path}/latent_features_{layer_name}_{split}.safetensors",
    )


if __name__ == "__main__":
    args = get_args()

    #config = load_config(args.config_file)
    with open(args.config_file, encoding="utf-8") as file:
        config_text = os.path.expandvars(file.read())
    
    config = yaml.safe_load(config_text)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    batch_size = 40
    data_path = config.get('data_path', None)
    print(f"data_path: {data_path}")
    ckpt_path = config.get('ckpt_path', None)
    splits = [args.split] if args.split else config.get('splits', ['test'])
    layer_name = args.layer_name or config.get('layer_name')
    crp_split = config.get('crp_split')

    for split in splits:
        main(model_name, ckpt_path, dataset_name, data_path, split, batch_size,
             layer_name, crp_split=crp_split)
