import argparse
import os
from typing import Any

import numpy as np
import torch
import yaml

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import load_maximization
try:
    from crp.visualization import FeatureVisualization
except ImportError:
    # for ViLT
    try:
        from crp.visualization import ViLTFeatureVisualization
    except ImportError:
        # for Qwen
        from crp.visualization import QwenFeatureVisualization

from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from zennit.core import Composite

from datasets import get_dataset
from models import get_fn_model_loader
from utils.helper import (
    get_layer_names_model,
    pad_neuron_references,
    validate_layer_name,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute relevances and activations"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/imagenet/resnet101_timm.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Process one split instead of all splits listed in the config.",
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default=None,
    )
    return parser.parse_args()


def make_vilt_collate_fn(processor):
    """Create a collator that converts PIL images into ViLT inputs."""

    def collate_fn(batch):
        # ImageQuestionAnswerDataset returns:
        #     image, question, predicted_label
        images, questions, targets = zip(*batch)

        inputs = processor(
            images=list(images),
            text=list(questions),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return inputs, list(targets)

    return collate_fn


def move_batch_to_device(
    batch: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def load_dataset(
    dataset_name,
    data_path,
    split,
    model_name,
):
    dataset = get_dataset(dataset_name)(
        data_path=data_path,
        transform=False,
        preprocessing=False,
        split=split,
    )

    feature_dataset = dataset
    predicted_targets = None

    if model_name in {"vilt", "qwen"}:
        from datasets.imagenet_mini import get_imagenetmini_for_vqa

        predictions_path = {
            "vilt": "vilt_predicted_targets.pt",
            "qwen": "qwen_predicted_targets.pt",
        }[model_name]

        predicted_targets = torch.load(
            predictions_path,
            map_location="cpu",
            weights_only=True,
        )

        feature_dataset = get_imagenetmini_for_vqa(
            dataset=dataset,
            predicted_labels=predicted_targets,
        )

    return dataset, feature_dataset, predicted_targets


def prepare_vilt_inputs(
    inputs,
    model,
    device,
):
    inputs = {
        name: value.to(device)
        for name, value in dict(inputs).items()
    }

    pixel_values = (
        inputs["pixel_values"]
        .detach()
        .to(dtype=torch.float32)
        .requires_grad_(True)
    )

    input_embeds = (
        model.get_input_embeddings()(inputs["input_ids"])
        .detach()
        .to(dtype=torch.float32)
        .requires_grad_(True)
    )

    additional_forward_kwargs = {
        "token_type_ids": inputs["token_type_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_mask": inputs["pixel_mask"],
    }

    return (
        inputs,
        pixel_values,
        input_embeds,
        additional_forward_kwargs,
    )


def main(
    model_name,
    ckpt_path,
    dataset_name,
    data_path,
    split,
    batch_size,
    layer_name,
    crp_split=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset, feature_dataset, predicted_targets = load_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        split=split,
        model_name=model_name,
    )

    print(f"Dataset loaded: split={dataset.split}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dataset size: {len(feature_dataset)}")

    model = get_fn_model_loader(model_name)(
        ckpt_path=ckpt_path,
        n_class=dataset.num_classes,
    )
    model = model.to(device)
    model.eval()

    print("Model loaded")

    original_layer_names = get_layer_names_model(
        model,
        model_name,
    )

    if not original_layer_names:
        raise RuntimeError("No model layers were found.")

    requested_layer_name = validate_layer_name(
        layer_name or original_layer_names[-1],
        original_layer_names,
    )

    crp_split = crp_split or split
    fname = f"{model_name}_{dataset_name}_{crp_split}"

    processor = None

    if model_name == "vit":
        from crp.concepts import TransformerChannelConcept
        from crp.visualization import PredictionFeatureVisualization
        from lxt.explicit.models.vilt import (
            attnlrp,
        )

        # Required: install LXT's AttnLRP backward rules.
        attnlrp.register(model)

        concept = TransformerChannelConcept()
        attribution = CondAttribution(concept, model)

        attribution_layer_names = original_layer_names
        attribution_layer_name = requested_layer_name

        layer_map = {
            layer: concept
            for layer in attribution_layer_names
        }

        if predicted_targets is None:
            predicted_targets = torch.load(
                "vit_predicted_targets.pt",
                map_location="cpu",
                weights_only=True,
            )

        fv = PredictionFeatureVisualization(
            concept,
            attribution,
            dataset,
            layer_map,
            preprocess_fn=None,
            path=f"crp_files/{fname}",
            device=device,
            max_target="sum",
            abs_norm=True,
            predicted_targets=predicted_targets,
        )

    elif model_name == "vilt":
        from crp.concepts import TransformerChannelConcept
        from crp.visualization import PredictionFeatureVisualization
        from models.vilt import ViltCRPWrapper
        from transformers import ViltProcessor

        processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )

        concept = TransformerChannelConcept()

        crp_model = ViltCRPWrapper(model)
        crp_model = crp_model.to(device)
        crp_model.eval()

        attribution = CondAttribution(
            crp_model,
            device,
            no_param_grad=True,
        )

        # ViltCRPWrapper stores the original model under `model`.
        attribution_layer_names = [
            name if name.startswith("model.") else f"model.{name}"
            for name in original_layer_names
        ]
        attribution_layer_name = (
            requested_layer_name
            if requested_layer_name.startswith("model.")
            else f"model.{requested_layer_name}"
        )

        layer_map = {
            layer: concept
            for layer in attribution_layer_names
        }

        fv = PredictionFeatureVisualization(
            attribution,
            feature_dataset,
            layer_map,
            processor,
            path=f"crp_files/{fname}",
            device=device,
            max_target="sum",
            abs_norm=True,
            predicted_targets=predicted_targets,
        )

    else:
        concept = ChannelConcept()
        attribution = CondAttribution(model)

        attribution_layer_names = original_layer_names
        attribution_layer_name = requested_layer_name

        layer_map = {
            layer: concept
            for layer in attribution_layer_names
        }

        fv = FeatureVisualization(
            attribution,
            dataset,
            layer_map,
            preprocess_fn=dataset.preprocessing,
            path=f"crp_files/{fname}",
            max_target="sum",
            abs_norm=True,
        )

    layer_index = attribution_layer_names.index(
        attribution_layer_name
    )

    if layer_index == 0:
        raise ValueError(
            f"Cannot compute a preceding layer for "
            f"{attribution_layer_name!r} because it is the first layer."
        )

    previous_layer_name = attribution_layer_names[layer_index - 1]

    print(f"Selected layer: {attribution_layer_name}")
    print(f"Previous layer: {previous_layer_name}")

    grad_composite = Composite()

    # Maximization files use the original, unprefixed layer name.
    d_c_sorted, _, _ = load_maximization(
        fv.ActMax.PATH,
        requested_layer_name,
    )

    d_c_sorted = d_c_sorted[:, :]
    num_neurons = d_c_sorted.shape[1]

    max_activations = [[] for _ in range(num_neurons)]
    mean_activations = [[] for _ in range(num_neurons)]
    cond_relevances = [[] for _ in range(num_neurons)]

    for neuron in tqdm(
        range(num_neurons),
        desc="Computing latent features",
    ):
        most_act_sample_ids = d_c_sorted[:, neuron]

        if torch.is_tensor(most_act_sample_ids):
            most_act_sample_ids = (
                most_act_sample_ids.detach().cpu().numpy()
            )
        else:
            most_act_sample_ids = np.asarray(
                most_act_sample_ids
            )

        out_of_range = most_act_sample_ids >= len(feature_dataset)

        if np.any(out_of_range):
            invalid = most_act_sample_ids[out_of_range][:10].tolist()
            raise ValueError(
                f"CRP references for neuron {neuron} contain "
                f"indices outside the {split!r} split "
                f"(size {len(feature_dataset)}), e.g. {invalid}."
            )

        # Remove CRP padding values such as -1.
        most_act_sample_ids = most_act_sample_ids[
            most_act_sample_ids >= 0
        ].astype(np.int64)

        if len(most_act_sample_ids) == 0:
            raise RuntimeError(
                f"No valid reference images found for neuron {neuron}."
            )

        dataset_subset = torch.utils.data.Subset(
            feature_dataset,
            most_act_sample_ids.tolist(),
        )

        if model_name == "vilt":
            dataloader = DataLoader(
                dataset_subset,
                batch_size=batch_size,
                shuffle=False,
                # Keeping this at zero avoids processor/tokenizer
                # serialization issues in notebook environments.
                num_workers=0,
                pin_memory=device == "cuda",
                collate_fn=make_vilt_collate_fn(processor),
            )
        else:
            dataloader = DataLoader(
                dataset_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=device == "cuda",
            )

        for batch in dataloader:
            conditions = [
                {
                    attribution_layer_name: [int(neuron)],
                }
            ]

            if model_name == "vilt":
                inputs, _ = batch

                (
                    inputs,
                    pixel_values,
                    input_embeds,
                    additional_forward_kwargs,
                ) = prepare_vilt_inputs(
                    inputs=inputs,
                    model=model,
                    device=device,
                )

                attr = attribution(
                    (pixel_values, input_embeds),
                    conditions,
                    grad_composite,
                    record_layer=attribution_layer_names,
                    start_layer=attribution_layer_name,
                    init_rel=lambda activation: activation.clamp(
                        min=0
                    ),
                    exclude_parallel=False,
                    additional_forward_kwargs=(
                        additional_forward_kwargs
                    ),
                )

            else:
                x, *_ = batch
                x = x.to(
                    device,
                    non_blocking=device == "cuda",
                )
                x.requires_grad_(True)

                attr = attribution(
                    x,
                    conditions,
                    grad_composite,
                    record_layer=attribution_layer_names,
                    start_layer=attribution_layer_name,
                    init_rel=lambda activation: activation.clamp(
                        min=0
                    ),
                )

            previous_relevance = (
                attr.relevances[previous_layer_name]
                .detach()
                .cpu()
            )
            previous_activation = (
                attr.activations[previous_layer_name]
                .detach()
                .cpu()
            )

            lower_relevance = (
                previous_relevance * previous_activation
            )

            conditional_relevance = concept.attribute(
                lower_relevance,
                abs_norm=True,
            )

            cond_relevances[neuron].append(
                conditional_relevance.detach().cpu()
            )

            activation = (
                attr.activations[attribution_layer_name]
                .detach()
                .cpu()
                .clamp(min=0)
            )

            if model_name == "vit":
                if activation.ndim != 3:
                    raise ValueError(
                        "Expected ViT activation with shape "
                        "[batch, tokens, channels], but received "
                        f"{tuple(activation.shape)}."
                    )

                max_activations[neuron].append(
                    activation.amax(dim=1)
                )
                mean_activations[neuron].append(
                    activation.mean(dim=1)
                )

            elif model_name == "vilt":
                if activation.ndim != 3:
                    raise ValueError(
                        "Expected ViLT activation with shape "
                        "[batch, sequence, channels], but received "
                        f"{tuple(activation.shape)}."
                    )

                # ViLT concatenates:
                #     padded text tokens + image tokens
                text_length = inputs["input_ids"].shape[1]

                image_activation = activation[
                    :,
                    text_length:,
                    :,
                ]

                if image_activation.shape[1] == 0:
                    raise RuntimeError(
                        "No image tokens were found in the ViLT "
                        f"activation. Activation shape: "
                        f"{tuple(activation.shape)}; "
                        f"text length: {text_length}."
                    )

                # The first image token is the image CLS token.
                image_patch_activation = image_activation[:, 1:, :]

                if image_patch_activation.shape[1] == 0:
                    raise RuntimeError(
                        "ViLT activation contains no image-patch "
                        "tokens after removing the image CLS token."
                    )

                max_activations[neuron].append(
                    image_patch_activation.amax(dim=1)
                )
                mean_activations[neuron].append(
                    image_patch_activation.mean(dim=1)
                )

            else:
                if activation.ndim != 4:
                    raise ValueError(
                        "Expected CNN activation with shape "
                        "[batch, channels, height, width], but "
                        f"received {tuple(activation.shape)}."
                    )

                max_activations[neuron].append(
                    activation.amax(dim=(2, 3))
                )
                mean_activations[neuron].append(
                    activation.mean(dim=(2, 3))
                )

        if not max_activations[neuron]:
            raise RuntimeError(
                f"No activations were collected for neuron {neuron}."
            )

        if not cond_relevances[neuron]:
            raise RuntimeError(
                f"No conditional relevances were collected for "
                f"neuron {neuron}."
            )

        max_activations[neuron] = torch.cat(
            max_activations[neuron],
            dim=0,
        )
        mean_activations[neuron] = torch.cat(
            mean_activations[neuron],
            dim=0,
        )
        cond_relevances[neuron] = torch.cat(
            cond_relevances[neuron],
            dim=0,
        )

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

    output_directory = (
        f"results/global_features/{dataset_name}/{model_name}"
    )
    os.makedirs(output_directory, exist_ok=True)

    output_path = os.path.join(
        output_directory,
        f"latent_features_{requested_layer_name}_{split}.safetensors",
    )

    save_file(
        {
            "max_act": max_activations.contiguous(),
            "mean_act": mean_activations.contiguous(),
            "cond_rel": cond_relevances.contiguous(),
            "reference_lengths": cond_rel_lengths.contiguous(),
        },
        output_path,
    )

    print(f"Saved latent features to: {output_path}")


if __name__ == "__main__":
    args = get_args()

    with open(
        args.config_file,
        encoding="utf-8",
    ) as config_file:
        config_text = os.path.expandvars(
            config_file.read()
        )

    config = yaml.safe_load(config_text)

    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    batch_size = config.get("batch_size", 40)
    data_path = config.get("data_path")
    ckpt_path = config.get("ckpt_path")
    layer_name = (
        args.layer_name or config.get("layer_name")
    )
    crp_split = config.get("crp_split")

    splits = (
        [args.split]
        if args.split
        else config.get("splits", ["test"])
    )

    print(f"Data path: {data_path}")

    for split in splits:
        main(
            model_name=model_name,
            ckpt_path=ckpt_path,
            dataset_name=dataset_name,
            data_path=data_path,
            split=split,
            batch_size=batch_size,
            layer_name=layer_name,
            crp_split=crp_split,
        )
