"""Compute CRP reference embeddings for image classifiers and ViLT VQA.

The ViLT path analyses one fixed question across all images. Because ViLT is
multimodal, its question is part of the experiment and therefore also part of
the cache/output identity.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import load_maximization
# Different PURE/zennit-crp variants expose the visualizer under different
# names. Use one internal name throughout the rest of this script.
try:
    from crp.visualization import FeatureVisualization as CRPFeatureVisualization
except ImportError:
    try:
        from crp.visualization import (
            ViLTFeatureVisualization as CRPFeatureVisualization,
        )
    except ImportError as exc:
        import crp.visualization as _crp_visualization

        available = [
            name
            for name in dir(_crp_visualization)
            if "visual" in name.lower() or "feature" in name.lower()
        ]
        raise ImportError(
            "This crp installation provides neither FeatureVisualization nor "
            "ViLTFeatureVisualization. Available visualization symbols: "
            f"{available}"
        ) from exc
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPVisionModel,
    ViltForQuestionAnswering,
    ViltProcessor,
)
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_canonizer, get_fn_model_loader
from utils.helper import (
    CustomDataset,
    get_layer_names_model,
    pad_neuron_references,
    validate_layer_name,
)
from utils.lrp_composites import ReferenceEpsilonPlusFlat
from utils.render import crop_and_mask_images


DEFAULT_VILT_CHECKPOINT = "dandelin/vilt-b32-finetuned-vqa"
DEFAULT_VILT_QUESTION = "What is in the image?"
DEFAULT_VILT_LAYER = "vilt.encoder.layer.11.output.dense"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CRP reference embeddings for classifiers or ViLT VQA."
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
        help="Process one split instead of all splits in the config.",
    )
    parser.add_argument("--layer_name", type=str, default=None)
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Fixed question used for ViLT. Overrides the config value.",
    )
    return parser.parse_args()


def is_vilt_model(model_name: str, checkpoint: str | None = None) -> bool:
    values = f"{model_name} {checkpoint or ''}".lower()
    return "vilt" in values


def question_cache_key(question: str) -> str:
    readable = "_".join(question.lower().replace("?", "").split())[:48]
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


class ViltImageDataset(Dataset):
    """Turn an existing image dataset into ViLT pixel tensors.

    The wrapped dataset may return a PIL image, ``(image, target)``, or a
    dictionary containing an ``image`` field. The original target is retained
    when available. It is not used as a VQA answer target; predictions come
    from ViLT's answer-vocabulary logits.
    """

    def __init__(
        self,
        dataset: Dataset,
        processor: ViltProcessor,
        source_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        source_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.dataset = dataset
        self.image_processor = processor.image_processor
        self.preprocessing = lambda value: value
        self.num_classes = processor.image_processor.size
        self.source_mean = torch.tensor(source_mean, dtype=torch.float32).view(3, 1, 1)
        self.source_std = torch.tensor(source_std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _unpack(sample: Any) -> tuple[Any, Any]:
        if isinstance(sample, dict):
            image = sample["image"]
            target = sample.get("target", sample.get("label", 0))
            return image, target
        if isinstance(sample, (tuple, list)):
            image = sample[0]
            target = sample[1] if len(sample) > 1 else 0
            return image, target
        return sample, 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = self._unpack(self.dataset[index])
        image = self._prepare_for_vilt(image)
        # ``image`` is a floating-point tensor in [0, 1]. ViLT's processor
        # otherwise assumes tensor values are in [0, 255] and divides by 255
        # again, which would collapse the normalized result close to -1.
        encoded = self.image_processor(
            images=image,
            return_tensors="pt",
            do_rescale=False,
        )
        pixel_values = encoded["pixel_values"].squeeze(0)
        return pixel_values, torch.as_tensor(target, dtype=torch.long)

    def _prepare_for_vilt(self, image: Any) -> Any:
        """Convert an already ImageNet-normalized CHW tensor back to [0, 1].

        PURE datasets may apply their configured normalization even when
        instantiated with ``preprocessing=False``. Hugging Face image
        processors must receive PIL/uint8 data or floating point data in
        [0, 1], so undo that source normalization before ViLT preprocesses it.
        """

        if not isinstance(image, torch.Tensor):
            return image

        image = image.detach().cpu().float()
        if image.ndim != 3:
            raise ValueError(
                f"Expected a CHW image tensor, received shape {tuple(image.shape)}"
            )

        # Accept HWC tensors too, although PURE normally returns CHW.
        if image.shape[0] not in (1, 3, 4) and image.shape[-1] in (1, 3, 4):
            image = image.permute(2, 0, 1)

        if image.shape[0] == 4:
            image = image[:3]
        elif image.shape[0] == 1:
            image = image.expand(3, -1, -1)

        minimum = float(image.min())
        maximum = float(image.max())
        if minimum < 0.0 or maximum > 1.0:
            image = image * self.source_std + self.source_mean

        # Small interpolation/rounding excursions should not make the HF
        # processor reject an otherwise valid image.
        return image.clamp(0.0, 1.0)


class ViltVQAAdapter(nn.Module):
    """Present ViLT to CRP as a tensor-in/logits-out PyTorch model."""

    def __init__(
        self,
        model: ViltForQuestionAnswering,
        processor: ViltProcessor,
        question: str,
    ):
        super().__init__()
        self.vilt_vqa = model
        self.question = question

        tokens = processor.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        token_type_ids = tokens.get("token_type_ids", torch.zeros_like(input_ids))

        self.register_buffer("fixed_input_ids", input_ids, persistent=False)
        self.register_buffer("fixed_attention_mask", attention_mask, persistent=False)
        self.register_buffer("fixed_token_type_ids", token_type_ids, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        pixel_mask = torch.ones(
            (batch_size, height, width),
            dtype=torch.long,
            device=pixel_values.device,
        )
        outputs = self.vilt_vqa(
            input_ids=self.fixed_input_ids.expand(batch_size, -1),
            attention_mask=self.fixed_attention_mask.expand(batch_size, -1),
            token_type_ids=self.fixed_token_type_ids.expand(batch_size, -1),
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True,
        )
        return outputs.logits


def get_vilt_layer_names(model: nn.Module) -> list[str]:
    """Return dense layers whose output dimension is ViLT's hidden size."""

    hidden_size = model.vilt_vqa.config.hidden_size
    result = []
    for name, module in model.named_modules():
        if not name.startswith("vilt.encoder.layer."):
            continue
        if isinstance(module, nn.Linear) and module.out_features == hidden_size:
            result.append(name)
    return result


def resolve_vilt_layer(model: nn.Module, requested: str | None) -> tuple[str, list[str]]:
    available = get_vilt_layer_names(model)
    if not available:
        raise RuntimeError("No supported ViLT encoder layers were found.")

    requested = requested or DEFAULT_VILT_LAYER
    candidates = [
        requested,
        f"vilt_vqa.{requested}",
        f"vilt_vqa.vilt.{requested}" if requested.startswith("encoder.") else "",
    ]
    layer_name = next((name for name in candidates if name in available), None)
    if layer_name is None:
        options = "\n".join(available)
        raise ValueError(f"Unknown ViLT layer {requested!r}. Available layers:\n{options}")
    return layer_name, available


def build_standard_composite(model_name: str, layer_name: str):
    """Preserve the original composites for supported classifier families."""

    if model_name == "resnet50_transformers":
        return ReferenceEpsilonPlusFlat(
            canonizers=get_canonizer(model_name),
            skip_layer=layer_name,
            epsilon=1e-6,
            zero_params=["bias"],
        )
    if model_name == "vgg16":
        return EpsilonPlusFlat([SequentialMergeBatchNorm()])

    # Retain the original behavior for models that do not need a custom
    # reference composite. Add another explicit branch when a model family
    # requires its own canonizer.
    return None


def build_vilt_composite(model: nn.Module, layer_name: str, config: dict[str, Any]):
    """Build the project-provided transformer composite when configured.

    Attention LRP is implementation-specific. Setting ``vilt_use_composite``
    to false permits activation-only reference collection. Relevance-based
    receptive fields require a ViLT-compatible canonizer/composite.
    """

    if not config.get("vilt_use_composite", False):
        return None

    canonizer_name = config.get("vilt_canonizer", "vilt")
    canonizers = get_canonizer(canonizer_name)
    return ReferenceEpsilonPlusFlat(
        canonizers=canonizers,
        skip_layer=layer_name,
        epsilon=1e-6,
        zero_params=["bias"],
    )


def create_feature_visualization(
    attribution: CondAttribution,
    dataset: Dataset,
    layer_map: dict[str, ChannelConcept],
    preprocess_fn,
    path: str,
):
    # PURE's ViLTFeatureVisualization and upstream CRP's
    # FeatureVisualization have slightly different constructor signatures.
    # Pass only keyword arguments supported by the installed implementation.
    candidate_kwargs = {
        "preprocess_fn": preprocess_fn,
        "path": path,
        "max_target": "sum",
        "abs_norm": True,
    }
    signature = inspect.signature(CRPFeatureVisualization.__init__)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        supported_kwargs = candidate_kwargs
    else:
        supported_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in signature.parameters
        }

    omitted = sorted(set(candidate_kwargs) - set(supported_kwargs))
    if omitted:
        print(
            f"{CRPFeatureVisualization.__name__} does not accept and will omit: "
            f"{omitted}"
        )

    return CRPFeatureVisualization(
        attribution,
        dataset,
        layer_map,
        **supported_kwargs,
    )


def maybe_run_feature_collection(
    fv: CRPFeatureVisualization,
    config: dict[str, Any],
) -> None:
    """Optionally create ActMax files if requested by the configuration."""

    if not config.get("run_feature_collection", False):
        return

    batch_size = int(config.get("collection_batch_size", 16))
    start = int(config.get("collection_start", 0))
    end = config.get("collection_end")
    end = len(fv.dataset) if end is None else min(int(end), len(fv.dataset))

    if hasattr(fv, "run"):
        signature = inspect.signature(fv.run)
        kwargs = {}
        if "batch_size" in signature.parameters:
            kwargs["batch_size"] = batch_size
        fv.run(**kwargs)
        return

    if hasattr(fv, "run_distributed"):
        fv.run_distributed(
            composite=None,
            data_start=start,
            data_end=end,
            batch_size=batch_size,
            checkpoint=int(config.get("collection_checkpoint", 1000)),
            on_device=config.get("device", "cuda"),
        )
        return

    raise RuntimeError(
        "FeatureVisualization exposes neither run() nor run_distributed(). "
        "Generate the ActMax cache using the collection API provided by your CRP version."
    )


def extract_pixel_values(batch: Any) -> torch.Tensor:
    """Accept tensor or Hugging Face BatchFeature outputs from CustomDataset."""

    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        if "pixel_values" not in batch:
            raise KeyError(f"Expected pixel_values, received keys {list(batch)}")
        return batch["pixel_values"]
    if hasattr(batch, "pixel_values"):
        return batch.pixel_values
    raise TypeError(f"Unsupported processed image batch: {type(batch)!r}")


@torch.inference_mode()
def embed_reference_images(
    ref_imgs,
    processor,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    dataset = CustomDataset(ref_imgs, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    embeddings = []
    for batch in loader:
        pixel_values = extract_pixel_values(batch).to(device)
        output = model(pixel_values=pixel_values)
        embeddings.append(output.last_hidden_state.mean(dim=1).cpu())
    if not embeddings:
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        return torch.empty((0, hidden_size))
    return torch.cat(embeddings, dim=0)


def main(
    model_name: str,
    ckpt_path: str | None,
    dataset_name: str,
    data_path: str | None,
    split: str,
    layer_name: str | None,
    config: dict[str, Any],
    crp_split: str | None = None,
    question: str | None = None,
) -> None:
    nref = int(config.get("nref", 40))
    embedding_batch_size = int(config.get("embedding_batch_size", 10))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_vilt = is_vilt_model(model_name, ckpt_path)

    base_dataset = get_dataset(dataset_name)(
        data_path=data_path,
        preprocessing=False if use_vilt else config.get("preprocessing", False),
        split=split,
    )
    print(f"Dataset loaded: {len(base_dataset)} samples")

    if use_vilt:
        checkpoint = ckpt_path or DEFAULT_VILT_CHECKPOINT
        question = question or config.get("question", DEFAULT_VILT_QUESTION)
        processor = ViltProcessor.from_pretrained(checkpoint)
        raw_model = ViltForQuestionAnswering.from_pretrained(checkpoint)
        model = ViltVQAAdapter(raw_model, processor, question).to(device).eval()
        source_mean = tuple(
            float(value)
            for value in config.get("source_image_mean", [0.485, 0.456, 0.406])
        )
        source_std = tuple(
            float(value)
            for value in config.get("source_image_std", [0.229, 0.224, 0.225])
        )
        if len(source_mean) != 3 or len(source_std) != 3:
            raise ValueError("source_image_mean and source_image_std must have 3 values")
        dataset = ViltImageDataset(
            base_dataset,
            processor,
            source_mean=source_mean,
            source_std=source_std,
        )
        layer_name, layer_names = resolve_vilt_layer(model, layer_name)
        composite = build_vilt_composite(model, layer_name, config)
        preprocessing = dataset.preprocessing
        experiment_suffix = question_cache_key(question)

        image, _ = dataset[0]
        print(f"ViLT checkpoint: {checkpoint}")
        print(f"ViLT question: {question}")
        print(
            f"ViLT processed sample: shape={tuple(image.shape)}, "
            f"range=[{image.min().item():.4f}, {image.max().item():.4f}]"
        )
    else:
        model = get_fn_model_loader(model_name)(
            ckpt_path=ckpt_path,
            n_class=base_dataset.num_classes,
        ).to(device).eval()
        dataset = base_dataset
        layer_names = get_layer_names_model(model, model_name)
        layer_name = validate_layer_name(layer_name or layer_names[-1], layer_names)
        composite = build_standard_composite(model_name, layer_name)
        preprocessing = dataset.preprocessing
        experiment_suffix = "classifier"

    print(f"Model loaded on {device}")
    print(f"Analyzing layer: {layer_name}")

    attribution = CondAttribution(model)
    concept = ChannelConcept()
    layer_map = {name: concept for name in layer_names}

    crp_split = crp_split or split
    # Existing PURE ActMax caches use ``model_dataset_split``. Keep that as
    # the default lookup convention so previously collected references remain
    # usable. New experiments may opt into question-specific ActMax caches.
    cache_name = f"{model_name}_{dataset_name}_{crp_split}"
    if use_vilt and config.get("question_specific_actmax_cache", False):
        cache_name = f"{cache_name}_{experiment_suffix}"
    fv = create_feature_visualization(
        attribution=attribution,
        dataset=dataset,
        layer_map=layer_map,
        preprocess_fn=preprocessing,
        path=str(Path("crp_files") / cache_name),
    )
    maybe_run_feature_collection(fv, config)

    d_c_sorted, _, _ = load_maximization(fv.ActMax.PATH, layer_name)
    num_neurons = d_c_sorted.shape[1]
    print(f"Loaded ActMax cache for {num_neurons} components")

    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device).eval()
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    clip_embeddings = []
    dino_embeddings = []
    use_rf = bool(config.get("use_receptive_field", composite is not None))

    for neuron in tqdm(range(num_neurons), desc="Embedding references"):
        ref_imgs = fv.get_max_reference(
            [neuron],
            layer_name,
            "activation",
            (0, nref),
            composite=composite,
            rf=use_rf,
            batch_size=nref,
            plot_fn=crop_and_mask_images,
        )[neuron]
        clip_embeddings.append(
            embed_reference_images(
                ref_imgs,
                clip_processor,
                clip_model,
                device,
                embedding_batch_size,
            )
        )
        dino_embeddings.append(
            embed_reference_images(
                ref_imgs,
                dino_processor,
                dino_model,
                device,
                embedding_batch_size,
            )
        )

    clip_embeddings, clip_lengths = pad_neuron_references(clip_embeddings)
    dino_embeddings, dino_lengths = pad_neuron_references(dino_embeddings)
    if not torch.equal(clip_lengths, dino_lengths):
        raise RuntimeError("CLIP and DINO produced inconsistent reference counts.")

    output_dir = (
        Path("results")
        / "global_features"
        / dataset_name
        / model_name
        / experiment_suffix
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_layer_name = layer_name.replace("/", "_")
    output_file = output_dir / f"latent_embeddings_{safe_layer_name}_{split}.safetensors"
    save_file(
        {
            "CLIP": clip_embeddings.contiguous(),
            "DINO": dino_embeddings.contiguous(),
            "reference_lengths": clip_lengths.contiguous(),
        },
        str(output_file),
    )
    print(f"Saved embeddings to {output_file}")


if __name__ == "__main__":
    args = get_args()
    with open(args.config_file, encoding="utf-8") as file:
        config = yaml.safe_load(os.path.expandvars(file.read()))

    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    layer_name = args.layer_name or config.get("layer_name")
    data_path = config.get("data_path")
    ckpt_path = config.get("ckpt_path")
    crp_split = config.get("crp_split")
    question = args.question or config.get("question")
    splits = [args.split] if args.split else config.get("splits", ["test"])

    print(f"Splits: {splits}")
    print(f"CRP split: {crp_split}")
    for current_split in splits:
        main(
            model_name=model_name,
            ckpt_path=ckpt_path,
            dataset_name=dataset_name,
            data_path=data_path,
            split=current_split,
            layer_name=layer_name,
            config=config,
            crp_split=crp_split,
            question=question,
        )
