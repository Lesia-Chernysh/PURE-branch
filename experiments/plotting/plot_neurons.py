import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import umap
import zennit.image as zimage
import zennit.rules as z_rules

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
try:
    from crp.visualization import FeatureVisualization
except ImportError:
    # for ViLT
    try:
        from crp.visualization import ViLTFeatureVisualization
    except ImportError:
        # for Qwen
        from crp.visualization import QwenFeatureVisualization
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from safetensors import safe_open
from scipy import stats
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat, LayerMapComposite

from datasets import get_dataset
from models import (
    get_canonizer,
    get_fn_model_loader,
)
from utils.helper import (
    get_layer_names_model,
    load_config,
    reference_lengths,
    validate_layer_name,
)
from utils.lrp_composites import ReferenceEpsilonPlusFlat
from utils.render import vis_opaque_img_border


def get_parser():
    parser = ArgumentParser(
        description=(
            "Display reference images and latent clusters for "
            "selected model neurons."
        )
    )

    parser.add_argument(
        "--config_file",
        default="configs/imagenet/resnet50_timm.yaml",
    )
    parser.add_argument(
        "--neurons",
        default="1,2,3,4,5",
        help="Comma-separated neuron indices.",
    )
    parser.add_argument(
        "--embeddings",
        default="pure",
        choices=["pure", "CLIP", "activations"],
    )
    parser.add_argument(
        "--split",
        default=None,
    )
    parser.add_argument(
        "--layer_name",
        default=None,
    )
    parser.add_argument(
        "--num_clusters",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--n_refimgs",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--mode",
        default="activation",
        choices=["activation", "relevance"],
    )
    parser.add_argument(
        "--batch_size",
        default=40,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        default="results/neuron_plots",
    )

    args = parser.parse_args()

    config = load_config(args.config_file)
    config["config_name"] = os.path.basename(
        args.config_file
    )[:-5]

    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    return args


def safe_torch_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def get_dataset_preprocessing(dataset):
    return getattr(dataset, "preprocessing", None)


def to_display_image(image):
    """Convert PIL, NumPy, or tensor images to RGB PIL images."""

    if hasattr(image, "convert"):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        array = image

        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            array = np.transpose(array, (1, 2, 0))

        if np.issubdtype(array.dtype, np.floating):
            array = np.nan_to_num(array)

            if array.min() < 0 or array.max() > 1:
                minimum = array.min()
                maximum = array.max()

                if maximum > minimum:
                    array = (
                        array - minimum
                    ) / (
                        maximum - minimum
                    )
                else:
                    array = np.zeros_like(array)

            array = np.clip(
                array * 255,
                0,
                255,
            ).astype(np.uint8)

        return to_pil_image(array).convert("RGB")

    if torch.is_tensor(image):
        tensor = image.detach().cpu().float()

        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "Cannot convert a batch containing more than "
                    f"one image: {tuple(tensor.shape)}."
                )
            tensor = tensor[0]

        if tensor.ndim == 3 and tensor.shape[-1] in {1, 3, 4}:
            tensor = tensor.permute(2, 0, 1)

        tensor = torch.nan_to_num(tensor)

        if tensor.min() < 0 or tensor.max() > 1:
            minimum = tensor.min()
            maximum = tensor.max()

            if maximum > minimum:
                tensor = (
                    tensor - minimum
                ) / (
                    maximum - minimum
                )
            else:
                tensor = torch.zeros_like(tensor)

        return to_pil_image(
            tensor.clamp(0, 1)
        ).convert("RGB")

    raise TypeError(
        f"Unsupported image type: {type(image)}"
    )


def make_image_grid(
    images,
    size=(150, 150),
    maximum=8,
):
    images = [
        to_display_image(image)
        for image in images[:maximum]
    ]

    if not images:
        return np.full(
            (size[0], size[1], 3),
            255,
            dtype=np.uint8,
        )

    resize = torchvision.transforms.Resize(size)
    pil_to_tensor = torchvision.transforms.PILToTensor()

    tensors = [
        pil_to_tensor(resize(image))
        for image in images
    ]

    grid = make_grid(
        tensors,
        nrow=min(maximum, len(tensors)),
        padding=0,
    )

    return np.asarray(
        zimage.imgify(grid.detach().cpu())
    )


def load_latent_representations(
    dataset_name,
    model_name,
    layer_name,
    split,
    embedding_type,
    n_refimgs,
):
    base_path = (
        Path("results/global_features")
        / dataset_name
        / model_name
    )

    if embedding_type == "pure":
        file_path = (
            base_path
            / f"latent_features_{layer_name}_{split}.safetensors"
        )
        tensor_key = "cond_rel"

    elif embedding_type == "activations":
        file_path = (
            base_path
            / f"latent_features_{layer_name}_{split}.safetensors"
        )
        tensor_key = "mean_act"

    elif embedding_type == "CLIP":
        file_path = (
            base_path
            / f"latent_embeddings_{layer_name}_{split}.safetensors"
        )
        tensor_key = "CLIP"

    else:
        raise ValueError(
            f"Unknown embedding type: {embedding_type}"
        )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Latent feature file does not exist: {file_path}"
        )

    tensors = {}

    with safe_open(
        str(file_path),
        framework="pt",
        device="cpu",
    ) as file:
        for key in file.keys():
            tensors[key] = file.get_tensor(key)

    if tensor_key not in tensors:
        raise KeyError(
            f"{tensor_key!r} was not found in {file_path}. "
            f"Available keys: {list(tensors)}"
        )

    embeddings = tensors[tensor_key][:, :n_refimgs]
    lengths = reference_lengths(
        tensors,
        tensor_key,
    )

    non_finite = ~torch.isfinite(embeddings)

    if non_finite.any():
        count = non_finite.sum().item()
        raise FloatingPointError(
            f"{file_path} contains {count} non-finite values "
            f"in {tensor_key!r}. Recompute the latent features "
            "before plotting them."
        )

    return embeddings, lengths


def compute_umap(neuron_embeddings):
    values = (
        neuron_embeddings
        .detach()
        .cpu()
        .numpy()
    )

    n_samples = len(values)

    if n_samples < 3:
        raise ValueError(
            "At least three reference samples are required "
            "for UMAP."
        )

    reducer = umap.UMAP(
        n_neighbors=min(6, n_samples - 1),
        min_dist=0.3,
        spread=1.0,
        random_state=123,
    )

    return reducer.fit_transform(values)


def draw_density(axis, coordinates):
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    x_range = max(float(np.ptp(x)), 1e-5)
    y_range = max(float(np.ptp(y)), 1e-5)

    xmin = x.min() - 0.2 * x_range
    xmax = x.max() + 0.2 * x_range
    ymin = y.min() - 0.2 * y_range
    ymax = y.max() + 0.2 * y_range

    grid_x, grid_y = np.mgrid[
        xmin:xmax:100j,
        ymin:ymax:100j,
    ]

    try:
        positions = np.vstack([
            grid_x.ravel(),
            grid_y.ravel(),
        ])
        values = np.vstack([x, y])

        kernel = stats.gaussian_kde(
            values,
            0.5,
        )

        density = kernel(positions).reshape(
            grid_x.shape
        ).T

        axis.contour(
            density,
            extent=[xmin, xmax, ymin, ymax],
            cmap="Greys",
            alpha=0.3,
            extend="min",
            vmax=density.max(),
            zorder=0,
        )

    except (np.linalg.LinAlgError, ValueError):
        # KDE may fail for collinear or identical points.
        pass

    axis.scatter(
        x,
        y,
        alpha=0.7,
        color="black",
        s=10,
        zorder=1,
    )

    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)

    return x, y


def validate_modules(model, layer_names):
    available_modules = dict(model.named_modules())

    missing = [
        name
        for name in layer_names
        if name not in available_modules
    ]

    if missing:
        raise ValueError(
            "These requested attribution layers do not exist "
            "in the wrapped model:\n"
            + "\n".join(missing[:30])
        )


def prepare_model_and_visualization(
    args,
    dataset,
    model,
    layer_names,
    layer_name,
    fv_name,
    device,
):
    """Configure CRP attribution for the selected architecture."""

    model_name = args.model_name

    feature_dataset = dataset
    reference_layer_name = layer_name
    composite = None

    if model_name == "resnet50_transformers":
        concept = ChannelConcept()

        layer_map = {
            layer: concept
            for layer in layer_names
        }

        canonizers = get_canonizer(model_name)

        composite = ReferenceEpsilonPlusFlat(
            canonizers=canonizers,
            skip_layer=layer_name,
            epsilon=1e-6,
            zero_params=["bias"],
        )

        attribution = CondAttribution(model)

        fv = FeatureVisualization(
            attribution,
            dataset,
            layer_map,
            preprocess_fn=get_dataset_preprocessing(dataset),
            path=fv_name,
            max_target="sum",
            abs_norm=True,
        )

    elif model_name == "vgg16":
        concept = ChannelConcept()

        layer_map = {
            layer: concept
            for layer in layer_names
        }

        composite = ReferenceEpsilonPlusFlat(
            canonizers=[SequentialMergeBatchNorm()],
            skip_layer=layer_name,
        )

        attribution = CondAttribution(model)

        fv = FeatureVisualization(
            attribution,
            dataset,
            layer_map,
            preprocess_fn=get_dataset_preprocessing(dataset),
            path=fv_name,
            max_target="sum",
            abs_norm=True,
        )

    elif model_name == "vit":
        from crp.concepts import TransformerChannelConcept
        from crp.visualization import PredictionFeatureVisualization
        from lxt.efficient import (
            monkey_patch,
            monkey_patch_zennit,
        )
        from lxt.efficient.patches import (
            dropout_forward,
            layer_norm_forward,
            non_linear_forward,
            patch_method,
        )
        from timm.layers.attention import Attention
        from timm.layers.norm import LayerNorm as TimmLayerNorm

        from utils.attnlrp_timm_vit_conservative import (
            identity_forward,
            timm_attention_forward,
        )

        concept = TransformerChannelConcept()

        layer_map = {
            layer: concept
            for layer in layer_names
        }

        attribution = CondAttribution(
            concept,
            model,
        )

        if model.cls_token is not None:
            model.cls_token.requires_grad_(True)

        if model.pos_embed is not None:
            model.pos_embed.requires_grad_(True)

        for block in model.blocks:
            block.attn.fused_attn = False

        patch_map = {
            torch.nn.GELU: partial(
                patch_method,
                non_linear_forward,
                keep_original=True,
            ),
            torch.nn.Dropout: partial(
                patch_method,
                dropout_forward,
            ),
            TimmLayerNorm: partial(
                patch_method,
                layer_norm_forward,
            ),
            Attention: partial(
                patch_method,
                timm_attention_forward,
                keep_original=True,
            ),
            torch.nn.Identity: partial(
                patch_method,
                identity_forward,
                keep_original=True,
            ),
        }

        monkey_patch(
            model,
            patch_map=patch_map,
            verbose=True,
        )
        monkey_patch_zennit(verbose=True)

        composite = LayerMapComposite([
            (
                torch.nn.Conv2d,
                z_rules.Gamma(0.5),
            ),
            (
                torch.nn.Linear,
                z_rules.Gamma(10),
            ),
        ])

        predicted_targets = safe_torch_load(
            "vit_predicted_targets.pt"
        )

        fv = PredictionFeatureVisualization(
            concept,
            attribution,
            dataset,
            layer_map,
            preprocess_fn=None,
            path=fv_name,
            device=device,
            predicted_targets=predicted_targets,
        )

    elif model_name == "vilt":
        from crp.concepts import TransformerChannelConcept
        from crp.visualization import PredictionFeatureVisualization
        from lxt.explicit.models.vilt import attnlrp
        from transformers import ViltProcessor

        from datasets.imagenet_mini import (
            get_imagenetmini_for_vqa,
        )
        from models.vilt import ViltCRPWrapper

        predicted_targets = safe_torch_load(
            "vilt_predicted_targets.pt"
        )

        feature_dataset = get_imagenetmini_for_vqa(
            dataset=dataset,
            predicted_labels=predicted_targets,
        )

        processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )

        # Register AttnLRP before creating the CRP wrapper.
        attnlrp.register(model)

        crp_model = ViltCRPWrapper(model)
        crp_model = crp_model.to(device).float()
        crp_model.eval()

        concept = TransformerChannelConcept()

        wrapped_layer_names = [
            (
                name
                if name.startswith("model.")
                else f"model.{name}"
            )
            for name in layer_names
        ]

        reference_layer_name = (
            layer_name
            if layer_name.startswith("model.")
            else f"model.{layer_name}"
        )

        validate_modules(
            crp_model,
            wrapped_layer_names,
        )

        layer_map = {
            wrapped_layer: concept
            for wrapped_layer in wrapped_layer_names
        }

        attribution = CondAttribution(
            crp_model,
            device,
            no_param_grad=True,
        )

        # AttnLRP is already registered. Avoid registering a
        # second high-Gamma composite during visualization.
        composite = None

        fv = PredictionFeatureVisualization(
            attribution,
            feature_dataset,
            layer_map,
            processor,
            path=fv_name,
            device=device,
            max_target="sum",
            abs_norm=True,
            predicted_targets=predicted_targets,
        )

    else:
        # Generic CNN fallback.
        concept = ChannelConcept()

        layer_map = {
            layer: concept
            for layer in layer_names
        }

        canonizers = get_canonizer(model_name)

        composite = EpsilonPlusFlat(
            canonizers=canonizers,
        )

        attribution = CondAttribution(model)

        fv = FeatureVisualization(
            attribution,
            dataset,
            layer_map,
            preprocess_fn=get_dataset_preprocessing(dataset),
            path=fv_name,
            max_target="sum",
            abs_norm=True,
        )

    return {
        "fv": fv,
        "composite": composite,
        "feature_dataset": feature_dataset,
        "reference_layer_name": reference_layer_name,
    }


def get_reference_images(
    fv,
    model_name,
    neuron,
    layer_name,
    mode,
    n_valid,
    batch_size,
    composite,
):
    common_kwargs = {
        "concept_ids": [neuron],
        "layer_name": layer_name,
        "mode": mode,
        "r_range": (0, n_valid),
        "rf": True,
        "batch_size": min(batch_size, n_valid),
        "plot_fn": vis_opaque_img_border,
    }

    if model_name == "vilt":
        if layer_name.startswith("model"):
            layer_name = layer_name.replace("model.", "")
            common_kwargs["layer_name"] = layer_name
            
        references = fv.get_max_reference(
            **common_kwargs,
            attribute=True,
        )
    else:
        references = fv.get_max_reference(
            **common_kwargs,
            attribute=True,
            composite=composite,
        )

    if neuron not in references:
        raise KeyError(
            f"get_max_reference did not return neuron {neuron}. "
            f"Returned keys: {list(references)}"
        )

    neuron_references = references[neuron]

    if not isinstance(neuron_references, (list, tuple)):
        raise TypeError(
            "Expected rendered reference images, but "
            f"get_max_reference returned {type(neuron_references)}. "
            "This usually means attribution was not computed."
        )

    images = [
        to_display_image(image)
        for image in neuron_references[:n_valid]
    ]

    if len(images) != n_valid:
        raise RuntimeError(
            f"Expected {n_valid} reference images for neuron "
            f"{neuron}, but received {len(images)}."
        )

    return images


def main():
    args = get_parser()

    model_name = args.model_name
    dataset_name = args.dataset_name

    configured_splits = getattr(
        args,
        "splits",
        ["test"],
    )

    if args.split is None and len(configured_splits) != 1:
        raise ValueError(
            "The config contains multiple splits. Select one "
            "using --split."
        )

    split = args.split or configured_splits[0]
    crp_split = getattr(
        args,
        "crp_split",
        None,
    ) or split

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")

    dataset_arguments = {
        "data_path": args.data_path,
        "preprocessing": False,
        "split": split,
    }

    # ViLTProcessor expects untransformed PIL images.
    if model_name == "vilt":
        dataset_arguments["transform"] = False

    dataset = get_dataset(dataset_name)(
        **dataset_arguments
    )

    model = get_fn_model_loader(model_name)(
        n_class=dataset.num_classes,
        ckpt_path=args.ckpt_path,
    )
    model = model.to(device).float()
    model.eval()

    layer_names = get_layer_names_model(
        model,
        model_name,
    )

    if not layer_names:
        raise RuntimeError(
            f"No layers were returned for model {model_name!r}."
        )

    layer_name = validate_layer_name(
        args.layer_name or layer_names[-1],
        layer_names,
    )

    print(f"Selected artifact layer: {layer_name}")

    fv_name = (
        f"crp_files/"
        f"{model_name}_{dataset_name}_{crp_split}"
    )

    setup = prepare_model_and_visualization(
        args=args,
        dataset=dataset,
        model=model,
        layer_names=layer_names,
        layer_name=layer_name,
        fv_name=fv_name,
        device=device,
    )

    fv = setup["fv"]
    composite = setup["composite"]
    reference_layer_name = setup["reference_layer_name"]

    print(f"CRP reference layer: {reference_layer_name}")

    if (
        not os.path.isdir(fv.RelMax.PATH)
        or not os.listdir(fv.RelMax.PATH)
    ):
        raise FileNotFoundError(
            f"No CRP artifacts were found at {fv.RelMax.PATH}. "
            "Run CRP maximization with the same model, dataset, "
            f"and split ({split!r}) first."
        )

    embeddings, lengths = load_latent_representations(
        dataset_name=dataset_name,
        model_name=model_name,
        layer_name=layer_name,
        split=split,
        embedding_type=args.embeddings,
        n_refimgs=args.n_refimgs,
    )

    neurons = [
        int(value.strip())
        for value in args.neurons.split(",")
        if value.strip()
    ]

    if not neurons:
        raise ValueError(
            "No neuron indices were provided."
        )

    for neuron in neurons:
        if neuron < 0 or neuron >= embeddings.shape[0]:
            raise IndexError(
                f"Neuron {neuron} is outside [0, "
                f"{embeddings.shape[0] - 1}]."
            )

    n_clusters = args.num_clusters

    figure, axes = plt.subplots(
        2 + n_clusters,
        len(neurons),
        dpi=300,
        figsize=(
            max(4, len(neurons) * 3.1),
            2.5 + 1.8 * n_clusters,
        ),
        gridspec_kw={
            "height_ratios": [
                3,
                1,
                *np.ones(n_clusters).tolist(),
            ],
        },
        squeeze=False,
    )

    output_directory = (
        Path(args.output_dir)
        / dataset_name
        / model_name
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    virtual_neuron_directory = (
        Path("results/virtual_neurons")
        / dataset_name
        / model_name
        / layer_name
    )
    virtual_neuron_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for column, neuron in enumerate(neurons):
        n_valid = min(
            int(lengths[neuron]),
            args.n_refimgs,
        )

        minimum_references = max(
            n_clusters,
            3,
        )

        if n_valid < minimum_references:
            raise ValueError(
                f"Neuron {neuron} has only {n_valid} valid "
                f"references; at least {minimum_references} "
                "are required."
            )

        neuron_embeddings = embeddings[
            neuron,
            :n_valid,
        ]

        reference_images = get_reference_images(
            fv=fv,
            model_name=model_name,
            neuron=neuron,
            layer_name=reference_layer_name,
            mode=args.mode,
            n_valid=n_valid,
            batch_size=args.batch_size,
            composite=composite,
        )

        coordinates = compute_umap(
            neuron_embeddings
        )

        x, y = draw_density(
            axes[0, column],
            coordinates,
        )

        for index, image in enumerate(reference_images):
            image_box = OffsetImage(
                image.resize((100, 100)),
                zoom=0.15,
            )

            annotation = AnnotationBbox(
                image_box,
                (x[index], y[index]),
                frameon=True,
                pad=0,
            )

            axes[0, column].add_artist(annotation)

        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        axes[0, column].set_title(
            f"Neuron #{neuron}"
        )

        all_references_grid = make_image_grid(
            reference_images
        )

        axes[1, column].imshow(
            all_references_grid
        )
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])

        if column == 0:
            axes[1, column].set_ylabel("All")

        clusterer = KMeans(
            n_clusters=n_clusters,
            n_init=20,
            random_state=123,
        )

        cluster_labels = clusterer.fit_predict(
            neuron_embeddings.detach().cpu().numpy()
        )

        virtual_neurons = []

        for cluster_index in range(n_clusters):
            cluster_images = [
                image
                for index, image in enumerate(reference_images)
                if cluster_labels[index] == cluster_index
            ]

            virtual_neurons.append(cluster_images)

            cluster_grid = make_image_grid(
                cluster_images
            )

            axes[
                2 + cluster_index,
                column,
            ].imshow(cluster_grid)

            axes[
                2 + cluster_index,
                column,
            ].set_xticks([])

            axes[
                2 + cluster_index,
                column,
            ].set_yticks([])

            if column == 0:
                axes[
                    2 + cluster_index,
                    column,
                ].set_ylabel(
                    f"Cluster {cluster_index + 1}"
                )

        arrays_to_save = {
            f"img_{cluster_index}_{image_index}": np.asarray(
                image
            )
            for cluster_index, cluster_images in enumerate(
                virtual_neurons
            )
            for image_index, image in enumerate(
                cluster_images
            )
        }

        np.savez_compressed(
            virtual_neuron_directory
            / f"{layer_name}_{neuron}.npz",
            **arrays_to_save,
        )

        plt.imsave(
            output_directory
            / f"neuron_{neuron}_references.png",
            all_references_grid,
        )

    figure.tight_layout()

    neuron_string = "_".join(
        str(neuron)
        for neuron in neurons
    )

    output_path = (
        output_directory
        / f"neurons_{neuron_string}.pdf"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved plot to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
