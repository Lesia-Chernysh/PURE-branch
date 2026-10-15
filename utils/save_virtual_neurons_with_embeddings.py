"""Drop-in helpers for plot_neurons.py.

The embedding rows must be in the same order as reference_images.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from crp.helper import load_maximization


def get_reference_indices(fv, layer_name, neuron, mode, count):
    maximization_path = fv.RelMax.PATH if mode == "relevance" else fv.ActMax.PATH
    sorted_indices, _, _ = load_maximization(maximization_path, layer_name)
    indices = sorted_indices[:count, neuron]
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().numpy()
    indices = np.asarray(indices, dtype=np.int64)
    return indices[indices >= 0]


def load_original_images(dataset, indices):
    """Load RGB source images, bypassing model transforms when possible."""
    base_dataset = getattr(dataset, "dataset", dataset)
    images = []
    for index in indices:
        if hasattr(base_dataset, "samples"):
            image_path = base_dataset.samples[int(index)][0]
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())
        else:
            image = base_dataset[int(index)][0]
            if torch.is_tensor(image):
                raise TypeError(
                    "This dataset exposes only transformed tensors. Add a method "
                    "that loads its original source image, or store sample paths."
                )
            images.append(image.convert("RGB"))
    return images


def save_virtual_neuron_npz(
    output_path,
    reference_images,
    reference_embeddings,
    cluster_labels,
    reference_indices=None,
):
    if torch.is_tensor(reference_embeddings):
        reference_embeddings = reference_embeddings.detach().cpu().float().numpy()
    else:
        reference_embeddings = np.asarray(reference_embeddings, dtype=np.float32)

    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    if len(reference_images) != len(reference_embeddings):
        raise ValueError(
            f"Got {len(reference_images)} images but "
            f"{len(reference_embeddings)} embeddings."
        )
    if len(reference_images) != len(cluster_labels):
        raise ValueError(
            f"Got {len(reference_images)} images but "
            f"{len(cluster_labels)} cluster labels."
        )

    payload = {
        "cluster_labels": cluster_labels,
        "all_embeddings": reference_embeddings,
    }
    if reference_indices is not None:
        payload["reference_indices"] = np.asarray(reference_indices, dtype=np.int64)

    for cluster_id in sorted(np.unique(cluster_labels)):
        member_positions = np.flatnonzero(cluster_labels == cluster_id)
        for member_id, position in enumerate(member_positions):
            payload[f"img_{cluster_id}_{member_id}"] = np.asarray(
                reference_images[position]
            ).copy()
            payload[f"embedding_{cluster_id}_{member_id}"] = (
                reference_embeddings[position].copy()
            )
            payload[f"position_{cluster_id}_{member_id}"] = np.asarray(
                position, dtype=np.int64
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


# Replace the existing arrays_to_save / np.savez_compressed block with:
#
# reference_indices = get_reference_indices(
#     fv, reference_layer_name, neuron, args.mode, n_valid
# )
# original_reference_images = load_original_images(
#     feature_dataset, reference_indices
# )
#
# save_virtual_neuron_npz(
#     output_path=virtual_directory / f"{layer_name}_{neuron}.npz",
#     reference_images=original_reference_images,
#     reference_embeddings=neuron_embeddings,
#     cluster_labels=cluster_labels,
#     reference_indices=reference_indices,  # optional but recommended
# )
