from typing import Callable

from datasets.imagenet import get_imagenet
from datasets.imagenet_mini import get_imagenetmini, get_imagenetmini_for_vqa

DATASETS = {
    "imagenet": get_imagenet,
    "imagenet-mini": get_imagenetmini,
    "imagenetmini_for_vqa": get_imagenetmini_for_vqa
}


def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.

    """
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        print(f"Loading {dataset_name}")
        return dataset
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")
