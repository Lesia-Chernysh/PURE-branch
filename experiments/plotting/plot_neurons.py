import os
from argparse import ArgumentParser
from typing import List

import torchvision
import umap
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import load_maximization

from crp.visualization import FeatureVisualization
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from safetensors import safe_open
from scipy import stats
from sklearn.cluster import KMeans
from torchvision.utils import make_grid
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_canonizer, get_fn_model_loader
from utils.helper import get_layer_names_model, reference_lengths, validate_layer_name
from utils.render import vis_opaque_img_border
import torch
import numpy as np
import matplotlib.pyplot as plt
import zennit.image as zimage


def get_parser(fixed_arguments: List[str] = []):
    parser = ArgumentParser(
        description='Compute and display the top-k most relevant neurons for a given data sample/prediction.', )

    parser.add_argument('--config_file',
                        default="configs/imagenet/resnet50_timm.yaml")
    parser.add_argument('--neurons',
                        default="1,2,3,4,5")
    parser.add_argument('--embeddings',
                        default="pure") # "pure", "CLIP", "activations"
    parser.add_argument('--split', default=None)
    parser.add_argument('--layer_name', default=None)
    parser.add_argument('--num_clusters', default=2, type=int)
    parser.add_argument('--n_refimgs', default=20, type=int)
    args = parser.parse_args()

    with open(parser.parse_args().config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["config_name"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    for k, v in config.items():
        if k not in fixed_arguments:
            setattr(args, k, v)

    return args


args = get_parser()
model_name = args.model_name
dataset_name = args.dataset_name

configured_splits = getattr(args, "splits", ["test"])
if args.split is None and len(configured_splits) != 1:
    raise ValueError("The config contains multiple splits; select one with --split.")
SPLIT = args.split or configured_splits[0]

crp_split = getattr(args, "crp_split", None) or SPLIT
fv_name = f"crp_files/{model_name}_{dataset_name}_{crp_split}"
batch_size = 100
n_refimgs = args.n_refimgs
mode = "activation"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=False, split=SPLIT)

ckpt_path = args.ckpt_path

model = get_fn_model_loader(model_name)(n_class=dataset.num_classes, ckpt_path=ckpt_path).to(device)
model.eval()
canonizers = get_canonizer(model_name)
composite = EpsilonPlusFlat(canonizers)
cc = ChannelConcept()

layer_names = get_layer_names_model(model, model_name)
layer_map = {layer: cc for layer in layer_names}

print(layer_names)
layer_name = args.layer_name or layer_names[-1]
layer_name = validate_layer_name(layer_name, layer_names)

attribution = CondAttribution(model)

fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                          path=fv_name, max_target="max", abs_norm=False)

if not os.path.isdir(fv.RelMax.PATH) or not os.listdir(fv.RelMax.PATH):
    raise FileNotFoundError(
        f"No CRP artifacts found at {fv.RelMax.PATH}. Run crp_run for the "
        f"same dataset and split ({SPLIT!r}) before plotting."
    )


d_c_sorted, a, rf_c_sorted = load_maximization(fv.ActMax.PATH, layer_name)


tensors = {}
path = f"results/global_features/{dataset_name}/{model_name}"

if args.embeddings == "pure":
    with safe_open(f"{path}/latent_features_{layer_name}_{SPLIT}.safetensors", framework="pt", device="cpu") as f:
       for key in f.keys():
           tensors[key] = f.get_tensor(key)
    embeddings = tensors["cond_rel"][:, :n_refimgs]
    lengths = reference_lengths(tensors, "cond_rel")
elif args.embeddings == "CLIP":
    with safe_open(f"{path}/latent_embeddings_{layer_name}_{SPLIT}.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    embeddings = tensors["CLIP"][:, :n_refimgs]
    lengths = reference_lengths(tensors, "CLIP")
elif args.embeddings == "activations":
    with safe_open(f"{path}/latent_features_{layer_name}_{SPLIT}.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    embeddings = tensors["mean_act"][:, :n_refimgs]
    lengths = reference_lengths(tensors, "mean_act")
else:
    raise ValueError("--embeddings must be one of: pure, CLIP, activations")


n_clusters = args.num_clusters

neurons = torch.tensor([int(i) for i in args.neurons.split(",")])

for inds_ in [neurons]:
    fig, axs = plt.subplots(2 + n_clusters, len(neurons), dpi=300, figsize=(len(neurons) * 4/1.3, 6/1.4),
                            gridspec_kw={'height_ratios': [len(neurons), 1, *np.ones(n_clusters).tolist()]},
                            squeeze=False)
    for i, inds in enumerate(inds_):
        print(i)
        neuron = inds.item()
        if neuron < 0 or neuron >= embeddings.shape[0]:
            raise IndexError(f"Neuron {neuron} is outside [0, {embeddings.shape[0] - 1}].")
        n_valid = min(int(lengths[neuron]), n_refimgs)
        if n_valid < max(n_clusters, 3):
            raise ValueError(
                f"Neuron {neuron} has only {n_valid} valid references; "
                f"need at least {max(n_clusters, 3)}."
            )
        neuron_embeddings = embeddings[neuron, :n_valid]
        ref_imgs = fv.get_max_reference([neuron], layer_name, mode, (0, n_valid),
                                        composite=composite, rf=True, batch_size=n_valid,
                                        plot_fn=vis_opaque_img_border)

        embedding = umap.UMAP(n_neighbors=min(6, n_valid - 1), min_dist=0.3, spread=1.0)
        X = embedding.fit_transform(neuron_embeddings)
        x, y = X[:, 0], X[:, 1]
        xmin = x.min() - 0.2 * (x.max() - x.min())
        xmax = x.max() + 0.2 * (x.max() - x.min())
        ymin = y.min() - 0.2 * (y.max() - y.min())
        ymax = y.max() + 0.2 * (y.max() - y.min())
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values, 0.5)
        Z = np.reshape(kernel(positions).T, X.shape).T
        axs[0][i].contour(Z, extent=[xmin, xmax, ymin, ymax], cmap="Greys", alpha=0.3, extend='min', vmax=Z.max() * 1, zorder=0)
        axs[0][i].scatter(x, y, alpha=0.7, c="black", s=10)

        for j, img_ in enumerate(ref_imgs[neuron][:n_valid]):
            imagebox = OffsetImage(img_.resize((100, 100)), zoom=0.15)
            ab = AnnotationBbox(imagebox, (x[j], y[j]), frameon=True, pad=0)
            axs[0][i].add_artist(ab)

        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[0][i].text(0.02, 0.98, f'#{inds.item()}', ha='left', va='top', transform=axs[0][i].transAxes)

        resize = torchvision.transforms.Resize((150, 150))

        NUM = 8
        ref_imgs_ = ref_imgs[neuron][:n_valid]
        grid = make_grid(
            [resize(torch.from_numpy(np.asarray(k)).permute((2, 0, 1))) for k in ref_imgs_[:NUM]],
            nrow=NUM,
            padding=0)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        axs[1][i].imshow(grid)
        axs[1][i].set_xticks([])
        axs[1][i].set_yticks([])
        axs[1][i].set_ylabel("all") if i == 0 else None



        cluster = KMeans(n_clusters=n_clusters, n_init=20, random_state=123).fit(neuron_embeddings)
        labels = np.array(cluster.labels_)
        for lab in np.unique(labels):
            ref_imgs_cluster = [r for k, r in enumerate(ref_imgs_) if labels[k] == lab]
            grid = make_grid(
                [resize(torch.from_numpy(np.asarray(k)).permute((2, 0, 1))) for k in ref_imgs_cluster[:NUM]],
                nrow=NUM,
                padding=0)
            grid = np.array(zimage.imgify(grid.detach().cpu()))
            axs[2 + lab][i].imshow(grid)
            axs[2 + lab][i].set_xticks([])
            axs[2 + lab][i].set_yticks([])
            axs[2 + lab][i].set_ylabel(f"{lab + 1}") if i == 0 else None

    plt.tight_layout()

    path = f"results/neuron_plots/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/neurons_{'_'.join([str(n.item()) for n in neurons])}.pdf", dpi=300)

    plt.show()

