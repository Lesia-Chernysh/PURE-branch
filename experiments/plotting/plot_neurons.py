import os
from argparse import ArgumentParser

import torchvision
from safetensors.torch import save_file

import umap
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import load_maximization
from crp.image import plot_grid
try:
    from crp.visualization import FeatureVisualization
except ImportError:
    # for ViLT
    try:
        from crp.visualization import ViLTFeatureVisualization
    except ImportError:
        # for Qwen
        from crp.visualization import QwenFeatureVisualization
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from safetensors import safe_open
from scipy import stats
from sklearn.cluster import KMeans
from torchvision.utils import make_grid
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm

from datasets import get_dataset
from models import get_canonizer, get_fn_model_loader
from utils.helper import (get_layer_names_model, load_config, reference_lengths,
                          validate_layer_name)
from utils.render import vis_opaque_img_border
from utils.lrp_composites import ReferenceEpsilonPlusFlat
import torch
import numpy as np
import matplotlib.pyplot as plt
import zennit.image as zimage

from timm.layers import maybe_add_mask, resolve_self_attn_mask
from timm.layers.attention import Attention
from timm.layers.norm import LayerNorm as TimmLayerNorm

from functools import partial

from lxt.efficient import monkey_patch, monkey_patch_zennit
from lxt.efficient.patches import (
    dropout_forward,
    layer_norm_forward,
    non_linear_forward,
    patch_method,
    stop_gradient,
)

from zennit.image import imgify
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

from utils.attnlrp_timm_vit_conservative import *

def get_parser():
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
    parser.add_argument('--mode', default='activation', type=str)
    args = parser.parse_args()

    config = load_config(args.config_file)
    config["config_name"] = os.path.basename(args.config_file)[:-5]

    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) is None:
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
mode = args.mode

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=False, split=SPLIT)

ckpt_path = args.ckpt_path

model = get_fn_model_loader(model_name)(n_class=dataset.num_classes, ckpt_path=ckpt_path).to(device)
model.eval()

layer_names = get_layer_names_model(model, model_name)

print(layer_names)
layer_name = args.layer_name or layer_names[-1]
layer_name = validate_layer_name(layer_name, layer_names)
           
if model_name == "resnet50_transformers":
    cc = ChannelConcept()
    layer_map = {layer: cc for layer in layer_names}
  
    canonizers = get_canonizer(model_name)
    composite = ReferenceEpsilonPlusFlat(
          canonizers=canonizers,
          skip_layer=layer_name,
          epsilon=1e-6,
          zero_params=["bias"],
      )
    attribution = CondAttribution(model)

    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                              path=fv_name, max_target="sum", abs_norm=True)
  
elif model_name == "vgg16":
      cc = ChannelConcept()
      layer_map = {layer: cc for layer in layer_names} 
  
      composite = ReferenceEpsilonPlusFlat(
          canonizers=[SequentialMergeBatchNorm()],
          skip_layer=layer_name,
      )
      attribution = CondAttribution(model)
  
      fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                                path=fv_name, max_target="sum", abs_norm=True)
elif model_name == "vit":
    from crp.concepts import TransformerChannelConcept
    from crp.visualization import PredictionFeatureVisualization

    tcc = TransformerChannelConcept()
    layer_map = {layer: tcc for layer in layer_names} 
  
    composite = ReferenceEpsilonPlusFlat(
          canonizers=[SequentialMergeBatchNorm()],
          skip_layer=layer_name,
    )
    attribution = CondAttribution(tcc, model)

    # modify ViT for AttnLRP
    # Enable these only so their residual relevance can be measured.
    if model.cls_token is not None:
            model.cls_token.requires_grad_(True)
    if model.pos_embed is not None:
            model.pos_embed.requires_grad_(True)
    
    for block in model.blocks:
            block.attn.fused_attn = False
    
    
    # apply monkey patching
    PATCH_MAP = {
        torch.nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
        torch.nn.Dropout: partial(patch_method, dropout_forward),
        # timm uses its own LayerNorm class
        TimmLayerNorm: partial(patch_method, layer_norm_forward),
        Attention: partial(
            patch_method, timm_attention_forward, keep_original=True
        ),
        torch.nn.Identity: partial(patch_method, identity_forward, keep_original=True),
    }
    
    monkey_patch(
        model,
        patch_map=PATCH_MAP,
        verbose=True,
    )
    
    monkey_patch_zennit(verbose=True)

    composite = LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Gamma(0.5)),
        (torch.nn.Linear, z_rules.Gamma(10)),
    ])

    predicted_targets = torch.load("vit_predicted_targets.pt")
  
    fv = PredictionFeatureVisualization(tcc, attribution, dataset, layer_map,
                                  preprocess_fn=None,
                                  path=fv_name, device=device,
                                  predicted_targets=predicted_targets)

elif model_name == "vilt":
    from crp.concepts import TransformerChannelConcept
    from crp.visualization import PredictionFeatureVisualization
    from lxt.explicit.models.vilt import attnlrp as lrp_rule_composite
    from transformers.activations import gelu, GELUActivation
    ################# LXT
    import lxt.explicit.modules as lm
    import lxt.explicit.functional as lf
    import lxt.explicit.rules as rules
    from lxt.explicit.core import Composite
    from zennit.composites import LayerMapComposite
    import zennit.rules as z_rules

    class GeLU(nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x):
            return gelu(x)

    lrp_rule_composite.register(vilt_vqa)

    crp_model = ViltCRPWrapper(vilt_vqa)

    tcc = TransformerChannelConcept()
    layer_map = {layer: tcc for layer in layer_names} 
  
    composite = Composite(
        {
            #nn.Linear: rules.EpsilonRule,
            nn.Tanh: rules.IdentityRule,
            GELUActivation: rules.IdentityRule,
            GeLU: rules.IdentityRule,
        },
        zennit_composite=LayerMapComposite([
            (nn.Conv2d, z_rules.Gamma(100)),
            (nn.Linear, z_rules.Gamma(0.05)),
        ])
    )

    attribution = CondAttribution(
        crp_model,
        device,
        no_param_grad=True
    )

    predicted_targets = torch.load("vilt_predicted_targets.pt")
  
    fv = PredictionFeatureVisualization(tcc, attribution, dataset, layer_map,
                                  preprocess_fn=None,
                                  path=fv_name, device=device,
                                  predicted_targets=predicted_targets)

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

        ref_c = fv.get_max_reference([neuron], layer_name, mode, (0, n_valid), composite=composite, plot_fn=None)
        plot_grid(ref_c, figsize=(6, 4))
        fig = plt.gcf()
      
        output_dir = f"results/neuron_plots/{dataset_name}/{model_name}"
        os.makedirs(output_dir, exist_ok=True)
      
        fig.savefig(
            f"{output_dir}/neuron_{neuron}_heatmaps.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
      
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
        print(labels)

        virtual_neurons = {neuron: []} 
      
        for lab in np.unique(labels):
            ref_imgs_cluster = [r for k, r in enumerate(ref_imgs_) if labels[k] == lab]
            virtual_neurons[neuron].append(ref_imgs_cluster)
          
            grid = make_grid(
                [resize(torch.from_numpy(np.asarray(k)).permute((2, 0, 1))) for k in ref_imgs_cluster[:NUM]],
                nrow=NUM,
                padding=0)
            grid = np.array(zimage.imgify(grid.detach().cpu()))
            axs[2 + lab][i].imshow(grid)
            axs[2 + lab][i].set_xticks([])
            axs[2 + lab][i].set_yticks([])
            axs[2 + lab][i].set_ylabel(f"{lab + 1}") if i == 0 else None

        imgs_as_arrays = [
            [np.array(img) for img in row]
            for row in virtual_neurons[neuron]
        ]

        path = f"results/virtual_neurons/{dataset_name}/{model_name}{layer_name}"
        os.makedirs(path, exist_ok=True)
        
        np.savez_compressed(
            f"{path}/{layer_name}_{neuron}.npz",
            **{
                f"img_{i}_{j}": arr
                for i, row in enumerate(imgs_as_arrays)
                for j, arr in enumerate(row)
            }
        )

    plt.tight_layout()

    path = f"results/neuron_plots/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/neurons_{'_'.join([str(n.item()) for n in neurons])}.pdf", dpi=300)

    plt.show()

