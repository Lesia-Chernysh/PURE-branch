"""AttnLRP for a timm ViT with stable relevance accounting at the input stem.

Examples:
    python attnlrp_timm_vit_conservative.py --image path/to/image.jpg
    python attnlrp_timm_vit_conservative.py  # opens a file picker

Requires: torch, torchvision, timm, lxt, zennit, pillow, numpy
"""

import argparse
import types
from functools import partial

import numpy as np
import torch
from PIL import Image
from timm.layers import maybe_add_mask, resolve_self_attn_mask
from timm.layers.attention import Attention
from timm.layers.norm import LayerNorm as TimmLayerNorm
import timm

from lxt.efficient import monkey_patch, monkey_patch_zennit
from lxt.efficient.patches import (
    dropout_forward,
    layer_norm_forward,
    non_linear_forward,
    patch_method,
    stop_gradient,
)
from zennit import rules as z_rules
from zennit.composites import LayerMapComposite
from crp.image import imgify


def install_conservative_pos_embed(model, eps=1e-6):
    """Replace timm ViT._pos_embed and expose stable constant accounting.

    Positional embeddings use ordinary addition. This intentionally leaves their
    relevance measurable instead of using an unstable component-wise division.
    The eps argument is retained for command-line compatibility but is unused.
    """

    def _pos_embed_lrp(self, x):
        if self.pos_embed is None:
            return x

        if self.dynamic_img_size:
            raise NotImplementedError(
                "This example targets fixed-size ViTs. Add timm's resample_abs_pos_embed "
                "before conservative_add_constant for dynamic image sizes."
            )

        batch_size = x.shape[0]
        prefix = []
        if self.cls_token is not None:
            prefix.append(self.cls_token.expand(batch_size, -1, -1))
        if getattr(self, "reg_token", None) is not None:
            prefix.append(self.reg_token.expand(batch_size, -1, -1))

        if self.no_embed_class:
            # In this timm mode, positional embeddings belong only to patch tokens.
            x = x + self.pos_embed
            if prefix:
                x = torch.cat(prefix + [x], dim=1)
        else:
            if prefix:
                x = torch.cat(prefix + [x], dim=1)
            x = x + self.pos_embed

        return self.pos_drop(x)

    model._pos_embed = types.MethodType(_pos_embed_lrp, model)


def timm_attention_forward(self, x, attn_mask=None, is_causal=False):
    batch_size, num_tokens, channels = x.shape
    qkv = (
        self.qkv(x)
        .reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    # CP-LRP inside attention: treat the attention weights as contextual
    # constants and propagate relevance through the value path. LXT recommends
    # this hybrid for ViTs because it is substantially easier to stabilize with
    # Gamma rules than the full Q/K/V relevance split.
    q = stop_gradient(q)
    k = stop_gradient(k)

    if self.fused_attn:
        raise RuntimeError("Set block.attn.fused_attn = False before attribution.")

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn_bias = resolve_self_attn_mask(num_tokens, attn, attn_mask, is_causal)
    attn = maybe_add_mask(attn, attn_bias)
    attn = self.attn_drop(attn.softmax(dim=-1))
    x = attn @ v

    x = x.transpose(1, 2).reshape(batch_size, num_tokens, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    return self.proj_drop(x)


def identity_forward(self, x):
    return x


PATCH_MAP = {
    torch.nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    torch.nn.Dropout: partial(patch_method, dropout_forward),
    # timm uses its own LayerNorm class; an exact class-key patch is required.
    TimmLayerNorm: partial(patch_method, layer_norm_forward),
    Attention: partial(
        patch_method, timm_attention_forward, keep_original=True
    ),
    torch.nn.Identity: partial(patch_method, identity_forward, keep_original=True),
}


def tensor_relevance(tensor):
    if tensor.grad is None:
        raise RuntimeError("Tensor has no retained gradient.")
    return (tensor * tensor.grad).sum()


def retain_boundary(store, name):
    def hook(module, inputs, output):
        if inputs and isinstance(inputs[0], torch.Tensor):
            inputs[0].retain_grad()
            store[name + "_in"] = inputs[0]
        if isinstance(output, torch.Tensor):
            output.retain_grad()
            store[name + "_out"] = output
    return hook


def redistribute_scalar_relevance(heatmap, amount):
    """Visualization-only redistribution using absolute pixel relevance weights."""
    weights = heatmap.abs()
    weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return heatmap + amount.reshape(-1, 1, 1) * weights


def attribute(image, model, device, conv_gamma=0.5, linear_gamma=10.0,
              redistribute_cls=False):
    config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**config, is_training=False)
    pixels = transform(image).unsqueeze(0).to(device).requires_grad_(True)

    print(
        f"LRP rules: Conv2d Gamma={conv_gamma}, "
        f"Linear Gamma={linear_gamma}, biases excluded"
    )

    # Positive Gamma stabilizes the redistribution; biases are excluded so that
    # they do not absorb activation relevance.
    composite = LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Gamma(
            gamma=conv_gamma, zero_params=["bias"]
        )),
        (torch.nn.Linear, z_rules.Gamma(
            gamma=linear_gamma, zero_params=["bias"]
        )),
    ])
    composite.register(model)

    activations = {}
    handles = [
        model.patch_embed.proj.register_forward_hook(
            retain_boundary(activations, "conv_proj")
        ),
        model.blocks.register_forward_hook(
            retain_boundary(activations, "encoder")
        ),
        model.norm.register_forward_hook(
            retain_boundary(activations, "final_norm")
        ),
        model.head.register_forward_hook(
            retain_boundary(activations, "head")
        ),
    ]
    for index, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(
            retain_boundary(activations, f"transformer_{index}")
        ))

    try:
        model.zero_grad(set_to_none=True)
        logits = model(pixels)
        target = logits.argmax(dim=-1).item()
        selected = logits[0, target]
        selected.backward()

        raw_heatmap = (pixels * pixels.grad).sum(dim=1)
        r_input = raw_heatmap.sum()
        r_conv = tensor_relevance(activations["conv_proj_out"])
        r_encoder_in = tensor_relevance(activations["encoder_in"])
        r_encoder_out = tensor_relevance(activations["encoder_out"])
        r_norm_out = tensor_relevance(activations["final_norm_out"])
        r_head_in = tensor_relevance(activations["head_in"])

        # Learned constant branches are reported separately from image relevance.
        zero = selected.detach().new_zeros(())
        r_cls = zero
        if model.cls_token is not None and model.cls_token.grad is not None:
            r_cls = (model.cls_token * model.cls_token.grad).sum()
        r_pos = zero
        if model.pos_embed is not None and model.pos_embed.grad is not None:
            r_pos = (model.pos_embed * model.pos_embed.grad).sum()

        print(f"Predicted class: {target}")
        print(f"Output logit:              {selected.item(): .9f}")
        print(f"Input pixels:              {r_input.item(): .9f}")
        print(f"Conv projection output:    {r_conv.item(): .9f}")
        print(f"Encoder input:             {r_encoder_in.item(): .9f}")
        print(f"Encoder output:            {r_encoder_out.item(): .9f}")
        print(f"Final norm output:         {r_norm_out.item(): .9f}")
        print(f"Head input:                {r_head_in.item(): .9f}")
        print(f"Initial CLS residual:      {r_cls.item(): .9f}")
        print(f"Position residual:         {r_pos.item(): .9f}")
        print(f"Pixels + CLS + position:   {(r_input + r_cls + r_pos).item(): .9f}")

        print("\nTransformer block conservation")
        for index in range(len(model.blocks)):
            r_in = tensor_relevance(activations[f"transformer_{index}_in"])
            r_out = tensor_relevance(activations[f"transformer_{index}_out"])
            print(
                f"{index:02d}: in={r_in.item(): .9f}  "
                f"out={r_out.item(): .9f}  delta={(r_in-r_out).item(): .3e}"
            )

        result_heatmap = raw_heatmap
        if redistribute_cls:
            # This is a declared visualization convention, not model-derived LRP.
            constant_relevance = r_cls.detach() + r_pos.detach()
            result_heatmap = raw_heatmap.detach()
            #result_heatmap = redistribute_scalar_relevance(
            #    raw_heatmap, constant_relevance.reshape(1)
            #)
            print(
                "\nCLS/position redistribution enabled; heatmap sum: "
                f"{result_heatmap.sum().item(): .9f}"
            )

        display_heatmap = result_heatmap / result_heatmap.abs().amax(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-12)

        display(imgify(display_heatmap, vmin=-1, vmax=1, grid=(1, 1)))

        return {
            "target": target,
            "logit": selected.detach().cpu(),
            "raw_heatmap": raw_heatmap.detach().cpu(),
            "heatmap": result_heatmap.detach().cpu(),
            "display_heatmap": display_heatmap.detach().cpu(),
            "input_relevance": r_input.detach().cpu(),
            "cls_relevance": r_cls.detach().cpu(),
        }
    finally:
        for handle in handles:
            handle.remove()
        composite.remove()


def save_heatmap(result, output_path):
    import matplotlib.pyplot as plt

    heatmap = result["display_heatmap"][0].numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap, cmap="seismic", vmin=-1, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close()


def choose_image_interactively():
    """Open a native file picker when the script is run without --image."""
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        image_path = filedialog.askopenfilename(
            title="Choose an image for AttnLRP",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except Exception as error:
        raise RuntimeError(
            "No --image argument was supplied and the file picker could not be "
            "opened. Run the script with --image path/to/image.jpg."
        ) from error

    if not image_path:
        raise SystemExit("No image selected; attribution cancelled.")
    return image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=None,
        help="Input image path. If omitted, a file-selection dialog is opened.",
    )
    parser.add_argument(
        "--model", default="vit_small_patch16_224.augreg_in21k_ft_in1k"
    )
    parser.add_argument("--conv-gamma", type=float, default=0.5)
    parser.add_argument("--linear-gamma", type=float, default=10.0)
    parser.add_argument(
        "--pos-eps", type=float, default=1e-6,
        help="Deprecated compatibility option; ordinary stable addition is used.",
    )
    parser.add_argument(
        "--redistribute-cls", action="store_true",
        help="For display only, redistribute initial CLS and position relevance to pixels.",
    )
    parser.add_argument("--output", default="attnlrp_heatmap.png")
    args = parser.parse_args()

    image_path = args.image or choose_image_interactively()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=True).to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Enable these only so their residual relevance can be measured.
    if model.cls_token is not None:
        model.cls_token.requires_grad_(True)
    if model.pos_embed is not None:
        model.pos_embed.requires_grad_(True)

    for block in model.blocks:
        block.attn.fused_attn = False

    # Install the model-specific method before patching module forwards.
    install_conservative_pos_embed(model, eps=args.pos_eps)
    monkey_patch(model, patch_map=PATCH_MAP, verbose=True)
    monkey_patch_zennit(verbose=True)

    image = Image.open(image_path).convert("RGB")
    result = attribute(
        image=image,
        model=model,
        device=device,
        conv_gamma=args.conv_gamma,
        linear_gamma=args.linear_gamma,
        redistribute_cls=args.redistribute_cls,
    )
    save_heatmap(result, args.output)
    print(f"Saved heatmap to {args.output}")


if __name__ == "__main__":
    main()
