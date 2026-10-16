import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from semanticlens.scores import polysemanticity_score


IMAGE_RE = re.compile(r"^img_(\d+)_(\d+)$")
EMBEDDING_RE = re.compile(r"^embedding_(\d+)_(\d+)$")
NEURON_RE = re.compile(r"^(.*)_(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute SemanticLens polysemanticity scores from virtual-neuron "
            "NPZ files containing original images and precomputed embeddings."
        )
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--question",
        default=None,
        help="Fixed VQA question used to create ViLT artifacts.",
    )
    parser.add_argument(
        "--layer-name",
        default=None,
        help=(
            "Optional layer filter. For ViLT, names with or without the "
            "leading 'model.' prefix are equivalent."
        ),
    )
    return parser.parse_args()


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def canonical_layer_name(layer, model_name=None):
    layer = str(layer)
    if (model_name or "").lower() == "vilt" and layer.startswith("model."):
        return layer[len("model."):]
    return layer


def parse_identity(path, root, model_name=None):
    match = NEURON_RE.match(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse neuron ID from {path.name}")
    neuron = int(match.group(2))
    layer = path.parent.name if path.parent != root else match.group(1)
    return canonical_layer_name(layer, model_name), neuron


def load_virtual_clusters(path):
    images = defaultdict(dict)
    embeddings = defaultdict(dict)

    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            image_match = IMAGE_RE.match(key)
            if image_match:
                cluster_id, member_id = map(int, image_match.groups())
                images[cluster_id][member_id] = archive[key].copy()
                continue

            embedding_match = EMBEDDING_RE.match(key)
            if embedding_match:
                cluster_id, member_id = map(int, embedding_match.groups())
                embeddings[cluster_id][member_id] = archive[key].copy()

    result = {}
    for cluster_id in sorted(set(images) | set(embeddings)):
        image_ids = set(images[cluster_id])
        embedding_ids = set(embeddings[cluster_id])
        if image_ids != embedding_ids:
            raise ValueError(
                f"Image/embedding member mismatch in {path}, cluster "
                f"{cluster_id}: images={sorted(image_ids)}, "
                f"embeddings={sorted(embedding_ids)}"
            )

        member_ids = sorted(image_ids)
        result[cluster_id] = {
            "images": [images[cluster_id][i] for i in member_ids],
            "embeddings": torch.from_numpy(
                np.stack([embeddings[cluster_id][i] for i in member_ids])
            ).float(),
        }
    return result


def semanticlens_score(embeddings, n_clusters, random_state):
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected [samples, dimensions], got {tuple(embeddings.shape)}"
        )
    if embeddings.shape[0] < n_clusters:
        return math.nan
    if embeddings.shape[1] == 0:
        return math.nan
    if not torch.isfinite(embeddings).all():
        return math.nan

    # SemanticLens expects [components, samples, embedding dimensions].
    concept_db = embeddings.unsqueeze(0)
    score = polysemanticity_score(
        concept_db,
        replace_empty_clusters=True,
        random_state=random_state,
        n_clusters=n_clusters,
    )
    return float(score[0].cpu())


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_layer(rows):
    grouped = defaultdict(list)
    for row in rows:
        if math.isfinite(row["polysemanticity"]):
            grouped[row["layer"]].append(row["polysemanticity"])

    summaries = []
    for layer in sorted({row["layer"] for row in rows}, key=natural_key):
        values = np.asarray(grouped[layer], dtype=np.float64)
        summaries.append(
            {
                "layer": layer,
                "valid_virtual_neurons": len(values),
                "mean_polysemanticity": (
                    float(values.mean()) if len(values) else math.nan
                ),
                "median_polysemanticity": (
                    float(np.median(values)) if len(values) else math.nan
                ),
                "std_polysemanticity": (
                    float(values.std()) if len(values) else math.nan
                ),
            }
        )
    return summaries


def plot_layers(rows, summaries, output_path):
    layers = [row["layer"] for row in summaries]
    grouped = defaultdict(list)
    for row in rows:
        if math.isfinite(row["polysemanticity"]):
            grouped[row["layer"]].append(row["polysemanticity"])

    positions = np.arange(1, len(layers) + 1)
    values = [grouped[layer] for layer in layers]
    figure, axis = plt.subplots(
        figsize=(max(10, 0.55 * len(layers)), 6),
        constrained_layout=True,
    )

    nonempty = [(p, v) for p, v in zip(positions, values) if v]
    if nonempty:
        axis.boxplot(
            [v for _, v in nonempty],
            positions=[p for p, _ in nonempty],
            widths=0.6,
            showfliers=False,
        )

    rng = np.random.default_rng(123)
    for position, scores in zip(positions, values):
        if scores:
            axis.scatter(
                position + rng.normal(0, 0.05, len(scores)),
                scores,
                s=14,
                alpha=0.55,
            )

    means = [row["mean_polysemanticity"] for row in summaries]
    axis.plot(positions, means, color="black", marker="o", label="Layer mean")
    axis.set_xticks(positions)
    axis.set_xticklabels(layers, rotation=65, ha="right")
    axis.set_ylabel("SemanticLens polysemanticity")
    axis.set_xlabel("Layer")
    axis.set_title("Residual polysemanticity of PURE virtual neurons")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_model(rows, model_mean, output_path):
    valid = [
        row for row in rows if math.isfinite(row["polysemanticity"])
    ]
    if not valid:
        return

    scores = np.asarray(
        [row["polysemanticity"] for row in valid], dtype=np.float64
    )
    ranked = sorted(
        valid,
        key=lambda row: row["polysemanticity"],
        reverse=True,
    )[:20]
    ranked.reverse()

    figure, (histogram, ranking) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )
    histogram.hist(scores, bins=min(30, max(5, int(np.sqrt(len(scores))))))
    histogram.axvline(
        model_mean,
        color="tab:red",
        linestyle="--",
        label=f"Model mean = {model_mean:.4f}",
    )
    histogram.set_xlabel("SemanticLens polysemanticity")
    histogram.set_ylabel("Number of virtual neurons")
    histogram.set_title("Whole-model distribution")
    histogram.legend()

    labels = [
        f'{row["layer"]} / n{row["neuron"]} / c{row["cluster"]}'
        for row in ranked
    ]
    ranking.barh(labels, [row["polysemanticity"] for row in ranked])
    ranking.set_xlabel("SemanticLens polysemanticity")
    ranking.set_title("20 most polysemantic virtual neurons")
    ranking.grid(axis="x", alpha=0.25)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_root.rglob("*.npz"), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No NPZ files below {args.input_root}")

    requested_layer = (
        canonical_layer_name(args.layer_name, args.model_name)
        if args.layer_name
        else None
    )

    rows = []
    for index, path in enumerate(paths, start=1):
        layer, neuron = parse_identity(
            path,
            args.input_root,
            model_name=args.model_name,
        )
        if requested_layer is not None and layer != requested_layer:
            continue
        clusters = load_virtual_clusters(path)
        print(f"[{index}/{len(paths)}] {path}: {len(clusters)} clusters")

        for cluster_id, values in clusters.items():
            embeddings = values["embeddings"]
            score = semanticlens_score(
                embeddings,
                n_clusters=args.n_clusters,
                random_state=args.random_state,
            )
            rows.append(
                {
                    "file": str(path.relative_to(args.input_root)),
                    "model": args.model_name or "",
                    "checkpoint": args.checkpoint or "",
                    "question": args.question or "",
                    "layer": layer,
                    "neuron": neuron,
                    "cluster": cluster_id,
                    "image_count": len(values["images"]),
                    "embedding_dimension": embeddings.shape[-1],
                    "polysemanticity": score,
                }
            )

    if not rows:
        suffix = f" for layer {requested_layer!r}" if requested_layer else ""
        raise FileNotFoundError(
            f"No matching virtual-neuron records were found{suffix} below "
            f"{args.input_root}."
        )

    summaries = summarize_by_layer(rows)
    scores = np.asarray(
        [row["polysemanticity"] for row in rows if math.isfinite(row["polysemanticity"])],
        dtype=np.float64,
    )
    model_mean = float(scores.mean()) if len(scores) else math.nan
    model_summary = {
        "metric": "SemanticLens polysemanticity_score",
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "question": args.question,
        "requested_layer": requested_layer,
        "n_clusters": args.n_clusters,
        "random_state": args.random_state,
        "virtual_neuron_count": len(rows),
        "valid_virtual_neuron_count": len(scores),
        "layer_count": len(summaries),
        "mean_polysemanticity": model_mean,
        "median_polysemanticity": (
            float(np.median(scores)) if len(scores) else math.nan
        ),
        "std_polysemanticity": (
            float(scores.std()) if len(scores) else math.nan
        ),
    }

    write_csv(
        args.output_dir / "virtual_neuron_polysemanticity.csv",
        rows,
        [
            "file", "model", "checkpoint", "question", "layer",
            "neuron", "cluster", "image_count",
            "embedding_dimension", "polysemanticity",
        ],
    )
    write_csv(
        args.output_dir / "layer_polysemanticity.csv",
        summaries,
        [
            "layer", "valid_virtual_neurons", "mean_polysemanticity",
            "median_polysemanticity", "std_polysemanticity",
        ],
    )
    with (args.output_dir / "model_polysemanticity.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(model_summary, handle, indent=2, allow_nan=True)

    plot_layers(
        rows,
        summaries,
        args.output_dir / "polysemanticity_by_layer.png",
    )
    plot_model(
        rows,
        model_mean,
        args.output_dir / "polysemanticity_model.png",
    )
    print(json.dumps(model_summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
