
experiment=experiments.disentangling.eval_CLIP_alignment

for n_clusters in {2..5}; do
    python3 -m $experiment --config_file "configs/imagenet-mini/resnet50_transformers.yaml" --num_clusters $n_clusters
done
