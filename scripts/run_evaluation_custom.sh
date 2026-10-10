
experiment=experiments.disentangling.eval_CLIP_alignment
config="configs/imagenet-mini/resnet50_transformers.yaml"
layer="layer4.1.conv2"

for split in all; do
  for n_clusters in {2..5}; do
    python3 -m $experiment --config_file "$config" --split "$split" --layer_name "$layer" --num_clusters "$n_clusters"
  done
done
