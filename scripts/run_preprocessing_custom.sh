config="configs/imagenet-mini/resnet50_transformers.yaml"

# Keep split-local CRP indices separate. This reuses crp_files/*_train and *_val.
python3 -m experiments.preprocessing.compute_latent_features --config_file "$config"
python3 -m experiments.preprocessing.compute_embeddings --config_file "$config"
