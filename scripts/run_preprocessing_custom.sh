python3 -m experiments.preprocessing.compute_latent_features --config_file "configs/imagenet-mini/resnet50_transformers.yaml" --split "train" --layer_name "layer4.1.conv2"
python3 -m experiments.preprocessing.compute_embeddings --config_file "configs/imagenet-mini/resnet50_transformers.yaml" --split "train" --layer_name "layer4.1.conv2"
