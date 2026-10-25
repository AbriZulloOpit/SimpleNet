#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

datapath="../xray-garments-resized"
project="XRayGarments_Results"
group="simplenet_xray_garments"
trained_run="xray_garments_run"
test_run="xray_garments_test"
dataset_name="mvtec_xray-garments"

source_checkpoint="results/${project}/${group}/${trained_run}/models/0/${dataset_name}/ckpt.pth"
test_checkpoint_dir="results/${project}/${group}/${test_run}/models/0/${dataset_name}"

if [[ ! -f "$source_checkpoint" ]]; then
	echo "Checkpoint non trovato: $source_checkpoint" >&2
	exit 1
fi

mkdir -p "$test_checkpoint_dir"
cp "$source_checkpoint" "$test_checkpoint_dir/ckpt.pth"

python3 main.py \
	--gpu 0 \
	--seed 0 \
	--log_group "$group" \
	--log_project "$project" \
	--results_path results \
	--run_name "$test_run" \
	--test \
	--save_segmentation_images \
	net \
	-b wideresnet50 \
	-le layer2 \
	-le layer3 \
	--pretrain_embed_dimension 1536 \
	--target_embed_dimension 1536 \
	--patchsize 3 \
	--meta_epochs 5 \
	--embedding_size 256 \
	--gan_epochs 4 \
	--noise_std 0.015 \
	--dsc_hidden 1024 \
	--dsc_layers 2 \
	--dsc_margin .5 \
	--pre_proj 1 \
	dataset \
	--batch_size 2 \
	--resize 329 \
	--imagesize 288 \
	-d xray-garments \
	mvtec \
	"$datapath"