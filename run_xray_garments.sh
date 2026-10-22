#!/bin/bash

datapath=../xray-garments-simplenet

datasets=('xray-garments')

dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main.py \
net \
dataset \
	"$python" main.py \