#!/bin/bash

datasets=("eth_hotel" "eth_univ" "ucy_univ" "ucy_zara01" "ucy_zara02")

for dataset in ${datasets[@]}; do
  python3 ./data/preprocessor.py --dataset $dataset --data_path ./data/ --skip 10 --frame_stride 2
done
