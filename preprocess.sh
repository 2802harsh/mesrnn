#!/bin/bash

# datasets=("eth_hotel" "eth_univ" "ucy_univ" "ucy_zara01" "ucy_zara02")
datasets=("biwi_hotel" "crowds_students003" "crowds_students001" "crowds_zara01" "crowds_zara03" "lcas" "wildtrack")

for dataset in ${datasets[@]}; do
  python3 ./data/preprocessor.py --dataset $dataset --data_path ./data/trajnet_dataset --skip 10 --frame_stride 2
done
