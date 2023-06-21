#!/bin/bash

# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

# conda activate sovits

CUDA_VISIBLE_DEVICES=4,5,6,7

python resample.py --sr2 44100 --in_dir dataset_raw/databaker --out_dir2 dataset/44k/databaker_conformer_large_l9

python preprocess_flist_config.py \
    --speech_encoder conformer-ppg-large \
    --source_dir dataset/44k/databaker_conformer_large_l9 \
    --output_dir filelists/databaker_conformer_large_l9

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python preprocess_hubert_f0.py \
    --config_dir filelists/databaker_conformer_large_l9 \
    --f0_predictor dio \
    --in_dir dataset/44k/databaker_conformer_large_l9 \
    --num_processes 4

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
  -c filelists/databaker_conformer_large_l9/config.json \
  -m databaker_conformer_large_l9

