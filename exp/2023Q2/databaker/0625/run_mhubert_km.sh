#!/bin/bash

# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

# conda activate sovits

CUDA_VISIBLE_DEVICES=0,1

python resample.py --sr2 44100 --in_dir dataset_raw/databaker --out_dir2 dataset/44k/databaker_mhubert_km

python preprocess_flist_config.py \
    --speech_encoder mhubert-km \
    --source_dir dataset/44k/databaker_mhubert_km \
    --output_dir filelists/databaker_mhubert_km

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python preprocess_hubert_f0.py \
    --config_dir filelists/databaker_mhubert_km \
    --f0_predictor dio \
    --in_dir dataset/44k/databaker_mhubert_km \
    --num_processes 4

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
  -c filelists/databaker_mhubert_km/config.json \
  -m databaker_mhubert_km

