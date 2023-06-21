# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

python resample.py --sr2 44100 --in_dir dataset_raw/databaker --out_dir2 dataset/44k/databaker_conformer

python preprocess_flist_config.py \
    --speech_encoder conformer-ppg \
    --source_dir dataset/44k/databaker_conformer \
    --output_dir filelists/databaker_conformer

CUDA_VISIBLE_DEVICES=6 python preprocess_hubert_f0.py \
    --config_dir filelists/databaker_conformer \
    --f0_predictor dio \
    --in_dir dataset/44k/databaker_conformer \
    --num_processes 4

CUDA_VISIBLE_DEVICES=1,6 python train.py -c filelists/databaker_conformer/config.json -m databaker_conformer

