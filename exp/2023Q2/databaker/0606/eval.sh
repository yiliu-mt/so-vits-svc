#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_conformer/G_50000.pth" -c "filelists/databaker_conformer/config.json" -s SSB3000 -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
# 	    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/vc/wav.scp --output_dir logs/databaker_conformer/test

# CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_contentvec_nsf/G_50000.pth" -c "filelists/ljspeech_contentvec_nsf/config.json" -s SSB3000 -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
# 	    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/vc/wav.scp --output_dir logs/ljspeech_contentvec_nsf/test

num_step=${1-170000}

# for spk in databaker jams liuyi xiaolin; do
# for spk in liuyi xiaolin; do
#   CUDA_VISIBLE_DEVICES=0 python inference.py -m "logs/databaker_conformer/G_${num_step}.pth" -c "filelists/databaker_conformer/config.json" -s SSB3000 -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
# 	    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/$spk/wav_test.scp --output_dir logs/databaker_conformer/test/$num_step/$spk
# done

for test_dir in /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/*; do
  wav_scp=$test_dir/wav_test.scp
  emotion=$(basename $test_dir)
  CUDA_VISIBLE_DEVICES=0 python inference.py -m "logs/databaker_conformer/G_${num_step}.pth" -c "filelists/databaker_conformer/config.json" -s SSB3000 -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
	    --wav_scp $wav_scp --output_dir logs/databaker_conformer/test/$num_step/MDT-TTS-G005/$emotion
done

