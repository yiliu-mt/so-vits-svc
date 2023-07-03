#!/bin/bash

num_step=${1-500000}
speech_encoder=${2-conformer_large512l8_v2}
data_name=databaker
speaker=SSB3000

CUDA_VISIBLE_DEVICES=7

# for spk in databaker jams liuyi xiaolin; do
for spk in xiaolin; do
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python inference.py -m "logs/${data_name}_${speech_encoder}/G_${num_step}.pth" -c "filelists/${data_name}_${speech_encoder}/config.json" -s ${speaker} -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
	    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/$spk/wav_test.scp --output_dir logs/${data_name}_${speech_encoder}/test/$num_step/$spk
done

for test_dir in /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/*; do
  wav_scp=$test_dir/wav_test.scp
  emotion=$(basename $test_dir)
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python inference.py -m "logs/${data_name}_${speech_encoder}/G_${num_step}.pth" -c "filelists/${data_name}_${speech_encoder}/config.json" -s ${speaker} -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
	    --wav_scp $wav_scp --output_dir logs/${data_name}_${speech_encoder}/test/$num_step/MDT-TTS-G005/$emotion
done

