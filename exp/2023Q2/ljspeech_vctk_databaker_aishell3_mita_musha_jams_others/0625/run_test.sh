# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

cp -r filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0625_contentvec
cp -r filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0625_whisper_large

python preprocess_flist_config.py \
    --speech_encoder vec768l12 \
    --speech_decoder vits-hifigan \
    --use_f0 false --bidirectional_flow true --speaker_grl true --ppg_std 0.1 --vae_std 0.1 \
    --source_dir dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607 \
    --output_dir filelists/test

CUDA_VISIBLE_DEVICES=1 python train.py \
    -c filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0625_contentvec/config.json \
    -m ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0625_contentvec

CUDA_VISIBLE_DEVICES=1 python train.py \
    -c filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0625_whisper_large/config.json \
    -m ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0625_whisper_large
