# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

# Train clusters
python cluster/train_cluster.py --gpu --dataset dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607 \
    --spk SSB3003 --output logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams


## test: cluster-based timbre leakage control
CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/G_300000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    -cm logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/kmeans_10000.pt -cr 0.5 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/SSB3003_xiaolin



# Train feature retrieval
python train_index.py -c filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/config.json \
    --spk SSB3003 \
    --root_dir dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607\
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams

## test: RVC
CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/G_300000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    -fr -cm logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/feature_and_index.pkl -cr 0.5 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/SSB3003_xiaolin_rvc

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/G_300000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    -fr -cm logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/feature_and_index.pkl -cr 0.5 \
    --wav_scp /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/pride/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/SSB3003_pride_rvc

