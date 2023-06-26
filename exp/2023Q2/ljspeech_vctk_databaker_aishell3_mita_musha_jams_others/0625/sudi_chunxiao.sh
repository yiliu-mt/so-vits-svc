# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits


## test: cluster-based timbre leakage control
CUDA_VISIBLE_DEVICES=0 python inference.py -m "/nfs1/yi.liu/src/so-vits-svc/logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/G_300000.pth" \
    -c "/nfs1/yi.liu/src/so-vits-svc/logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/yichao/sudi_chunxiao_20230625/wav_test.scp \
    --output_dir /nfs1/yi.liu/src/so-vits-svc/logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/SSB3003_yichao_sudi_chunxiao_20230625