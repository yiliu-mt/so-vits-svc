# docker run --ipc host --gpus all -v /home:/home -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/G_55000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/config.json" \
    -s LYG0002 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/yangjie_227/wav_test.scp \
    --output_dir /home/tmp-yi.liu/yangjie/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/LYG0002_yangjie_227