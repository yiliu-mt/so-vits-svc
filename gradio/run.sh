# docker run --rm --gpus all -p 8911:8911 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vits:demo bash

# PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 /root/miniconda3/bin/conda run python gradio/run.py \
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 \
  python gradio/run.py \
    --config /nfs1/yi.liu/src/so-vits-svc/logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json \
    --model /nfs1/yi.liu/src/so-vits-svc/logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth \
    --slice_db -40 \
    -f0p dio \
    -a \
    -lg 1 \
    --port 8080

    # --config logs/databaker_conformer_large_l9/config.json \
    # --model logs/databaker_conformer_large_l9/G_430000.pth \

    # --config /nfs1/yi.liu/src/so-vits-svc/logs/databaker_contentvec/config.json \
    # --model /nfs1/yi.liu/src/so-vits-svc/logs/databaker_contentvec/G_280000.pth \

