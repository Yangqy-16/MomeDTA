CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --config-file src/configuration/frozen4down_pdbbind.yaml \
    TRAINER.STRATEGY auto \
    OUTPUT_DIR /home/qingyuyang/test/output/pdbbind/frozen4down/try_aff_val_dim256
    # --num-gpus 2 \