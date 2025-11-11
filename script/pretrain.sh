CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --config-file src/configuration/pretrain.yaml \
    TRAINER.STRATEGY auto \
    OUTPUT_DIR /home/qingyuyang/test/output/pretrain/try1
    # --num-gpus 2 \