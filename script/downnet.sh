CUDA_VISIBLE_DEVICES=1 \
python train.py \
    --config-file src/configuration/downnet.yaml \
    TRAINER.STRATEGY auto \
    OUTPUT_DIR /home/qingyuyang/test/output/downnet/try1
    # --num-gpus 2 \