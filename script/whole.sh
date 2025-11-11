CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --config-file src/configuration/whole.yaml \
    TRAINER.STRATEGY auto \
    OUTPUT_DIR /home/qingyuyang/test/output/whole/try1
    # --num-gpus 2 \