CUDA_VISIBLE_DEVICES=1 \
python train.py \
    --config-file src/configuration/frozen4down.yaml \
    TRAINER.STRATEGY auto \
    OUTPUT_DIR /home/qingyuyang/test/output/kiba/frozen4down/try_5e-5
    # --num-gpus 2 \