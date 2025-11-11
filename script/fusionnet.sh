CUDA_VISIBLE_DEVICES=1 \
python train.py \
    --config-file src/configuration/fusionnet.yaml \
    # TRAINER.STRATEGY auto \
    # OUTPUT_DIR /home/qingyuyang/test/output/kiba/fusionnet/fulldata_man_lwca_try
    # --num-gpus 2 \