CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --config-file /home/qingyuyang/test/output/kiba/warm_fold0/moe_fullemb_2d_MANnew_16_0.0001/csv_log/version_0/hparams.yaml \
    --resume /home/qingyuyang/test/output/kiba/warm_fold0/moe_fullemb_2d_MANnew_16_0.0001/best_ckpts/epoch=497-mse=0.2065.ckpt