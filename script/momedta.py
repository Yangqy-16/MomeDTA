from subprocess import call
import os

###### NOTE: Set params here! ######
dataset = 'metz'
setting = 'warm'
fold = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

use_1d = 'true'
use_2d = 'true'
use_3d = 'true'

encoder = '1d-cnn'
add_pool = 'true'
fusion = 'MANnew'

bs = '16'
lr = '0.0001'

log_step = 100 if dataset == 'davis' else 400
####################################

out_dir_modal = ''
if use_1d == 'true':
    out_dir_modal += '1d'
if use_2d == 'true':
    out_dir_modal += '2d'
if use_3d == 'true':
    out_dir_modal += '3d'

model_name = 'MoE'
if (use_1d == 'true') + (use_2d == 'true') + (use_3d == 'true') <= 1:
    model_name = 'Ablation'

out_dir_pool = 'pool' if add_pool == 'true' else ''

call([
    'python', 'train.py',
    '--config-file', 'src/configuration/momedta.yaml',
    'MODEL.NAME', model_name,
    'DATASET.DS', dataset,
    'DATASET.SETTING', setting,
    'DATASET.FOLD', str(fold),
    'DATALOADER.TRAIN.BATCH_SIZE', str(bs),
    'MODEL.USE_1D', use_1d,
    'MODEL.USE_2D', use_2d,
    'MODEL.USE_3D', use_3d,
    'MODEL.ENCODER', encoder,
    'MODEL.ADD_POOL', add_pool,
    'MODEL.FUSION', fusion,
    'MODULE.OPTIMIZER.LR', lr,
    'TRAINER.LOG_EVERY_N_STEPS', str(log_step),
    'OUTPUT_DIR', f'/data/qingyuyang/dta_ours/output/{dataset}/{setting}_fold{fold}/momedta_{out_dir_modal}_{fusion}_{bs}_{lr}', #_{encoder}_{out_dir_pool}
    'SEED', '42'
])