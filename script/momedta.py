from subprocess import call
import os

###### NOTE: Set params here! ######
dataset = 'davis'
setting = 'novel_pair'
fold = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

use_1d = 'true'
use_2d = 'true'
use_3d = 'true'

encoder = '1d-cnn'
add_pool = 'true'
fusion = 'MANnew'

lr = '0.0001'
####################################

out_dir_modal = ''
if use_1d == 'true':
    out_dir_modal += '1d'
if use_2d == 'true':
    out_dir_modal += '2d'
if use_3d == 'true':
    out_dir_modal += '3d'

out_dir_pool = 'pool' if add_pool == 'true' else ''

call([
    'python', 'train.py',
    '--config-file', 'src/configuration/fusionnet.yaml',
    'DATASET.DS', dataset,
    'DATASET.SETTING', setting,
    'DATASET.FOLD', str(fold),
    # 'DATASET.TRAIN_DF', f'/data/qingyuyang/dta_ours/data/{dataset}/splits/{setting}/fold_{fold}_train.csv',
    # 'DATASET.VAL_DF', f'/data/qingyuyang/dta_ours/data/{dataset}/splits/{setting}/fold_{fold}_valid.csv',
    # 'DATASET.TEST_DF', f'/data/qingyuyang/dta_ours/data/{dataset}/splits/{setting}/fold_{fold}_test.csv',
    # 'DATASET.DATA_PATH', f'/data/qingyuyang/dta_ours/data/{dataset}',
    'MODEL.USE_1D', use_1d,
    'MODEL.USE_2D', use_2d,
    'MODEL.USE_3D', use_3d,
    'MODEL.ENCODER', encoder,
    'MODEL.ADD_POOL', add_pool,
    'MODEL.FUSION', fusion,
    'MODULE.OPTIMIZER.LR', lr,
    'OUTPUT_DIR', f'/home/qingyuyang/test/output/{dataset}/{setting}_fold{fold}/{out_dir_modal}_{fusion}_{encoder}_{lr}_{out_dir_pool}',
    'SEED', '42'
])