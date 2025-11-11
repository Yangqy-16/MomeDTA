from subprocess import call
import os

###### NOTE: Set params here! ######
dataset = 'kiba'
setting = 'warm'
fold = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# use_1d = 'true'
# use_2d = 'true'
# use_3d = 'true'

# encoder = '1d-cnn'
# add_pool = 'true'
# fusion = 'MANnew'

bs = 16
lr = '0.00005'
####################################

# out_dir_modal = ''
# if use_1d == 'true':
#     out_dir_modal += '1d'
# if use_2d == 'true':
#     out_dir_modal += '2d'
# if use_3d == 'true':
#     out_dir_modal += '3d'

# out_dir_pool = 'pool' if add_pool == 'true' else ''

call([
    'python', 'train.py',
    '--config-file', 'src/configuration/mgraph.yaml',
    'DATASET.DS', dataset,
    'DATASET.SETTING', setting,
    'DATASET.FOLD', str(fold),
    'MODEL.DRUG_2D_MODEL', 'null', #molclr
    'MODEL.DRUG_2D_DIM', 'null', #300
    'MODEL.PROT_2D_MODEL', 'null', #gearnet
    'MODEL.PROT_2D_DIM', 'null', #3072    
    'DATALOADER.TRAIN.BATCH_SIZE', str(bs),
    'MODULE.OPTIMIZER.LR', lr,
    'TRAINER.LOG_EVERY_N_STEPS', '200',
    'OUTPUT_DIR', f'/home/qingyuyang/test/output/{dataset}/{setting}_fold{fold}/mgraph_dmgcn_psage_{bs}_{lr}',
])