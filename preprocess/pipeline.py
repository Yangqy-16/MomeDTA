from subprocess import call

################ NOTE: Change these ################
input_csv = '/data/yueteng/DTI/Davis/davis_deepdta_redu_dataset.tsv'
root = '/data/qingyuyang/dta_ours/data/davis'
####################################################

print('===================== Running step1.py =====================')
call(['python', 'step1.py', '--input', input_csv, '--root', root])
print('===================== Running step2.py =====================')
call(['python', 'step2.py', '--root', root])
print('===================== Running selformer_rep.py =====================')
call(['python', 'selformer_rep.py', '--root', root])
print('===================== Running molclr_rep.py =====================')
call(['python', 'molclr_rep.py', '--root', root])
print('===================== Running unimol_conf.py =====================')
call(['python', 'unimol_conf.py', '--root', root])
print('===================== Running unimol_infer.py =====================')
call(['python', 'unimol_infer.py', '--root', root])
print('===================== Running esm2_rep.py =====================')
call(['python', 'esm2_rep.py', '--root', root])
print('===================== Running gearnet_rep.py =====================')
call(['python', 'gearnet_rep.py', '--root', root])
print('===================== Running saprot_rep.py =====================')
call(['python', 'saprot_rep.py', '--root', root])
print('===================== Running split.py =====================')
call(['python', 'split.py', '--root', root])
