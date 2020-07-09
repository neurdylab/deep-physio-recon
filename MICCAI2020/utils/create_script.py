import os

fname = 'miccai_rebuttal_findlab90.sh'
activate = 'source /home/bayrakrg/Tools/VENV/python37/bin/activate'
k_fold_path = '/home/bayrakrg/neurdy/pycharm/RV/k_fold_files/'
out_path = '/home/bayrakrg/neurdy/pycharm/RV/out/'
log_path = '/home/bayrakrg/neurdy/pycharm/RV/log'
top = '#!/bin/bash'
line = 'python main.py --model={} --roi_clust={} --uni_id={} --lr={} --train_fold=train_fold_{}.txt --test_fold=test_fold_{}.txt --decay_epoch={} --mode={} > {}/{}/{}.txt'

mode = ['train', 'test']
model = ['linear', 'Bi-LSTM', 'U-Net']
rois = ['findlab90']
lr = ['0.001', '0.000001', '0.000001']
decay = ['999999', '30', '400']

line_list = []
with open(fname, 'w') as f:
	f.write('{}\n'.format(top))
	f.write('{}\n'.format(activate))

	for roi in rois:
		for i, mo in enumerate(model):
			for m in mode:
				folds = os.listdir(k_fold_path)
				for fo in folds:
					if m in fo:
						id = fo.strip('tesrainfoldx_.')
						uni_id = '{}_{}_lr_{}'.format(mo, roi, lr[i])

						# create log directories
						log_path = '/home/bayrakrg/neurdy/pycharm/RV/log/{}'.format(uni_id)
						if not os.path.isdir(log_path):
							os.makedirs('/home/bayrakrg/neurdy/pycharm/RV/log/{}/train'.format(uni_id))
							os.makedirs('/home/bayrakrg/neurdy/pycharm/RV/log/{}/test'.format(uni_id))

						run = line.format(mo, roi, uni_id, lr[i], id, id, decay[i], m, log_path, m, id)
						# line_list.append(run)
						f.write('{}\n'.format(run))