import torch
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
from data_loader import *
from trainer import *
from model import *
from tqdm import tqdm
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import os

def test_model(opt):
    # create fold specific dictionaries
    test_data = get_dictionary(opt)

    # get number of  total channels
    chs = get_roi_len(opt)

    # device CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    # test
    test_set = data_to_tensor(test_data, opt.roi_list)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, **kwargs)

    # the network
    if opt.model == 'Bi-LSTM':
        model = BidirectionalLSTM(chs, 2000, 1)
    else:
        print('Error!')

    if opt.mode == 'test':
        model_file = '{}models/{}/saved_model_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)
        model.load_state_dict(torch.load(model_file, map_location=device))

        # count number of parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total number of parameters: %d' % pytorch_total_params)

    model = model.to(device)

    target_rvs, target_hrs, pred_rvs, pred_hrs, ids = test(model, device, test_loader, opt)

    # Save statistics and accuracy info
    fold_no = opt.train_fold.split('_')[-1]
    os.makedirs('{}results/{}/test_fold_{}/'.format(opt.out_dir, opt.uni_id, fold_no))
    prediction_file = '{}results/{}/test_fold_{}/pred_scans.csv'.format(opt.out_dir, opt.uni_id, fold_no)

    with open(prediction_file, "w") as f1, open('nan_files.txt', "w") as f3:
        for n, line in enumerate(ids):         
            # m = 137

            # if predictions are not nan, calculate pearson r to assess acc
            if np.isnan(pred_rvs[n]).all():
                f3.write('\n')
            else:
                rv_corr_coeff = sp.stats.pearsonr(pred_rvs[n][:].squeeze(), target_rvs[n][:].squeeze())
                hr_corr_coeff = sp.stats.pearsonr(pred_hrs[n][:].squeeze(), target_hrs[n][:].squeeze())

            
            # plot and save prediction vs output
            plt.figure(figsize=(20, 5))

            thr = zscore(target_hrs[n][:], axis=0)  # z-score normalization
            phr = zscore(pred_hrs[n][:], axis=0)  # z-score normalization
            
            plt.subplot(211)
            plt.plot(np.arange(0, len(phr)), phr)
            plt.plot(np.arange(0, len(phr)), thr)
            plt.ylabel('Heart Rate \n r={}'.format(np.round(hr_corr_coeff[0],3)))
            plt.title(ids[n])
            plt.legend(['Prediction', 'Target'], loc='upper right')
            sns.despine()


            trv = zscore(target_rvs[n][:], axis=0)  # z-score normalization
            prv = zscore(pred_rvs[n][:], axis=0)  # z-score normalization

            plt.subplot(212)
            plt.plot(np.arange(0, len(phr)), prv)
            plt.plot(np.arange(0, len(phr)), trv)
            plt.ylabel('Respiration Variation \n r={}'.format(np.round(rv_corr_coeff[0],3)))
            sns.despine()

            # save figure
            plt.savefig('/bigdata/HCP_rest/bad_samples/predicted/{}/test_fold_{}/QA/{}_QA.png'.format(opt.uni_id, fold_no, ids[n]))

            # writing to buffer
            f1.write('{}, {}, {}'.format(line.strip('\n'), str(rv_corr_coeff[0]), str(hr_corr_coeff[0])))
            f1.write('\n')

            # writing to disk
            f1.flush()

            # save time-series output (optional)
            rvp = '/bigdata/HCP_rest/bad_samples/predicted/{}/test_fold_{}/{}_rv_pred.mat'.format(opt.uni_id, fold_no, ids[n])
            rvt = '/bigdata/HCP_rest/bad_samples/predicted/{}/test_fold_{}/{}_rv_target.mat'.format(opt.uni_id, fold_no, ids[n])
            hrp = '/bigdata/HCP_rest/bad_samples/predicted/{}/test_fold_{}/{}_hr_pred.mat'.format(opt.uni_id, fold_no, ids[n])
            hrt = '/bigdata/HCP_rest/bad_samples/predicted/{}/test_fold_{}/{}_hr_target.mat'.format(opt.uni_id, fold_no, ids[n])

            savemat(rvp, {'rv_pred' : pred_rvs[n]})
            savemat(rvt, {'rv_target' : target_rvs[n]})
            savemat(hrp, {'hr_pred' : pred_hrs[n]})
            savemat(hrt, {'hr_target' : target_hrs[n]})


def main():
    # pass in command line arguments
    parser = argparse.ArgumentParser()

    # data 
    parser.add_argument('--input_dir', type=str, default='JOURNAL-NAME-2022/example_data/')
    parser.add_argument('--out_dir', type=str, default='JOURNAL-NAME-2022/inference/', help='Path to output directory')
    parser.add_argument('--roi_list', type=str, default=['schaefer', 'tractseg', 'tian', 'aan'], help='list of atlases')

    # infer
    parser.add_argument('--uni_id', type=str, default='Bi-LSTM_schaefertractsegtianaan_lr_0.001_l1_0.5', help='unique name')
    parser.add_argument('--train_fold', default='train_fold_0', help='train_fold_k')
    parser.add_argument('--model', type=str, default='Bi-LSTM', help='model name')
    parser.add_argument('--mode', type=str, default='test', help='Determines whether to backpropagate or not')
    parser.add_argument('--test_batch', type=int, default=1, help='Decides size of each val batch')
    parser.add_argument('--dropout', type=float, default=0.3, help='the percentage to drop at each epoch')

    opt = parser.parse_args()
    print(opt)

    if opt.mode == 'test':
        test_model(opt)
    else:
        print('Mode is confused.')


if __name__ == '__main__':
    main()
