import torch
import argparse
from torch.utils.data import DataLoader
from scipy.stats import zscore
from scipy.io import savemat
from src.data_loader import get_dictionary, get_roi_len, data_to_tensor
from src.demo import demo
from src.model import BidirectionalLSTM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

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
    test_loader = DataLoader(dataset=test_set, batch_size=1, **kwargs)

    # the network
    if opt.model == 'Bi-LSTM':
        model = BidirectionalLSTM(chs, opt.hidden_size, 1)
    else:
        print('Error!')

    if opt.mode == 'test':
        model_file = f'./weights/{opt.uni_id}/saved_model_split_{opt.train_fold}'
        model.load_state_dict(torch.load(model_file, map_location=device))

        # count number of parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total number of parameters: %d' % pytorch_total_params)

    model = model.to(device)

    target_rvs, target_hrs, pred_rvs, pred_hrs, ids = demo(model, device, test_loader, opt)

    # Save statistics and accuracy info
    os.makedirs(f'{opt.output_dir}/{opt.uni_id}/{opt.train_fold}', exist_ok=True)

    quality_metrics = {}
    
    for n, _ in enumerate(ids):         

        if not np.isnan(pred_rvs[n]).all() and not np.isnan(target_rvs[n]).all():
            rv_corr_coeff = np.corrcoef(pred_rvs[n][:].squeeze(), target_rvs[n][:].squeeze())[0, 1]
            hr_corr_coeff = np.corrcoef(pred_hrs[n][:].squeeze(), target_hrs[n][:].squeeze())[0, 1]
            quality_metrics[ids[n]] = {
                'RV Pearson r': rv_corr_coeff,
                'HR Paerson r': hr_corr_coeff
            }

            # plot and save prediction vs output
            plt.figure(figsize=(20, 5))

            thr = zscore(target_hrs[n][:], axis=0)  # z-score normalization
            phr = zscore(pred_hrs[n][:], axis=0)  # z-score normalization
            
            plt.subplot(211)
            plt.plot(np.arange(0, len(phr)), phr)
            plt.plot(np.arange(0, len(phr)), thr)
            plt.ylabel('Heart Rate \n r={}'.format(np.round(hr_corr_coeff,3)))
            plt.title(ids[n])
            plt.legend(['Prediction', 'Target'], loc='upper right')
            sns.despine()

            trv = zscore(target_rvs[n][:], axis=0)  # z-score normalization
            prv = zscore(pred_rvs[n][:], axis=0)  # z-score normalization

            plt.subplot(212)
            plt.plot(np.arange(0, len(phr)), prv)
            plt.plot(np.arange(0, len(phr)), trv)
            plt.ylabel('Respiration Variation \n r={}'.format(np.round(rv_corr_coeff,3)))
            sns.despine()

            # save figure
            plt.savefig(f'{opt.output_dir}/{opt.uni_id}/{opt.train_fold}/{ids[n]}_QA.png')

            # save time-series output (optional)
            data_types = ['rv_pred', 'rv_target', 'hr_pred', 'hr_target']
            data_arrays = [pred_rvs[n], target_rvs[n], pred_hrs[n], target_hrs[n]]
            for data_type, data_array in zip(data_types, data_arrays):
                file_path = f'{opt.output_dir}/{opt.uni_id}/{opt.train_fold}/{ids[n]}_{data_type}.mat'
                savemat(file_path, {data_type: data_array})


    # Save the quality metrics to a JSON file
    metrics_file = f'{opt.output_dir}/{opt.uni_id}/{opt.train_fold}/subject_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(quality_metrics, f, indent=4)

    print(f"Quality metrics saved to {metrics_file}")



def main():
    # pass in command line arguments
    parser = argparse.ArgumentParser()

    # data 
    parser.add_argument('--input_dir', type=str, default='./example_data')
    parser.add_argument('--output_dir', type=str, default='./results_demo', help='Path to output directory')
    parser.add_argument('--roi_list', type=str, default=['schaefer', 'tractseg', 'tian', 'aan'], help='list of atlases')

    # inference specific parameters
    parser.add_argument('--uni_id', type=str, default='Bi-LSTM_schaefertractsegtianaan_lr_0.001_l1_0.5', help='unique model name')
    parser.add_argument('--train_fold', default='train_fold_0', help='train_fold_k')
    parser.add_argument('--model', type=str, default='Bi-LSTM', help='model name')
    parser.add_argument('--mode', type=str, default='test', help='Determines whether to backpropagate or not')
    parser.add_argument('--test_batch', type=int, default=1, help='Decides size of each val batch')
    parser.add_argument('--dropout', type=float, default=0.3, help='the percentage to drop at each epoch')
    parser.add_argument('--hidden_size', type=int, default=2000, help='the number of hidden units')
    opt = parser.parse_args()
    print(opt)

    if opt.mode == 'test':
        test_model(opt)
    else:
        print('Mode is confused.')


if __name__ == '__main__':
    main()
