import torch
import argparse
from scipy.stats import zscore
from scipy.io import savemat
from src.better_data_loader import get_dataloader, Normalize
from src.infer import infer
from src.model import BidirectionalLSTM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

def test_model(opt):
    # device CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if output directory exists
    out_dir = Path(opt.out_dir) / opt.uni_id / opt.train_fold
    if out_dir.exists():
        if opt.overwrite:
            print(f"Warning: Output directory {out_dir} exists. Overwriting...")
        else:
            raise ValueError(f"Output directory {out_dir} already exists. Use --overwrite to force.")
    
    # Create data loader with normalization
    transform = Normalize()
    test_loader = get_dataloader(
        data_path=opt.input_dir,
        roi_list=opt.roi_list,
        mode='inference',
        transform=transform
    )

    # Get input size from dataset
    input_size = test_loader.dataset.input_size

    # the network
    if opt.model == 'Bi-LSTM':
        model = BidirectionalLSTM(input_size, opt.hidden_size, 1)
    else:
        print('Error!')

    if opt.mode == 'test':
        model_file = f'./weights/{opt.uni_id}/saved_model_split_{opt.train_fold}'
        model.load_state_dict(torch.load(model_file, map_location=device))

        # count number of parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total number of parameters: %d' % pytorch_total_params)

    model = model.to(device)

    pred_rvs, pred_hrs, ids = infer(model, device, test_loader)

    # Save statistics and accuracy info
    os.makedirs(f'{opt.out_dir}/{opt.uni_id}/{opt.train_fold}', exist_ok=True)
    
    for n, _ in enumerate(ids):         
        if not np.isnan(pred_rvs[n]).all():
            # plot and save prediction
            plt.figure(figsize=(20, 5))

            phr = zscore(pred_hrs[n][:], axis=0)
            
            plt.subplot(211)
            plt.plot(np.arange(0, len(phr)), phr)
            plt.ylabel('Heart Rate (z-scored)')
            plt.title(f'Predictions for {ids[n]}', fontsize=14, pad=20)
            sns.despine()

            prv = zscore(pred_rvs[n][:], axis=0)

            plt.subplot(212)
            plt.plot(np.arange(0, len(phr)), prv)
            plt.ylabel('Respiration Variation (z-scored)')
            sns.despine()

            # save figure
            plt.savefig(f'{opt.out_dir}/{opt.uni_id}/{opt.train_fold}/{ids[n]}_QA.png')

            # save time-series output
            data_types = ['rv_pred', 'hr_pred']
            data_arrays = [pred_rvs[n], pred_hrs[n]]
            for data_type, data_array in zip(data_types, data_arrays):
                file_path = f'{opt.out_dir}/{opt.uni_id}/{opt.train_fold}/{ids[n]}_{data_type}.mat'
                savemat(file_path, {data_type: data_array})

def main():
    parser = argparse.ArgumentParser()

    # data 
    parser.add_argument('--input_dir', type=str, default='./data')
    parser.add_argument('--out_dir', type=str, default='./results', help='Path to output directory')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing output directory if it exists')
    
    # model parameters
    parser.add_argument('--hidden_size', type=int, default=2000, help='Number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')

    # inference specific parameters
    parser.add_argument('--uni_id', type=str, default='Bi-LSTM_schaefertractsegtianaan_lr_0.001_l1_0.5', help='unique model name')
    parser.add_argument('--train_fold', default='train_fold_0', help='train_fold_k')
    parser.add_argument('--model', type=str, default='Bi-LSTM', help='model name')
    parser.add_argument('--mode', type=str, default='test', help='Determines whether to backpropagate or not')
    parser.add_argument('--roi_list', type=str, default=['schaefer', 'tractseg', 'tian', 'aan'], 
                       help='list of atlases')
    
    opt = parser.parse_args()
    print(opt)

    if opt.mode == 'test':
        test_model(opt)
    else:
        print('Mode is confused.')

if __name__ == '__main__':
    main()
