from __future__ import print_function, division
import torch
import os

def infer(model, device, test_loader):
    model.eval()

    pred_rvs, pred_hrs = [], []
    ids = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            input = sample['fmri']
            # Get ID from the sample instead of the dataset path
            subject_id = sample['subject_id'][0]  # Get string from tensor/list
            print(f"Processing subject: {subject_id}")
            
            input = input.to(device)

            output_rv, output_hr = model(input)

            # convert torch to numpy array
            pred_rv = output_rv.detach().cpu().numpy()
            pred_hr = output_hr.detach().cpu().numpy()

            pred_rvs.append(pred_rv.squeeze())
            pred_hrs.append(pred_hr.squeeze())
            ids.append(subject_id)

    return pred_rvs, pred_hrs, ids