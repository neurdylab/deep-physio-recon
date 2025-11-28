from __future__ import print_function, division
from src.loss import *

def demo(model, device, test_loader, opt):
    model.eval()

    pred_rvs, pred_hrs, target_rvs, target_hrs = [], [], [], []
    ids = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            input = sample['roi']
            target_rv = sample['rv']
            target_hr = sample['hr']
            id = sample['hr_path']

            input, target_rv, target_hr = input.to(device), target_rv.to(device), target_hr.to(device)

            output_rv, output_hr = model(input)

            # convert torch to numpy array
            pred_rv = output_rv.detach().cpu().numpy()
            pred_hr = output_hr.detach().cpu().numpy()

            # print(pred.shape)
            target_rv = target_rv.detach().cpu().numpy()
            target_hr = target_hr.detach().cpu().numpy()

            pred_rvs.append(pred_rv.squeeze())
            pred_hrs.append(pred_hr.squeeze())
            target_rvs.append(target_rv.squeeze())
            target_hrs.append(target_hr.squeeze())
            ids.append(id[0].split('/')[-1])

    return target_rvs, target_hrs, pred_rvs, pred_hrs, ids