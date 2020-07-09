from __future__ import print_function, division
from loss import *
import scipy as sp
import os
import audtorch

def train(model, device, train_loader, optim, opt):
    # training
    model.train()

    total_loss = 0
    preds = []
    targets = []

    for batch_idx, sample in enumerate(train_loader):

        data = sample['roi']
        target = sample['rv']
        target = target.type(torch.FloatTensor)
        data, target = data.to(device), target.to(device)
        optim.zero_grad()
        output = model(data)
        # print(target.shape)
        # print(output.shape)

        loss = pearsonr(output, target)
        # loss = audtorch.metrics.functional.pearsonr(output.squeeze(), target.squeeze(), batch_first=True)
        # criterion = torch.nn.MSELoss()
        # loss = criterion(output, target)
        loss.backward()
        optim.step()

        # running average
        total_loss += loss.item()  #.item gives you a floating point value instead of a tensor

    # convert torch to numpy array
    pred = output.detach().cpu().numpy()
    # print(pred.shape)
    target = target.detach().cpu().numpy()
    # print(target.shape)
    avg_loss = total_loss / len(train_loader)

    return avg_loss, target, pred


def test(model, device, test_loader, opt):
    model.eval()

    total_loss = 0
    preds = []
    targets = []

    test_corr_file = '/home/bayrakrg/Data/RV/model_prediction/corr_score_{}'.format(opt.test_fold)
    with open(test_corr_file, "w") as file:

        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                data = sample['roi']
                target = sample['rv']
                rv_paths = sample['rv_path']

                data, target = data.to(device), target.to(device)
                output = model(data)
                output = output.to(device)
                # criterion = torch.nn.MSELoss()
                # loss = criterion(output, target)
                loss = pearsonr(output, target)

                # convert torch to numpy array
                pred = output.cpu().numpy()
                # print(pred.shape)
                target = target.cpu().numpy()
                # print(target.shape)

                if opt.mode == 'test':
                    name = rv_paths[0].split('/')[-1].rsplit('.mat')[0]
                    np.savetxt('/home/bayrakrg/Data/RV/model_prediction/{}/pred/{}.txt'.format(opt.test_fold.rstrip('.txt'), name), pred.squeeze())
                    np.savetxt('/home/bayrakrg/Data/RV/model_prediction/{}/gt/{}.txt'.format(opt.test_fold.rstrip('.txt'), name), target.squeeze())

                    corr_coeff = sp.stats.pearsonr(pred.squeeze(), target.squeeze())
                    file.write(str(corr_coeff[0]))
                    file.write('\n')

                preds.append(pred.squeeze())
                targets.append(target.squeeze())

                total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)

    return avg_loss, targets, preds