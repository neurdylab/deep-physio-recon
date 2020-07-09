import torch
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
from data_loader import *
# from data_loader_allvox import *
from trainer import *
from model import *
from tqdm import tqdm
import scipy as sp


# def print_network(net):
#     num_params = 0
#     for param in net.parameters():
#         num_params += param.numel()
#     print(net)
#     print('Total number of parameters: %d' % num_params)


def train_model(opt):
    # device CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

    # create fold specific dictionaries for train and validation split
    train_data = get_dictionary(opt.train_fold)
    keys = list(train_data.keys())

    val_split = round(len(train_data) * opt.val_split)
    val_data = {}
    for i in range(val_split):
        idx = random.randint(0, len(keys) - 1)
        val_data[keys[idx]] = train_data[keys[idx]]
        del train_data[keys[idx]]
        del keys[idx]

    # save the train and validation splits here
    with (open("k_fold_files/train_split_{}".format(opt.train_fold), "w")) as f:
        f.write('\n'.join(sorted(train_data.keys())))
        f.write('\n')  # newline at the end of file

    with (open("k_fold_files/val_split_{}".format(opt.train_fold), "w")) as f:
        f.write('\n'.join(sorted(val_data.keys())))
        f.write('\n')  # newline at the end of file

    # load the train/val data as tensor
    train_set = data_to_tensor(train_data, opt.roi_clust)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=opt.train_batch, shuffle=True, **kwargs)

    val_set = data_to_tensor(val_data, opt.roi_clust)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, shuffle=True, **kwargs)

    # load network
    if opt.model == 'linear':
        model = linear(90, 1, opt)
    elif opt.model == 'U-Net':
        model = UNet(90, 1, opt)
    elif opt.model == 'Bi-LSTM':
        model = BidirectionalLSTM(90, 600, 1)

    # load optimizer
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
    train_loss_file = '{}/results/{}/train_loss_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)
    f = open(train_loss_file, 'w')
    f.close()
    validate_loss_file = '{}/results/{}/validate_loss_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)
    f = open(validate_loss_file, 'w')
    f.close()

    model_file = '{}/models/{}/saved_model_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)

    seq_increase = 0
    min_loss = 10000

    if opt.continue_training:
        model.load_state_dict(torch.load(model_file))
        model = model.to(device)
    else:
        model = model.to(device)

    with tqdm(total=opt.epoch) as pbar:
        for epoch in range(1, opt.epoch + 1):

            avg_loss, target, pred = train(model, device, train_loader, optim, opt)

            avg_val_loss, targets, preds = test(model, device, val_loader, opt)

            with open(train_loss_file, "a") as file:
                file.write(str(avg_loss))
                file.write('\n')

            with open(validate_loss_file, "a") as file:
                file.write(str(avg_val_loss))
                file.write('\n')

            # save model or stop training
            if avg_val_loss < min_loss:
                min_loss = avg_val_loss
                with open(model_file, 'wb') as f:
                    torch.save(model.state_dict(), f)

            # early stopper
            elif opt.early_stop != -1:
                if avg_val_loss > min_loss:
                    seq_increase += 1
                    if seq_increase == opt.early_stop:
                        break
                else:
                    seq_increase = 0

            if opt.decay_epoch != -1:
                if epoch % opt.decay_epoch == 0:
                    opt.lr = opt.lr * opt.decay_rate
                    print('new lr: {}'.format(opt.lr))

            # progress bar
            pbar.set_description(
                "Epoch {}  \t Avg. Loss: {:.4f}    \t Avg. Val. Loss: {:.4f}".format(epoch, avg_loss, avg_val_loss))
            pbar.update(1)

    # n = 8
    # target = targets[n][:]
    # rv = preds[n][:]
    #
    # norm_target = (target - target.mean(axis=0)) / target.std(axis=0)  # z-score normalization
    # norm_pred = (rv - rv.mean(axis=0)) / rv.std(axis=0)  # z-score normalization
    #
    # # # plot prediction vs output
    # plt.figure(1)
    #
    # plt.plot(np.arange(0, 600), norm_pred)
    # plt.plot(np.arange(0, 600), norm_target)
    # plt.legend(['Prediction', 'Target'])
    # plt.show()


def test_model(opt):
    # create fold specific dictionaries
    test_data = get_dictionary(opt.test_fold)

    # device CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    test_set = data_to_tensor(test_data, opt.roi_clust)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, **kwargs)
    # print('hi!')

    # the network
    if opt.model == 'linear':
        model = linear(90, 1, opt)
    elif opt.model == 'U-Net':
        model = UNet(90, 1, opt)
    elif opt.model == 'Bi-LSTM':
        model = BidirectionalLSTM(90, 600, 1)

    if opt.mode == 'test':
        model_file = '{}models/{}/saved_model_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)
        model.load_state_dict(torch.load(model_file))
        # count number of parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total number of parameters: %d' % pytorch_total_params)

        # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = model.to(device)

    avg_loss, targets, preds = test(model, device, test_loader, opt)

    # # plot prediction vs output
    plt.figure(1)
    n = 14
    target = targets[n][:]
    rv = preds[n][:]

    norm_target = (target - target.mean(axis=0)) / target.std(axis=0)  # z-score normalization
    norm_pred = (rv - rv.mean(axis=0)) / rv.std(axis=0)  # z-score normalization

    plt.plot(np.arange(0, 600), norm_pred)
    plt.plot(np.arange(0, 600), norm_target)
    plt.legend(['Prediction', 'Target'])
    cross_corr = np.correlate(preds[n][:].squeeze(), targets[n].squeeze())
    corr_coeff = sp.stats.pearsonr(preds[n][:].squeeze(), targets[n].squeeze())
    plt.title(
        "{}th_{} | corr_coeff={} | cross_corr={}".format(n, opt.test_fold.rstrip('.txt'), corr_coeff[0], cross_corr[0]))
    plt.show()

    # Save statistics
    plot_stats = []
    test_corr_file = '{}/results/{}/test_corr_splits_{}'.format(opt.out_dir, opt.uni_id, opt.test_fold)
    with open(test_corr_file, "w") as file:
        # calculate test statistics
        for n, pred in enumerate(preds):
            corr_coeff = sp.stats.pearsonr(preds[n].squeeze(), targets[n].squeeze())

            file.write(str(corr_coeff[0]))
            file.write('\n')

            plot_stats.append(corr_coeff[0])

    # plot statistics
    fig3, ax3 = plt.subplots()
    ax3.set_title('BiLSTM_268')
    ax3.boxplot(plot_stats, notch=True)
    ax3.set_ylim(bottom=0, top=1)

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', alpha=0.2)


def main():
    # pass in command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--uni_id', type=str, default='linear_findlab90_lr_0.001')
    parser.add_argument('--epoch', type=int, default=3000, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.00001')
    parser.add_argument('--roi_clust', default='findlab90', help='clust10 | clust42 | findlab90')
    parser.add_argument('--test_fold', default='test_fold_0.txt', help='test_fold_k')
    parser.add_argument('--train_fold', default='train_fold_0.txt', help='train_fold_k')
    parser.add_argument('--val_split', type=float, default=0.25, help='percentage of the split')

    parser.add_argument('--out_dir', type=str, default='/home/bayrakrg/neurdy/pycharm/RV/out/', help='Path to output directory')
    parser.add_argument('--mode', type=str, default='train', help='Determines whether to backpropagate or not')
    parser.add_argument('--train_batch', type=int, default=32, help='Decides size of each training batch')
    parser.add_argument('--test_batch', type=int, default=1, help='Decides size of each val batch')
    parser.add_argument('--decay_rate', type=float, default=0.05, help='Rate at which the learning rate will be decayed')
    parser.add_argument('--decay_epoch', type=int, default=9999999, help='Decay the learning rate after every this many epochs (-1 means no lr decay)')
    parser.add_argument('--dropout', type=float, default=0.10, help='Continue training from saved model')
    parser.add_argument('--early_stop', type=int, default=50, help='Decide to stop early after this many epochs in which the validation loss increases (-1 means no early stopping)')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from saved model')

    opt = parser.parse_args()
    print(opt)

    if not os.path.isdir(os.path.join(opt.out_dir, 'models', opt.uni_id)):
        os.makedirs(os.path.join(opt.out_dir, 'models', opt.uni_id))
    if not os.path.isdir(os.path.join(opt.out_dir, 'results', opt.uni_id)):
        os.makedirs(os.path.join(opt.out_dir, 'results', opt.uni_id))

    if not os.path.isdir('/home/bayrakrg/Data/RV/model_prediction/{}/pred/'.format(opt.test_fold.rstrip('.txt'))):
        os.makedirs('/home/bayrakrg/Data/RV/model_prediction/{}/pred/'.format(opt.test_fold.rstrip('.txt')))
    if not os.path.isdir('/home/bayrakrg/Data/RV/model_prediction/{}/gt/'.format(opt.test_fold.rstrip('.txt'))):
        os.makedirs('/home/bayrakrg/Data/RV/model_prediction/{}/gt/'.format(opt.test_fold.rstrip('.txt')))

    if opt.mode == 'train':
        train_model(opt)
    elif opt.mode == 'test':
        test_model(opt)


if __name__ == '__main__':
    main()
