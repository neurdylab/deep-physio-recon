import copy
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from skimage import io, transform


def get_sub(path):
    fp = open(path, 'r')
    sublines = fp.readlines()
    roi_fold = []
    rv_fold = []

    for subline in sublines:
        roi_fold.append(subline)
        rv_info = subline.replace('rois', 'RV_filtds')
        rv_fold.append(rv_info)
    fp.close()
    return roi_fold, rv_fold


def get_dictionary(fold):
    roi_path = os.path.join("/home/bayrakrg/Data/RV/neuroimg_data/")
    rv_path = "/home/bayrakrg/Data/RV/RV_filt_ds"
    fold_path = os.path.join("/home/bayrakrg/neurdy/pycharm/RV/k_fold_files/", fold)
    roi_fold, rv_fold = get_sub(fold_path)

    # # LOOK AT YOUR DATA
    # x = os.path.join(rv_path, 'RV_filtds_983773_3T_rfMRI_REST1_RL.mat')
    # rv = loadmat(x)
    # rv.keys()
    # type(ROI['roi_dat']), ROI['roi_dat'].shape
    # type(ROI['roi_inds']), ROI['roi_inds'].shape
    # type(rv['rv_filt_ds']), rv['rv_filt_ds'].shape
    # type(rv['tax']), rv['tax'].shape

    data = {}
    for i, d in enumerate(roi_fold):
        subdir_parts = roi_fold[i].rstrip(".mat").split('_')
        subject_id = subdir_parts[1]
        # print("{}".format(subject_id))

        clust_list = os.listdir(roi_path)
        for clust in clust_list:
            if subject_id not in data:
                data[subject_id] = {clust: [roi_path + clust + '/' + d.rstrip('\n')],
                                    'RV_filt_ds': [rv_path + '/' + rv_fold[i].rstrip('\n')]}
            else:
                if clust not in data[subject_id]:
                    data[subject_id][clust] = [roi_path + clust + '/' + d.rstrip('\n')]
                    data[subject_id]['RV_filt_ds'] = [rv_path + '/' + rv_fold[i].rstrip('\n')]
                else:
                    data[subject_id][clust].append(roi_path + clust + '/' + d.rstrip('\n'))
                    data[subject_id]['RV_filt_ds'].append(rv_path + '/' + rv_fold[i].rstrip('\n'))

    for subj in data:
        paths = data[subj]['findlab90']
        scan_order = []
        for path in paths:
            scan_order.append(path.lstrip('/home/bayrakrg/Data/RV/neuroimg_data/findlab90/rois_'))

        for k in data[subj]:
            new_paths = []
            for scan_id in scan_order:
                for path in data[subj][k]:
                    if path.endswith(scan_id):
                        new_paths.append(path)
                        break
            data[subj][k] = new_paths


    # print(list(data.keys())) # subject_ids
    return data


class data_to_tensor():
    """ From pytorch example"""

    def __init__(self, data, roi_clust, transform=None):
        # go through all the data and load them in
        # start with one worker
        # as soon as I pass to the data loader it is gonna create a copy depending on the workers (threads)
        # copy of the data for each worker (for more heavy duty data)
        # random data augmentation usually needs multiple workers
        self.data = copy.deepcopy(data)
        self.paths = copy.deepcopy(data)
        self.idx_list = []

        for subj in self.data.keys():
            for folder in self.data[subj]:
                for i, val in enumerate(self.data[subj][folder]):
                    self.data[subj][folder][i] = loadmat(val)

        # make sure in get_item that we see all data by
        for subj in self.data.keys():
            for i, val in enumerate(self.data[subj]['RV_filt_ds']):
                self.idx_list.append([subj, i])

        self.keys = list(self.data.keys())  # so, we just do it once
        self.transform = transform
        self.roi_clust = roi_clust

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # load on the fly
        single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
        single_paths = self.paths[self.idx_list[idx][0]]
        rv_path = single_paths['RV_filt_ds'][self.idx_list[idx][1]]
        roi = single[self.roi_clust][self.idx_list[idx][1]]['roi_dat']
        rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds']

        # # multi-resolution
        # roi_10 = single['clust10'][self.idx_list[idx][1]]['roi_dat']
        # roi_42 = single['clust42'][self.idx_list[idx][1]]['roi_dat']
        # roi_90 = single['clust90'][self.idx_list[idx][1]]['roi_dat']
        #
        # roi = np.concatenate([roi_10, roi_42, roi_90], axis=1)

        # # MIXUP data augmentation
        # idx2 = random.randrange(len(self.idx_list))
        # single2 = self.data[self.idx_list[idx2][0]]  # passing the subject string to get the other dictionary
        # roi2 = single2[self.roi_clust][self.idx_list[idx2][1]]['roi_dat']
        # rv2 = single2['RV_filt_ds'][self.idx_list[idx2][1]]['rv_filt_ds']
        #
        # t = random.uniform(.75, .95)
        # t = max(t, (1-t))
        # roi = np.sum([roi1 * t, roi2 * (1-t)], axis=0)
        # rv = rv1 * t + rv2 * (1-t)

        # rv_global_norm = (all_rv - all_rv.mean(axis=0)) / all_rv.std(axis=0)  # global normalization
        # roi_global_norm = (all_roi - all_roi.mean(axis=0)) / all_roi.std(axis=0)  # global normalization

        rv_norm = (rv - rv.mean(axis=0)) / rv.std(axis=0)  # z-score normalization
        roi_norm = (roi - roi.mean(axis=0)) / roi.std(axis=0)  # z-score normalization

        # plt.plot(rv)
        # plt.plot(rv1)  # one of the rois
        # plt.legend(['rv', 'rv1'])
        # plt.show()
        #
        # plt.plot(roi[:, 5])
        # plt.plot(roi1[:, 5])  # one of the rois
        # plt.legend(['roi', 'roi1'])
        # plt.show()

        # swap axis because
        # numpy: W x C
        # torch: C X W
        roi_norm = roi_norm.transpose((1, 0))
        rv_norm = rv_norm.squeeze()

        sample = {'roi': roi_norm, 'rv': rv_norm}

        if self.transform:
            sample = self.transform(sample)

        sample = ToTensor()(sample)
        sample['rv_path'] = rv_path
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        roi, rv = sample['roi'], sample['rv']

        return {'roi': torch.from_numpy(roi).type(torch.FloatTensor), 'rv': torch.from_numpy(rv).type(torch.FloatTensor)}
