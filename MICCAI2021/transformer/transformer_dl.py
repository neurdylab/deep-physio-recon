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


def get_roi_len(dirs):
    roi_path = "/bigdata/HCP_1200/power+xifra/resting_min+prepro/bpf-ds/"
    roi_len = 0
    for dir in dirs:
        files = os.listdir(roi_path + dir)
        roi = loadmat(roi_path + dir + '/' + files[0])
        key = [x for x in list(roi.keys()) if x.startswith('roi_dat')][0]
        roi_len += roi[key].shape[1]
    return roi_len


def get_sub(path):
    fp = open(path, 'r')
    sublines = fp.readlines()
    hr_fold = []
    rv_fold = []
    roi_fold = []

    for subline in sublines:
        roi_fold.append(subline)
        hr_fold.append(subline.replace('.mat', '_hr_filt_ds.mat').replace('rois_', ''))
        rv_fold.append(subline.replace('.mat', '_rv_filt_ds.mat').replace('rois_', ''))
    fp.close()
    return roi_fold, hr_fold, rv_fold


def get_dictionary(fold):
    roi_path = os.path.join("/bigdata/HCP_1200/power+xifra/resting_min+prepro/bpf-ds/")
    hr_path = "/data/HR_filt_ds/"
    rv_path = "/data/RV_filt_ds/"
    fold_path = os.path.join("/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_files/", fold)
    roi_fold, hr_fold, rv_fold = get_sub(fold_path)

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

        # clust_list = os.listdir(roi_path)
        clust_list = ['schaefer', 'tractseg', 'tian', 'aan']
        if subject_id not in data:
            data[subject_id] = {clust_list[0]: [roi_path + clust_list[0] + '/' + d.rstrip('\n')],
                                clust_list[1]: [roi_path + clust_list[1] + '/' + d.rstrip('\n')],
                                clust_list[2]: [roi_path + clust_list[2] + '/' + d.rstrip('\n')],
                                clust_list[3]: [roi_path + clust_list[3] + '/' + d.rstrip('\n')],
                                'HR_filt_ds': [hr_path + hr_fold[i].rstrip('\n')],
                                'RV_filt_ds': [rv_path + rv_fold[i].rstrip('\n')]}
        else:
            if clust_list[0] and clust_list[1] and clust_list[2] and clust_list[3] not in data[subject_id]:
                data[subject_id][clust_list[0]] = [roi_path + clust_list[0] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[1]] = [roi_path + clust_list[1] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[2]] = [roi_path + clust_list[2] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[3]] = [roi_path + clust_list[3] + '/' + d.rstrip('\n')]
                data[subject_id]['HR_filt_ds'] = [hr_path + hr_fold[i].rstrip('\n')]
                data[subject_id]['RV_filt_ds'] = [rv_path + rv_fold[i].rstrip('\n')]
            else:
                data[subject_id][clust_list[0]].append(roi_path + clust_list[0] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[1]].append(roi_path + clust_list[1] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[2]].append(roi_path + clust_list[2] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[3]].append(roi_path + clust_list[3] + '/' + d.rstrip('\n'))
                data[subject_id]['HR_filt_ds'].append(hr_path + hr_fold[i].rstrip('\n'))
                data[subject_id]['RV_filt_ds'].append(rv_path + rv_fold[i].rstrip('\n'))

    # get the paths
    subj_excl = []
    for subj in data:
        paths = data[subj]['schaefer']
        # keep tract of the subjects that do not have all 4 scans
        if len(paths) == 4:
            subj_excl.append(subj)

        scan_order = []
        for path in paths:
            scan_order.append(path.lstrip('/bigdata/HCP_1200/power+xifra/resting_min+prepro/schaefer/bpf-ds/rois_').rstrip('.mat'))

        for k in data[subj]:
            new_paths = []
            for scan_id in scan_order:
                for path in data[subj][k]:
                    if scan_id in path:
                        new_paths.append(path)
                        break
            data[subj][k] = new_paths

    # print(list(data.keys())) # subject_ids
    return data


class data_to_tensor():
    """ From pytorch example"""

    def __init__(self, data, roi_list, transform=None):
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
            for i, val in enumerate(self.data[subj]['HR_filt_ds']):
                self.idx_list.append([subj, i])

        self.keys = list(self.data.keys())  # so, we just do it once
        self.transform = transform
        self.roi_list = roi_list

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # load on the fly
        single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
        single_paths = self.paths[self.idx_list[idx][0]]
        hr_path = single_paths['HR_filt_ds'][self.idx_list[idx][1]]
        schaefer = single[self.roi_list[0]][self.idx_list[idx][1]]['roi_dat'][0:600, :]
        tractseg = single[self.roi_list[1]][self.idx_list[idx][1]]['roi_dat'][0:600, :]
        tian = single[self.roi_list[2]][self.idx_list[idx][1]]['roi_dat'][0:600, :]
        aan = single[self.roi_list[3]][self.idx_list[idx][1]]['roi_dat'][0:600, :]
        hr = single['HR_filt_ds'][self.idx_list[idx][1]]['hr_filt_ds'][0:600, :]  # trimmed
        rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds'][0:600, :]  # trimmed

        # # TO DO multi-head

        hr_norm = (hr - hr.mean(axis=0)) / hr.std(axis=0)  # z-score normalization
        rv_norm = (rv - rv.mean(axis=0)) / rv.std(axis=0)  # z-score normalization
        schaefer_norm = (schaefer - schaefer.mean(axis=0)) / schaefer.std(axis=0)  # z-score normalization
        tractseg_norm = (tractseg - tractseg.mean(axis=0)) / tractseg.std(axis=0)  # z-score normalization
        tian_norm = (tian - tian.mean(axis=0)) / tian.std(axis=0)  # z-score normalization
        aan_norm = (aan - aan.mean(axis=0)) / aan.std(axis=0)  # z-score normalization
        roi_norm = np.hstack((schaefer_norm, tractseg_norm, tian_norm, aan_norm))

        # plt.subplot(611)
        # plt.plot(rv_norm, 'b')
        # plt.legend(['rv'])
        # plt.subplot(612)
        # plt.plot(hr_norm, 'b')
        # plt.legend(['hr'])
        # plt.subplot(613)
        # plt.plot(schaefer_norm[:, :15])
        # plt.legend(['schaefer'])
        # plt.subplot(614)
        # plt.plot(tractseg_norm[:, :15])
        # plt.legend(['tractseg'])
        # plt.subplot(615)
        # plt.plot(tian_norm)
        # plt.legend(['tian'])
        # plt.subplot(616)
        # plt.plot(aan_norm)
        # plt.legend(['aan'])
        # plt.show()

        # swap axis because
        # numpy: W x C
        # torch: C X W
        # roi_norm = roi_norm.transpose((1, 0))
        hr_norm = hr.squeeze()
        rv_norm = rv.squeeze()

        sample = {'roi': roi_norm, 'hr': hr_norm, 'rv': rv_norm}

        # if self.transform:
        #     sample = self.transform(sample)

        sample = ToTensor()(sample)
        sample['hr_path'] = hr_path
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        roi, hr, rv = sample['roi'], sample['hr'], sample['rv']

        return {'roi': torch.from_numpy(roi).type(torch.FloatTensor),
                'hr': torch.from_numpy(hr).type(torch.FloatTensor), 'rv': torch.from_numpy(rv).type(torch.FloatTensor)}
