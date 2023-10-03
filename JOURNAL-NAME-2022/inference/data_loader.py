import copy
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat, savemat
from scipy.stats import zscore
from skimage import io, transform


def get_dictionary(opt):

    roi_path = os.path.join(opt.input_dir, 'fMRI/')
    hr_path = os.path.join(opt.input_dir, 'HR/')
    rv_path = os.path.join(opt.input_dir, 'RV/')
    path = opt.input_dir

    rois, hrs, rvs = get_sub(path)

    data = {}
    for i, d in enumerate(rois):
        subdir_parts = rois[i].rstrip(".mat").split('_')
        subject_id = subdir_parts[1]

        clust_list = ['schaefer', 'tractseg', 'tian', 'aan']
        if subject_id not in data:
            data[subject_id] = {clust_list[0]: [roi_path + clust_list[0] + '/' + d.rstrip('\n')],
                                clust_list[1]: [roi_path + clust_list[1] + '/' + d.rstrip('\n')],
                                clust_list[2]: [roi_path + clust_list[2] + '/' + d.rstrip('\n')],
                                clust_list[3]: [roi_path + clust_list[3] + '/' + d.rstrip('\n')],
                                'HR_filt_ds': [hr_path + hrs[i].rstrip('\n')],
                                'RV_filt_ds': [rv_path + rvs[i].rstrip('\n')]}
        else:
            if clust_list[0] and clust_list[1] and clust_list[2] and clust_list[3] not in data[subject_id]:
                data[subject_id][clust_list[0]] = [roi_path + clust_list[0] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[1]] = [roi_path + clust_list[1] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[2]] = [roi_path + clust_list[2] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[3]] = [roi_path + clust_list[3] + '/' + d.rstrip('\n')]
                data[subject_id]['HR_filt_ds'] = [hr_path + hrs[i].rstrip('\n')]
                data[subject_id]['RV_filt_ds'] = [rv_path + rvs[i].rstrip('\n')]
            else:
                data[subject_id][clust_list[0]].append(roi_path + clust_list[0] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[1]].append(roi_path + clust_list[1] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[2]].append(roi_path + clust_list[2] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[3]].append(roi_path + clust_list[3] + '/' + d.rstrip('\n'))
                data[subject_id]['HR_filt_ds'].append(hr_path + hrs[i].rstrip('\n'))
                data[subject_id]['RV_filt_ds'].append(rv_path + rvs[i].rstrip('\n'))
    return data


def get_sub(path):

    files = os.listdir(path + '/fMRI/tractseg')
    roi = []
    hr = []
    rv = []

    for fname in files:
        roi.append(fname)
        # rest
        hr.append(fname.replace('.mat', '_hr_filt_ds.mat').replace('rois_', ''))
        rv.append(fname.replace('.mat', '_rv_filt_ds.mat').replace('rois_', ''))
    return roi, hr, rv


def get_roi_len(opt):
    roi_path = opt.input_dir + 'fMRI/'
    roi_len = 0
    dirs = os.listdir(roi_path)
    for dir in dirs:
        files = os.listdir(roi_path + dir)
        roi = loadmat(roi_path + dir + '/' + files[0])
        key = [x for x in list(roi.keys()) if x.startswith('roi_dat')][0]
        roi_len += roi[key].shape[1]
    return roi_len


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
        hr = single['HR_filt_ds'][self.idx_list[idx][1]]['hr_filt_ds'][0:600, :]  
        rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds'][0:600, :] 

        # z-score normalization on time axis
        hr_norm = zscore(hr, axis=1) 
        rv_norm = zscore(rv, axis=1) 

        schaefer_norm = zscore(schaefer, axis=0)
        tractseg_norm = zscore(tractseg, axis=0) 
        tian_norm = zscore(tian, axis=0)  
        aan_norm = zscore(aan, axis=0)  
        roi_norm = np.hstack((schaefer_norm, tractseg_norm, tian_norm, aan_norm))

        # swap axis because
        # numpy: W x C
        # torch: C X W
        roi_norm = roi_norm.transpose((1, 0))
        hr_norm = hr_norm.squeeze()
        rv_norm = rv_norm.squeeze()

        sample = {'roi': roi_norm, 'hr': hr_norm, 'rv': rv_norm}

        sample = ToTensor()(sample)
        sample['hr_path'] = hr_path.replace('_hr_filt_ds.mat', '').replace('JOURNAL-NAME-2022/example_data/HR/', '')
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        roi, hr, rv = sample['roi'], sample['hr'], sample['rv']

        return {'roi': torch.from_numpy(roi).type(torch.FloatTensor),
                'hr': torch.from_numpy(hr).type(torch.FloatTensor), 'rv': torch.from_numpy(rv).type(torch.FloatTensor)}
