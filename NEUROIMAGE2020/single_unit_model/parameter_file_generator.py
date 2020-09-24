# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:41:47 2019

@author: salasja
"""


import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

import seaborn as sns



def write_parameter_file(  file_log, dict_p ):
    
    with open(file_log, "w") as fid:        
        for k,v in dict_p.items():
            line="%s \t %s\n"%(k,v)
            fid.write(line)
        fid.write("\n")
        
    return 0

dic_params={
    "filters":	20,
    "kernel_size":	3,
    "depth_multiplier":	1,
    "epochs":	5,
    "model_name": "convnet_deep",
    "comment":	"single layer",
    "loss":	"mse",
    "data_set":	"sliced_large_zscore_glob_90",
    "seed_number":	50,
    "learning_rate": 0.001,
}


file_scan_info=r"C:/Users/salasja/Documents/MRI_files/datasets/data_large_90_ROI/files_users_summary_large_90.csv"
df_scans=pd.read_csv(file_scan_info, index_col=0)

user_list=df_scans["user"].unique()
nusers=len(user_list)

#raise Exception("Stopping")

num_iters=nusers

for k in range(num_iters):
    user=user_list[k]
    df_sel=df_scans[df_scans["user"]==user ]
    scans_sel=list( df_sel["scan_number"] )
    scan_str=" ".join([ "%d"%s for s in  scans_sel])
    
    print( "User ", user , "Scans: " , scan_str )    
    
    
    folder_out="run%d"%k
    try:
        os.mkdir(folder_out)
    except FileExistsError:
        pass
    parameter_file=os.path.join(  folder_out, "hyper_parameters.dat"  )
    file_model="model_convnet.py"
    os.system('copy %s %s'%(file_model, os.path.join( folder_out ,file_model ) ) )
    
    write_parameter_file(  parameter_file, dic_params )
    
    file_scans=os.path.join( folder_out, "scans_keep.txt"  )
    with open( file_scans, "w"  ) as fid_keep:
        fid_keep.write("%s \n"%scan_str)
    
    