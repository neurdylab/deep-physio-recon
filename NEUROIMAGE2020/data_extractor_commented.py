# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:13:12 2019

@author: salasja
"""


import numpy as np
import pandas as pd

from scipy.io import loadmat

import matplotlib.pyplot as plt

import os
import glob
import re

# Folder with Matlab files 

folder_root=r"C:\Users\Public\Documents\data_large"
num_roi=42
#Folder fMRI data 
folder_fmri=r"data_%d"%num_roi

#Folder physiological data 
folder_phys=r"RV_filt_ds"

folder_fmri_full=os.path.join( folder_root, folder_fmri  )
folder_phys_full=os.path.join( folder_root, folder_phys  )
#

#Parse the file name to get the RL or LR  
def get_run_code(x):
    m=re.search( r"_(LR|RL)", x  )
    if m:
        r=m.group(1)
    else :
        r=""
    return r

#Parse the file name to get the trial number 
def get_trial_code(x):
    m=re.search( r"_REST(\d{1})", x  )
    if m:
        r=m.group(1)
    else :
        r=""
    return r

#Extractng the file names from the folder 
def extract_data_folder( folder , file_col="file" ):
    
    file_list=glob.glob( folder+"/*"  )
    print(folder)
    users_tup=[ (l, re.search( r"_(\d{6})_", l ).group(1) ) for l in file_list   ]
    dic_files= dict(users_tup) 
    
    df=pd.DataFrame.from_dict(dic_files, orient="index")
    df=df.rename( columns={0: "user"} )
    df[file_col]=list( df.index )
    
    df=df.reset_index(drop=True)       
    df["runcode"]=df[file_col].apply(lambda x: get_run_code(x)  )
    df["trialcode"]=df[file_col].apply(lambda x: get_trial_code(x)  )
    return df

#%% Compile a dataframe with infomation on the available files 

df_fmri=extract_data_folder( folder_fmri_full, file_col="file_fmri" )
df_phys=extract_data_folder( folder_phys_full, file_col="file_phys" )

#%% Merge the fMRI and Physiological data 
df_files=pd.merge( df_fmri, df_phys, on=["user", "runcode","trialcode"] )


#%%
num_points=600

num_files=len(df_files)

#Create a name for the data set 
label_out="large_%d"%num_roi

#Initialize the arrays for storing the data 
fmri_all=np.zeros( [ num_files, num_points, num_roi ] )

rv_all=np.zeros( [ num_files, num_points  ] )

#Open up the files and extract the data 
for k, row in df_files.iterrows():

    print("Reading file" , k)
    file_fmri=row["file_fmri"]
    file_phys=row["file_phys"]
    file_content=loadmat(file_phys)
    rv=file_content["rv_filt_ds"].flatten()
    fmri=loadmat(file_fmri)["roi_dat"]    
    fmri_all[ k, :,: ]=fmri
    
    rv_all[k,:]=rv
    

#%% Save the extracted data from the MATLAB files into numpy array files 
file_out_fmri=os.path.join( folder_root, "fmri_all_%s"%label_out  )
np.save( file_out_fmri, fmri_all  )

file_out_rv=os.path.join( folder_root, "rv_all_%s"%label_out  )
np.save( file_out_rv, rv_all  )



#%% Export the dataframe with the infomation on the scans 
df_files["scan_number"]=list(df_files.index)
df_files.to_csv( os.path.join( folder_root ,"files_users_summary_%s.csv"%label_out ) )
