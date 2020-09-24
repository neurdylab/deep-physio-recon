# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:55:33 2019

@author: salasja
"""


import numpy as np
import os
from scipy.io import loadmat

from scipy import signal

import matplotlib.pyplot as plt
import pandas as pd

def zscore( v ):
    
    return (v-v.mean(axis=1, keepdims=True))/v.std(axis=1, keepdims=True)


def reshaper( v ):
    xr=np.reshape(v, [v.shape[0], v.shape[1]*v.shape[2]], order="F")
    return xr

def broadcaster(  v , sh):
    v_rs=np.reshape( v, [v.shape[0],1,1])

    tile_v=np.tile( v_rs  , sh )    
    return tile_v

def zscore_glob(v):
    #calculates zscore across the entire scans, not per ROI
    
    #convert array to shape (scans, values)
    xr=reshaper( v )
    #get the mean over all ROIs
    means=np.mean(xr, axis=1, keepdims=False)
    #get sd over all ROIs
    std=np.std(xr, axis=1, keepdims=False)    
    sh=[1,v.shape[1], v.shape[2]]
    #make array with repeated values
    tile_means=broadcaster(means, sh)    
    tile_std=broadcaster(std, sh)
    #calculate z score
    return (v-tile_means )/  tile_std 

def filter_rvt(y_in):
    T=0.72
    fcut=0.2 # upper cutoff frequency in Hz
    fs=1.0/T
    b, a = signal.butter(4, fcut, 'low', analog=False, fs=fs)
    y1 = signal.filtfilt(b, a, y_in, axis=1)
    b, a = signal.butter(4, 0.01, 'high', analog=False, fs=fs)
    y_out = signal.filtfilt(b, a, y1 , axis=1)
    return y_out , b ,a

def corr(y1, y2):    
    return np.corrcoef(y1,y=y2, rowvar=True)

#This function takes a signal of shape (nsamples, ntime, nchannels)
    #and returns an array of shape (N, Lwin, nchannels) with all possible windows
    #of size Lwin with stride 1
    
def rolling_slicer( Xin , Lwin=60 ):
    stride=1
    norig=Xin.shape[0]
    #splitter
    offset=0
    sigL=Xin.shape[1]-offset
    
    nsamp=(sigL-Lwin+1)*norig
    #resid=(sigL*norig)%Lwin
    
    nchannel=Xin.shape[2]
    n=0
    arr_new=np.zeros( [nsamp , Lwin, nchannel ] )
    arr_new[:,:,:]=np.nan
    
    #total_points_old=np.product( Xin.shape )
    #total_points_new=np.product( arr_new.shape )
    
    for k in range(norig):
        for t in range(offset, Xin.shape[1] -Lwin+1, stride ):        
            sl=Xin[k,t:t+Lwin,:]
            arr_new[n,:,:]=sl
            #assert sl.shape[0] == Lwin
            
            n=n+1        
            
    assert np.sum( np.isnan(arr_new) ) ==0
    return arr_new
    
#%% Data set selector

data_run="sliced_lightprep_90"

if data_run=="sliced_10":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_10_ROI"
    folder_root=r"C:\Users\Public\Documents\data_102819"
    file_name_fmri="fmri_all_roi.npy"
    file_name_rvt="rv_all_roi.npy"
    file_scan_info="files_users_summary_10_ROI.csv"
    filtering=False
    downsampling=False
    
    
elif data_run=="sliced_90":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_90_ROI"
    folder_root=r"C:\Users\salasja\Documents\MATLAB\HCP_data"
    file_name_fmri="fmri_all.npy"
    file_name_rvt="rv_all.npy"
    file_scan_info="files_users_summary_90_ROI.csv"
    filtering=True
    downsampling=True    

elif data_run=="sliced_large_42":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_large_42_ROI"
    folder_root=r"C:\Users\Public\Documents\data_large"
    file_name_fmri="fmri_all_large_42.npy"
    file_name_rvt="rv_all_large_42.npy"
    file_scan_info="files_users_summary_large_42.csv"
    filtering=False
    downsampling=False

elif data_run=="sliced_lightprep_42":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_lightprep_42_ROI"
    folder_root=r"C:\Users\Public\Documents\data_large"
    file_name_fmri="fmri_all_lightprep_42.npy"
    file_name_rvt="rv_all_large_42.npy"
    file_scan_info="files_users_summary_lightprep_42.csv"
    filtering=False
    downsampling=False
    
elif data_run=="sliced_lightprep_90":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_lightprep_90_ROI"
    folder_root=r"C:\Users\Public\Documents\data_large"
    file_name_fmri="fmri_all_lightprep_90.npy"
    file_name_rvt="rv_all_large_90.npy"
    file_scan_info="files_users_summary_lightprep_90.csv"
    filtering=False
    downsampling=False
    
    
elif data_run=="sliced_large_90":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_large_90_ROI"
    folder_root=r"C:\Users\Public\Documents\data_large"
    file_name_fmri="fmri_all_large_90.npy"
    file_name_rvt="rv_all_large_90.npy"
    file_scan_info="files_users_summary_large_90.csv"
    filtering=False
    downsampling=False    


elif data_run=="sliced_large_268":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_large_268_ROI"
    folder_root=r"C:\Users\Public\Documents\data_031120"
    file_name_fmri="fmri_all_large_268.npy"
    file_name_rvt="rv_all_large_90.npy"
    file_scan_info="files_users_summary_large_268.csv"
    filtering=False
    downsampling=False    

elif data_run=="sliced_parcel_90":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_parcel_90_ROI"
    folder_root=r"C:\Users\Public\Documents\data_031120"
    file_name_fmri="fmri_all_parcel_90.npy"
    file_name_rvt="rv_all_large_90.npy"
    file_scan_info="files_users_summary_parcel_90.csv"
    filtering=False
    downsampling=False

elif data_run=="sliced_parcel_42":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_parcel_42_ROI"
    folder_root=r"C:\Users\Public\Documents\data_031120"
    file_name_fmri="fmri_all_parcel_42.npy"
    file_name_rvt="rv_all_large_42.npy"
    file_scan_info="files_users_summary_parcel_42.csv"
    filtering=False
    downsampling=False

elif data_run=="sliced_parcel_10":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_parcel_10_ROI"
    folder_root=r"C:\Users\Public\Documents\data_031120"
    file_name_fmri="fmri_all_parcel_10.npy"
    file_name_rvt="rv_all_large_42.npy"
    file_scan_info="files_users_summary_parcel_10.csv"
    filtering=False
    downsampling=False

    
elif data_run=="sliced_large_92":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_large_92_ROI"
    folder_root=r"C:\Users\Public\Documents\data_large"
    file_name_fmri="fmri_all_wmcsf_92.npy"
    file_name_rvt="rv_all_large_90.npy"
    file_scan_info="files_users_summary_large_90.csv"
    filtering=False
    downsampling=False    

elif data_run=="sliced_meannorm_90":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_meannorm_90_ROI"
    folder_root=r"C:\Users\Public\Documents\data_meannorm"
    file_name_fmri="fmri_all_meannorm_90.npy"
    file_name_rvt="rv_all_meannorm_90.npy"
    file_scan_info="files_users_summary_meannorm_90.csv"
    filtering=False
    downsampling=False    


elif data_run=="sliced_headmotion_90":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_headmotion_90_ROI"
    folder_root=r"C:\Users\Public\Documents\data_headmotion"
    file_name_fmri="fmri_all_headmotion_90.npy"
    file_name_rvt="rv_all_headmotion_90.npy"
    file_scan_info="files_users_summary_headmotion_90.csv"
    filtering=False
    downsampling=False    
    
elif data_run=="sliced_downsamp_90":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_downsamp_90_ROI"
    folder_root=r"C:\Users\Public\Documents\data_downsamp"
    file_name_fmri="fmri_all_downsamp_90.npy"
    file_name_rvt="rv_all_downsamp_90.npy"
    file_scan_info="files_users_summary_downsamp_90.csv"
    filtering=False
    downsampling=False    
    
elif data_run=="sliced_downsampfilt_90":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_downsampfilt_90_ROI"
    folder_root=r"C:\Users\Public\Documents\data_downsampfilt"
    file_name_fmri="fmri_all_downsampfilt_90.npy"
    file_name_rvt="rv_all_downsampfilt_90.npy"
    file_scan_info="files_users_summary_downsampfilt_90.csv"
    filtering=False
    downsampling=False            

elif data_run=="sliced_downsamp2_90":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_downsamp2_90_ROI"
    folder_root=r"C:\Users\Public\Documents\data_downsamp2"
    file_name_fmri="fmri_all_downsamp2_90.npy"
    file_name_rvt="rv_all_downsamp2_90.npy"
    file_scan_info="files_users_summary_downsamp2_90.csv"
    filtering=False
    downsampling=False    
    
elif data_run=="sliced_downsampfilt2_90":
    folder_export=r"C:\Users\salasja\Documents\MRI_files\datasets\data_downsampfilt2_90_ROI"
    folder_root=r"C:\Users\Public\Documents\data_downsampfilt2"
    file_name_fmri="fmri_all_downsampfilt2_90.npy"
    file_name_rvt="rv_all_downsampfilt2_90.npy"
    file_scan_info="files_users_summary_downsampfilt2_90.csv"
    filtering=False
    downsampling=False            
    
    
else:
    raise Exception(" Bad data set selected ")
        
#%% Get information on scans
    
df_scan_info=pd.read_csv( os.path.join( folder_root , file_scan_info) , index_col=0  )    
df_scan_info.to_csv( os.path.join( folder_export, file_scan_info )  )

#%% Load fMRI data


file_in_fmri=os.path.join( folder_root, file_name_fmri  )
fmri_all=np.load( file_in_fmri )


#%% Load RVT data


file_in_rvt=os.path.join( folder_root, file_name_rvt  )
rvt_all =np.load( file_in_rvt  )

#rvt_all, _,_ =filter_rvt(rvt_all)
rvt_all=rvt_all.reshape(  (rvt_all.shape[0], rvt_all.shape[1], 1) )


#%% Filter stage

if filtering==True:
    fmri_all,_,_=filter_rvt( fmri_all )
    rvt_all, _,_ =filter_rvt(rvt_all)

#%% Dowsample
    
if downsampling==True:
    rvt_all=rvt_all[ :,::2 ]
    fmri_all=fmri_all[:,::2,: ]
    

#%% Limit the ammount of data 

fmri_all=fmri_all[:,:,:]
rvt_all=rvt_all[:,:,:]


#%% Get zscore versions

fmri_z= zscore_glob( fmri_all)
rvt_z = zscore( rvt_all )

#%%

        
#%% Copy and reshape        
Lwin=64

#slice and copy the X data
fmri_sliced=rolling_slicer( fmri_all, Lwin=Lwin )
fmri_sliced_z=rolling_slicer( fmri_z , Lwin=Lwin )



#repeat for y
rvt_sliced=rolling_slicer( rvt_all, Lwin=Lwin )    
rvt_sliced=rvt_sliced.reshape( ( rvt_sliced.shape[0], rvt_sliced.shape[1] )  )        

#repeat for y z-score
rvt_sliced_z=rolling_slicer( rvt_z, Lwin=Lwin )    
rvt_sliced_z=rvt_sliced_z.reshape( ( rvt_sliced_z.shape[0], rvt_sliced_z.shape[1] )  )       

# make a version with a single predicted output with the middle point
rvt_single=np.zeros( ( rvt_sliced.shape[0] , 1   ) )
rvt_single[:,:]=np.nan 
rvt_single[:,0]=rvt_sliced[:, rvt_sliced.shape[1]//2  ]

# make a zscore version for the single predicted output with the middle point
rvt_single_z=np.zeros( ( rvt_sliced_z.shape[0] , 1   ) )
rvt_single_z[:,:]=np.nan 
rvt_single_z[:,0]=rvt_sliced_z[:, rvt_sliced_z.shape[1]//2  ]


#make a version that predicst the last value  only
rvt_single_last=np.zeros( ( rvt_sliced.shape[0] , 1   ) )
rvt_single_last[:,:]=np.nan 
rvt_single_last[:,0]=rvt_sliced[:, rvt_sliced.shape[1] -1 ]

# make a zscore version for the predicted last value
rvt_single_z_last=np.zeros( ( rvt_sliced_z.shape[0] , 1   ) )
rvt_single_z_last[:,:]=np.nan 
rvt_single_z_last[:,0]=rvt_sliced_z[:, rvt_sliced_z.shape[1] -1  ]



#%% Export

#fmri full
file_out_fmri_full=os.path.join( folder_export, "fmri_full.npy"  )
np.save( file_out_fmri_full, fmri_all    )

#RVT full
file_out_rv_full=os.path.join( folder_export, "rv_full.npy"  )
np.save( file_out_rv_full, rvt_all   )

#fmri zscore
file_out_fmri_full_z=os.path.join( folder_export, "fmri_full_zscore_glob.npy"  )
np.save( file_out_fmri_full_z, fmri_z    )

#RV zscore
file_out_rv_full_z=os.path.join( folder_export, "rv_full_zscore.npy"  )
np.save( file_out_rv_full_z, rvt_z   )


#fmri sliced
file_out_fmri=os.path.join( folder_export, "fmri_sliced_%d.npy"%Lwin  )
np.save( file_out_fmri, fmri_sliced    )

# fmri Z score
file_out_fmri_z=os.path.join( folder_export, "fmri_sliced_zscore_glob_%d.npy"%Lwin  )
np.save( file_out_fmri_z, fmri_sliced_z    )

# RV
file_out_rv=os.path.join( folder_export, "rv_sliced_%d.npy"%Lwin  )
np.save( file_out_rv , rvt_sliced    )

# RV zscore
file_out_rv_z=os.path.join( folder_export, "rv_sliced_zscore_%d.npy"%Lwin  )
np.save( file_out_rv_z , rvt_sliced_z    )


# RV single output per piece
file_out_single=os.path.join( folder_export, "rv_single_point_%d.npy"%Lwin  )
np.save( file_out_single, rvt_single    )

# RV zscore single output per piece
file_out_single_z=os.path.join( folder_export, "rv_single_point_zscore_%d.npy"%Lwin  )
np.save( file_out_single_z, rvt_single_z    )


# RV single -last output per piece
file_out_single_last=os.path.join( folder_export, "rv_single_point_last_%d.npy"%Lwin  )
np.save( file_out_single_last, rvt_single_last    )

# RV zscore single -last output per piece
file_out_single_z_last=os.path.join( folder_export, "rv_single_point_zscore_last_%d.npy"%Lwin  )
np.save( file_out_single_z_last, rvt_single_z_last    )



#%%  Create arrays to locate the samples

num_slices=rvt_sliced.shape[0]

ntot= np.product( rvt_all.shape)  
L=rvt_all.shape[1]
Xflat1=np.arange( ntot  )

Xlocs=Xflat1.reshape( rvt_all.shape[0] , rvt_all.shape[1], 1  )

locator_slices=rolling_slicer( Xlocs, Lwin=Lwin ).astype(int)

#%% make locator for y single samples

locator_slices_y=locator_slices.reshape( ( locator_slices.shape[0], locator_slices.shape[1] )  )
locator_slices_single=np.zeros( ( locator_slices_y.shape[0] , 1   ) , dtype=int )
locator_slices_single[:,0]=locator_slices_y[:, locator_slices_y.shape[1]//2  ]


#%%
# Locations of samples inside the windows
file_out_locator=os.path.join( folder_export, "locator_slices_%d.npy"%Lwin  )
np.save( file_out_locator , locator_slices    )

#%%
# Shape of the original RVT array
file_out_shape=os.path.join( folder_export, "shape_original_data_%d.npy"%Lwin  )
np.save( file_out_shape , np.array( Xlocs.shape )   )

#%%

#Function to retrieve the data points
#The input is the sample number to inquire and the time point on the sample
#It returns the scan number and the tiem point within the scan 
def get_datapoint_source( sample_number, time_point , locator_slices_arr, nx_orig, ny_orig  ):
    #print("Finding ", sample_number, time_point  , nx_orig, ny_orig)
    
    #get the index of the requested sample and time point for channel 0
    z=locator_slices_arr[sample_number,time_point,0]    
        
    #print("Index global ", z)
    
    #convert the index to the scan and time point number requested
    scan_number, time_point_orig =np.unravel_index( z, (nx_orig , ny_orig ))    
    #print( scan_number, time_point_orig  )
    return scan_number, time_point_orig

#load the array, with the locations of the points, and array shapes from their files
def retrieve_data_sample_location( folder_in, Lwin ):
    # Locations of samples inside the windows
    file_out_locator=os.path.join( folder_in, "locator_slices_%d.npy"%Lwin  )
    locator_slices=np.load( file_out_locator      )
    #load array with original data shape information
    file_out_shape=os.path.join( folder_in, "shape_original_data_%d.npy"%Lwin  )
    shape_original=np.load( file_out_shape    )
    
    return locator_slices, shape_original

locator_slices_in, shape_original_in=retrieve_data_sample_location( folder_export, Lwin )
nx=shape_original_in[0]
ny=shape_original_in[1]
example_scan, example_point=get_datapoint_source(  0, Lwin//2 , locator_slices_in, nx , ny )
#example_scan, example_point=get_datapoint_source(  0,63 , locator_slices_single, nx , ny )

#%% Output the middle point of each slice, used for the RVT prediction

points_y_trial=[ get_datapoint_source(  k, Lwin//2 , locator_slices_in, nx , ny ) for k in range(num_slices) ]
scans_arr, locs_y_arr= zip(*points_y_trial)
df_y_points= pd.DataFrame( {"scan_num": scans_arr, "time_point": locs_y_arr } )

df_y_points.to_csv( os.path.join( folder_export, "locations_prediction_%s.csv"%(data_run) )  )

#%% Output the starting point of each slice

points_x_trial=[ get_datapoint_source(  k, 0 , locator_slices_in, nx , ny ) for k in range(num_slices) ]
scans_arr, locs_x_arr= zip(*points_x_trial)
df_x_points= pd.DataFrame( {"scan_num": scans_arr, "time_point": locs_x_arr } )
df_x_points.to_csv(  os.path.join(folder_export,"locations_slice_start_%s.csv"%(data_run) ) )

#%% Output the last point of each slice

points_z_trial=[ get_datapoint_source(  k, Lwin-1 , locator_slices_in, nx , ny ) for k in range(num_slices) ]
scans_arr, locs_z_arr= zip(*points_z_trial)
df_z_points= pd.DataFrame( {"scan_num": scans_arr, "time_point": locs_z_arr } )

df_z_points.to_csv( os.path.join( folder_export, "locations_prediction_last_%s.csv"%(data_run) )  )




