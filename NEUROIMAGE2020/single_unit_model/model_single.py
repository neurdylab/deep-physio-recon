# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:04:12 2019

@author: salasja
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:29:48 2019

@author: salasja
"""


import time

import numpy as np
import pandas as pd
import os

from scipy import signal

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SeparableConv1D
from keras.layers import Flatten
from keras.layers import Dropout

from keras.layers import MaxPooling1D

#%%

from keras.layers import Dropout
from keras import optimizers
#%%

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(21015183748)

#%%



def filter_rvt(y_in):
    T=0.72
    fcut=0.2 # upper cutoff frequency in Hz
    fs=1.0/T
    b, a = signal.butter(4, fcut, 'low', analog=False, fs=fs)
    y1 = signal.filtfilt(b, a, y_in, axis=1)
    b, a = signal.butter(4, 0.01, 'high', analog=False, fs=fs)
    y_out = signal.filtfilt(b, a, y1 , axis=1)
    return y_out , b ,a

def zscore( v ):
    
    return (v-v.mean(axis=1, keepdims=True))/v.std(axis=1, keepdims=True)



def get_datapoint_source( sample_number, time_point , locator_slices_arr, nx_orig, ny_orig  ):
    #print("Finding ", sample_number, time_point  , nx_orig, ny_orig)
    z=locator_slices_arr[sample_number,time_point,0]
    #print("Index global ", z)
    scan_number, time_point_orig =np.unravel_index( z, (nx_orig , ny_orig ))    
    #print( scan_number, time_point_orig  )
    return scan_number, time_point_orig

def retrieve_data_sample_location( folder_root, Lwin ):
    # Locations of samples inside the windows
    file_out_locator=os.path.join( folder_root, "locator_slices_%d.npy"%Lwin  )
    locator_slices=np.load( file_out_locator      )

    file_out_shape=os.path.join( folder_root, "shape_original_data_%d.npy"%Lwin  )
    shape_original=np.load( file_out_shape    )
    
    return locator_slices, shape_original

#%%
import datetime
    
def is_integer(x):    
    try:
        xtrial=int(x)            
        valid=True
    except ValueError:
        valid=False    
    return valid

def is_float(x):
    try:
        trial= float(x)+1.0
        valid=True
    except ValueError:
        valid=False    
    return valid

def converter_input(x):
    #tries to convert to integer, if not possible then float, or else keeps 
    # it as a string  
    if is_integer(x)==True:        
        #convert to integer if valid integer
        res=int(x)
    elif is_float(x)==True:
        #convert to float if valid float        
        res=float(x)
    else:
        #keep as string
        res=x
    return res


def read_params( file_params , text_lines=["comment"] ):
    with open(file_params, "r") as fid:
        lines=fid.readlines()        
    #get rid of the empty spaces     
    lines_clean=[l.split() for l in lines if len(l.split()) >=1  ]  
    #get rid of lines marked with the comment symbol
    lines_filtered=[ l for l in lines_clean if l[0][0] != "#"  ]        
    dic_params={}                    
    for l in lines_filtered:    
        k=l[0]
        if k in text_lines:
            #this line contains commentary that must be kept as it is, with all words
            #we join the words to put the commentary onto a single string
            v=" ".join(l[1:])
        else:
            #we take the first token (word or number) next to the keyword as a value
            v=l[1]
        #we convert to integer, float or keep as string, in that order of priority    
        dic_params[k]=converter_input(v)        
    return dic_params 

def write_parameter_log(  file_log, dict_p ):
    now=datetime.datetime.now()
    with open(file_log, "w") as fid:        
        time_str=now.strftime("%A %B %d %Y  %H:%M:%S")
        
        fid.write( "Start time: "+ time_str + "\n\n\n" )
        fid.write("Parameters used : \n \n")
        for k,v in dict_p.items():
            line="%s \t %s\n"%(k,v)
            fid.write(line)
        fid.write("\n")
        
    return 0


def overwrite_params( dic_in, dic_out ):
    #overwrite dic_out with whatever value is in dic_in
    for k,v in dic_in.items():
        try:
            val=dic_out[k]
            dic_out[k]=dic_in[k]
        except KeyError:            
            raise KeyError("The parameter %s in the input file is not defined in the code"%k)
    return 0

def read_scans_keep_test( file_scans_keep ):
    with open(file_scans_keep, "r") as fid_keep:
        line=fid_keep.readline()
        list_s=[ int(l) for l in line.split() ]
        
    return list_s

#%% parameters defaults
    
file_params="hyper_parameters.dat"
 
#read the input parameters
dic_params_in=read_params(file_params)



#defaults
dic_params={ "filters": 10, 
            "kernel_size": 63, 
            "depth_multiplier": 1 , 
            "epochs": 15, 
            "comment": "commentary for run", 
            "loss": "mse", 
            "data_set": "sliced_90",
            "seed_number": 50 ,
            "rem_scan": 0,
            "learning_rate": 0.001,
            "model_name": "convnet"}

overwrite_params( dic_params_in, dic_params )
learning_rate=dic_params["learning_rate"]
filters=dic_params["filters"]
kernel=dic_params["kernel_size"]
depth=dic_params["depth_multiplier"]
epochs=dic_params["epochs"]
comment=dic_params["comment"]
loss=dic_params["loss"]
data_run=dic_params["data_set"]
seed_number=dic_params["seed_number"]
model_name=dic_params["model_name"]

#write log file 
file_log="log_file.out"
write_parameter_log(  file_log, dic_params )



#%% Load fMRI data
folder_root=""

file_in_fmri=""
file_in_rvt=""



if data_run =="full_90":   
    folder_root=r"C:\Users\salasja\Documents\MRI_files\datasets\data_90_ROI"
    file_in_fmri=os.path.join( folder_root, "fmri_full.npy"  )
    file_in_rvt=os.path.join( folder_root, "rv_full.npy"  )    
    
elif data_run =="sliced_90":   
    folder_root=r"C:\Users\salasja\Documents\MRI_files\datasets\data_90_ROI"
    file_in_fmri=os.path.join( folder_root, "fmri_sliced_64.npy"  )
    file_in_rvt=os.path.join( folder_root, "rv_single_point_64.npy"  )     
    locator_slices_in, shape_original_in=retrieve_data_sample_location( folder_root, 64 )
    
elif data_run =="sliced_zscore_90":   
    folder_root=r"C:\Users\salasja\Documents\MRI_files\datasets\data_90_ROI"
    file_in_fmri=os.path.join( folder_root, "fmri_sliced_zscore_64.npy"  )
    file_in_rvt=os.path.join( folder_root, "rv_single_point_zscore_64.npy"  )     
    locator_slices_in, shape_original_in=retrieve_data_sample_location( folder_root, 64 )

    
elif data_run=="full_10":
    folder_root=r"C:\Users\salasja\Documents\MRI_files\datasets\data_10_ROI"
    file_in_fmri=os.path.join( folder_root, "fmri_full.npy"  )
    file_in_rvt=os.path.join( folder_root, "rv_full.npy"  )    
    
    
elif data_run=="sliced_10":
    folder_root=r"C:\Users\salasja\Documents\MRI_files\datasets\data_10_ROI"
    file_in_fmri=os.path.join( folder_root, "fmri_sliced_64.npy"  )
    file_in_rvt=os.path.join( folder_root, "rv_single_point_64.npy"  )       
    locator_slices_in, shape_original_in=retrieve_data_sample_location( folder_root, 64 )
    
elif data_run=="sliced_zscore_10":
    folder_root=r"C:\Users\salasja\Documents\MRI_files\datasets\data_10_ROI"
    file_in_fmri=os.path.join( folder_root, "fmri_sliced_zscore_64.npy"  )
    file_in_rvt=os.path.join( folder_root, "rv_single_point_zscore_64.npy"  )       
    locator_slices_in, shape_original_in=retrieve_data_sample_location( folder_root, 64 )    

elif data_run =="sliced_large_90":   
    folder_root=r"C:\Users\salasja\Documents\MRI_files\datasets\data_large_90_ROI"
    file_in_fmri=os.path.join( folder_root, "fmri_sliced_64.npy"  )
    file_in_rvt=os.path.join( folder_root, "rv_single_point_64.npy"  )     
    locator_slices_in, shape_original_in=retrieve_data_sample_location( folder_root, 64 )

elif data_run =="sliced_large_42":   
    folder_root=r"C:\Users\salasja\Documents\MRI_files\datasets\data_large_42_ROI"
    file_in_fmri=os.path.join( folder_root, "fmri_sliced_64.npy"  )
    file_in_rvt=os.path.join( folder_root, "rv_single_point_64.npy"  )     
    locator_slices_in, shape_original_in=retrieve_data_sample_location( folder_root, 64 )
        
elif data_run =="sliced_large_zscore_glob_90":   
    folder_root=r"C:\Users\salasja\Documents\MRI_files\datasets\data_large_90_ROI"
    file_in_fmri=os.path.join( folder_root, "fmri_sliced_zscore_glob_64.npy"  )
    file_in_rvt=os.path.join( folder_root, "rv_single_point_64.npy"  )     
    locator_slices_in, shape_original_in=retrieve_data_sample_location( folder_root, 64 )
    
else:
    raise Exception("Bad entry")


#%% Start timer
time_start=time.time()    

#%% Load fMRI data


fmri_all=np.load( file_in_fmri )


#%% Load RVT data

rvt_all =np.load( file_in_rvt  )




#%%

L=fmri_all.shape[1]
nsamples=fmri_all.shape[0]
channels=fmri_all.shape[2]

print( "Data has  %d samples, %d time points and  %d channels "%( nsamples, L, channels )   )


#%% Reshape 


y=rvt_all
X=fmri_all



#%% Select samples at random

seed(seed_number)

# Associate sample numbers with scan numbers and makes tuples of form ( sample_number ,scan_number)
scan_tups=[ (k, get_datapoint_source(  k,0 , locator_slices_in, shape_original_in[0] , shape_original_in[1] )[0] ) for k in range( locator_slices_in.shape[0] )  ]
scan_tups_time=[ (k, get_datapoint_source(  k,0 , locator_slices_in, shape_original_in[0] , shape_original_in[1] )[1] ) for k in range( locator_slices_in.shape[0] )  ]

#store the scan numbers in a dataframe
df_scans=pd.DataFrame.from_dict(  dict(scan_tups), orient="index"  ).rename(columns={0:"scan_number"})
df_scans_times=pd.DataFrame.from_dict(  dict(scan_tups_time), orient="index"  ).rename(columns={0:"time_point"})
df_scans[ "time_point" ]=df_scans_times["time_point"]

#array with all possible scans
num_scans=shape_original_in[0]
train_scan_size = num_scans-1
test_scan_size = num_scans - train_scan_size

#Determine all possible scan numbers
choices_possible_scans=np.arange( num_scans )

# read the scans to keep from file
list_scans_keep_test=read_scans_keep_test("scans_keep.txt")
rem_scan=set( list_scans_keep_test )
#create a set of the samples selected as training
set_selected_scans_train=set( [ s for s in choices_possible_scans if s not in rem_scan  ]  )
ser_in_training=df_scans["scan_number"].isin( set_selected_scans_train )
df_scans["in_train"]=ser_in_training
df_scans.to_csv("selected_scans_%s.csv"%(model_name)  )
df_scans_train=df_scans[ser_in_training ]
df_scans_test=df_scans[~ser_in_training ]


train_indices=np.array( df_scans_train.index )
test_indices=np.array( df_scans_test.index )

indices_all=np.append(train_indices, test_indices)


train_size = len(train_indices)
test_size = len(test_indices)



#%% Build test and train data sets

Xtrain= X[train_indices,:,:] 
Xtest = X[test_indices,:,:] 

ytrain=y[train_indices,:]
ytest=y[test_indices,:]  



#%%




model = Sequential()
model.add( Flatten() )
model.add(Dense(1 , use_bias=True , activation=None  ))

opt_adam=optimizers.Adam(lr=learning_rate)
model.compile(loss=loss, optimizer=opt_adam, metrics=['mean_absolute_error', 'mse'])




#%% Fit the model 

training=model.fit(Xtrain, ytrain, epochs=epochs , batch_size=512, validation_split=0, shuffle=True )


#%% Save history 

history = training.history

df_hist=pd.DataFrame.from_dict(history)
df_hist.to_csv("history_train_%s.csv"%model_name)



#%%
loss_and_metrics_test = model.evaluate(Xtest, ytest)



model.save('%s.h5'%model_name)  


#%%

#pred_y=model.predict(Xtest)
#eval_y=ytest


pred_y=model.predict(X)
eval_y=y

#%%
df_scans["pred_y"]=pred_y
df_scans["true_y"]=y
df_scans.to_csv("predictions_scans_%s.csv"%(model_name)  )


#%%

ser_cc=df_scans.groupby(["scan_number"]).apply( lambda x: np.corrcoef( x["pred_y"] , x["true_y"] )[0,1]  )

df_cc=pd.DataFrame({ "scan_number": ser_cc.index , "cc": ser_cc })
df_cc["in_train"]=[ k in set_selected_scans_train for k in ser_cc.index ]
df_cc.to_csv("cc_reconstructed.csv")


#%%


np.save( "indices_%s.npy"%(model_name) , indices_all  )

np.save(  "sizes_%s.npy"%(model_name)  , np.array([ nsamples, train_size, test_size ] ) )

#%%
    
import matplotlib.pyplot as plt

#%%
df_plot=df_cc[ df_cc["in_train"]==False  ]
cc_vals=df_plot["cc"].values


plt.figure(200)
plt.hist( cc_vals , bins=40 )
plt.title( "Predictions for test data 1D %s loss: %s"%( data_run, loss  ) )
plt.xlabel("Correlation")

#%%

folder_export_plots="plots"
try:
    os.mkdir(folder_export_plots)
except FileExistsError:
    pass


p=8
for p in range(num_scans):
    df_plot1=df_scans[  df_scans["scan_number"]==p ]
    cc=df_cc.loc[ p, "cc" ]
    in_train=df_cc.loc[p,"in_train"]
    if in_train==True:
        continue
    
    print("Plotting scan %d"%p )
    plt.figure(100, figsize=(20,20))
    plt.plot( df_plot1["pred_y"].values , label="predicted %0.2f"%cc  )
    plt.plot( df_plot1["true_y"].values , label="ground truth"  )
    plt.title("Reconstruction scan %d test data: %d"%(p,not in_train)  )
    plt.legend()
    file_plot="reconstructed_scan_%s_%d.png"%(model_name,p)
    plt.savefig( os.path.join( folder_export_plots ,file_plot ) )

    plt.close()
#%%
    
time_end=time.time()    
    
time_elapsed=time_end-time_start

print("Elapsed time " , time_elapsed )


