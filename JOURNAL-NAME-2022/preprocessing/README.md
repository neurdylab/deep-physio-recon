## Preprocessing Steps for DeepPhysioRecon
### fMRI

    Step 1: Extract time-series signals from fMRI data using four group-level atlases: [Schaefer Cortical](https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz), [PANDORA: TractSeg White Matter](https://github.com/MASILab/Pandora-WhiteMatterAtlas/blob/master/TractSeg/supplementary/TractSeg_HCP.nii.gz), [Melbourne Subcortical](https://github.com/yetianmed/subcortex/blob/master/Group-Parcellation/7T/Tian_Subcortex_S1_7T.nii), [AAN Brainstem](https://www.nmr.mgh.harvard.edu/resources/aan-atlas)
    
    extract_timeseries.ipynb

    Step 2: Detrending, Bandpass Filtering, Downsampling
    Regress out the linear and quadratic trends. Band-pass filter in a range of 0.01-0.15 Hz and downsample to math 1.44 s/volume sampling frquency. 

    fMRI_processing.m

### Respiration and Heart Rate Signals
    
    From the physiological data accompanying each scan, extract time series of respiration volume (RV) and heart rate (HR). The same filtering and downsampling procedure was applied to the RV and HR signals.

    physio_processing.m