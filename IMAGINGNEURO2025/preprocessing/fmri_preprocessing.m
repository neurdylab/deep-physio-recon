clear all
%% Preprocessing of fMRI data for DeepPhysioRecon
%% Step 2: Detrending, Bandpass Filtering, Downsampling

sub_id = '248339';
scan_id = 'REST1_LR';
atlases = {'schaefer', 'tian', 'tractseg', 'aan'};
proc_type = 'min'; % min: minimally processed, %fix: ICA-fix cleaned

% construct bpf
if strcmp(proc_type,'fix')
    nii_fname = ['/bigdata/HCP_rest/bad_samples/raw/', sub_id, '/fMRI/', proc_type, '/rfMRI_', scan_id ,'_hp2000_clean.nii.gz'];
else
    nii_fname = ['/bigdata/HCP_rest/bad_samples/raw/', sub_id, '/fMRI/', proc_type, '/rfMRI_', scan_id, '.nii.gz'];
end
disp(['Processing ...', nii_fname])

hdr = niftiinfo(nii_fname);
TR = hdr.PixelDimensions(4);

Ytmp = niftiread(nii_fname);
[~,bpFilter] = bandpass(double(Ytmp(:,5)),[0.01 0.15],1/TR);
clear Ytmp

path = '/bigdata/HCP_rest/bad_samples/processed/';

for i = 1:length(atlases)

    % load data to process
    fname = [path, proc_type, '/extracted/', atlases{i}, '/rois_', ...
        sub_id, '_rfMRI_' scan_id, '.mat'];
    
    % where to output
    out_fname = [path, proc_type, '/bpf-ds/', atlases{i}, '/rois_', ...
        sub_id, '_rfMRI_' scan_id, '.mat'];

    % load extracted time-series signals
    Ytmp2 = load(fname);
    Y = Ytmp2.Y;

    % time axis
    t = (1:length(Y))';
    
    % no motion correction but yes to removing linear and quad trends 
    XX = [ones(size(t,1),1), zscore([t, t.^2])];  % add a col of ones to the front of a design matrix
    MU = mean(Y,1); % save mean
    B = pinv(XX)*Y;
    Yr = Y - XX*B; % regress out the trends
    
    % bandpass filter
    Y_filt = filtfilt(bpFilter,double(Yr));
    
    % downsample
    Y_filt_ds = Y_filt(1:2:end,:);
    roi_dat = Y_filt_ds;
    
    % save
    save(out_fname,'roi_dat','MU');
    disp(['Done processing:', out_fname])

end
