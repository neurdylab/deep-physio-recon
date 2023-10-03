clear all
close all;
%% Preprocessing of physio data for DeepPhysioRecon

sub_id = '867468';
scan_id = 'rfMRI_REST2_RL';
path = '/bigdata/HCP_rest/bad_samples/raw/';

% fMRI meta data
hdr = niftiinfo([path, sub_id, '/fMRI/min/rfMRI_', scan_id, '.nii.gz']);
nframes = hdr.ImageSize(4); % number of fMRI volumes
TR = hdr.PixelDimensions(4);  % TR in seconds

% parameters
fs_phys = 400; % HCP raw data has 400 Hz sampling
dt_phys = 1/fs_phys; % spacing between each datapoint

% raw HCP physiological file
phys_file = [path, sub_id, '/physio/rfMRI_', scan_id, '_Physio_log.txt'];
physio_raw = load(phys_file);

% extract respiration and cardiac data
resp_wave = physio_raw(:,2);
card_wave = physio_raw(:,3);

%% Cardiac Data Processing
% Demean
card_dat = card_wave - mean(card_wave);

% Bandpass filter
fcut_BPF = [0.5,4];
Fn = fs_phys/2;
Wn = fcut_BPF/Fn; 
Nb = 2;
[B, A] = butter(Nb,Wn);
card_bpf = filtfilt(B,A,double(card_dat));

% Detect peaks
card_rng = iqr(card_bpf);
minHeight = 0.05*card_rng;
minDist = (fs_phys/2); % not needed here, no doublets...
[pks,locs] = findpeaks(card_bpf,'minpeakheight',minHeight,'minpeakdistance',minDist);
clear maxtab_c; maxtab_c(:,1) = locs; maxtab_c(:,2) = round(pks);

% Extract cardiac trigger times and approximate heart rate
card_trig_samples = locs;
card_trig_times = card_trig_samples*dt_phys;

% IBI & "instantaneous" HR
IBI = (diff(card_trig_times));
HR = (1./diff(card_trig_times))*60; %bpm

% check peak detection 
spc = dt_phys/TR;
time_axis = 0:spc:length(resp_wave)*spc-spc;
pts = time_axis(locs);

figure(101); clf; 
set(gcf,'color','w'); 
g1 = subplot(4,1,1); hold on;
plot(card_bpf); 
hold on;
plot(card_dat-mean(card_dat),'color',0.8*[1 1 1]);
legend('cardiac filtered','cardiac raw','Location','northwest'); 

figure(101);
g2 = subplot(4,1,2); hold on;
plot((1:length(card_bpf)),card_bpf);
title('cardiac');
hold on;
plot(maxtab_c(:,1),maxtab_c(:,2),'r.','markersize',10); 
legend('cardiac filtered','cardiac peaks','Location','northwest'); 
xlabel('physio sample #');

% check heart rate as a function of time
figure(101);
g3 = subplot(4,1,3); hold on;
plot(card_trig_samples(1:end-1),HR,'b'); ylabel('beats per min'); title('cardiac rate');
hold on;
plot(card_trig_samples(1:end-1),HR,'c.'); % +dots
xlabel('physio sample #');
linkaxes([g1,g2],'x');
ylim([40 95]);

% IBI time series (by sample number - flag outliers)
figure(101);
g4 = subplot(4,1,4); hold on;
plot(IBI); 
xlabel('index'); ylabel('IBI');
drawnow;

%% resp:
% de-mean
resp.wave = resp_wave - mean(resp_wave);

%% Calculate Respiration Variation (RV) and Heart Rate (HR)
    
% lookup table of IBI (denoised) v. cardiac trig
% time (assigned to halfway between the respective beats)
t_ibi = 0.5*(card_trig_times(2:end) + card_trig_times(1:end-1));
assert(length(t_ibi)==length(IBI))

% sampling to match fMRI tr (center of each tr)
Twin = 6; % sec windows (3s on either side)
t_fmri = (TR/2)+0:TR:TR*nframes;

% make RV & HR regressors, as well as pulseox amplitude (stdev)
RV = [];
HR = [];

for kk=1:nframes
    t = t_fmri(kk);
    
    % heart rate
    % ---------------------- %
    % get time bin centered at this TR
    t1 = max(0,t-Twin*0.5);
    t2 = min(TR*nframes,t+Twin*0.5);
    % find IBI's falling within this interval
    inds = intersect(find(t_ibi<=t2),find(t_ibi>=t1));
    HR(kk) = (60./median(IBI(inds)));
    
    % pulse amplitude
    % ---------------------- %
    if length(resp.wave)~=length(card_dat)
      error('resp & card sampled at different rates');
    else
      np = length(resp.wave);
    end

    % window (in samples)
    i1 = max(1,floor((t - Twin*0.5)/dt_phys)); 
    i2 = min(np, floor((t + Twin*0.5)/dt_phys));

    % respiration variation
    % ---------------------- %
    RV(kk) = std(resp.wave(i1:i2));
end

f = figure('visible','on', 'units', 'normalized', 'outerposition', [0, 0, 1, 1]);
% figure(103); 
hold on; set(gcf,'color','w');
subplot(411);
plot(time_axis,card_bpf); hold on;
plot(pts,maxtab_c(:,2),'r.','markersize',10); 
title('Cardiac Peaks')
xlim([0 1200]);

subplot(412); 
plot(HR); title('Heart Rate');
% ylim([40 100]);

subplot(413);
plot(time_axis, resp_wave)
title('Raw Respiration')
xlim([0 1200]);

subplot(414);
plot(zscore(RV)); title('Respiratory Variation');
xlabel('TR'); 

out_path = '/bigdata/HCP_rest/bad_samples/processed/physio/QA/';
saveas(f, [out_path, sub_id, '_', scan_id, '_QA.png']);

%% Bandpass Filter Downsample 
% time axes for visualizing rv and hr
time_axis1 = [0:TR:nframes*TR-TR];
time_axis2 = time_axis1(1:2:end);

% filtering (0.01-0.15 Hz) and downsampling (x2) of HR waveform
[~,bpFilter] = bandpass(HR, [0.01 0.15], 1/TR);
hr_filt = filtfilt(bpFilter,HR) + mean(HR);
hr_filt_ds = hr_filt(1:2:end); 

figure(103);
% visualize hr
subplot(211); % hr
hold on;
plot(time_axis1,HR);
plot(time_axis2,hr_filt_ds); 
legend('hr','hr filt ds', 'Orientation', 'horizontal'); 
xlabel('time'); title('Heart Rate');
 
save([out_path, ['/HR_filt_ds/', sub_id, '_', scan_id, '_hr_filt_ds.mat']], 'hr_filt_ds');

% filtering (0.01-0.15 Hz) and downsampling (x2) of RV waveform
[~,bpFilter] = bandpass(RV, [0.01 0.15], 1/TR);
rv_filt = filtfilt(bpFilter,RV);
rv_filt_ds = rv_filt(1:2:end); 

% visualize rv
subplot(212); % RV
hold on;
plot(time_axis1,RV-mean(RV));
plot(time_axis2,rv_filt_ds); 
legend('rv','rv filt ds', 'Orientation', 'horizontal');
xlabel('time'); title('Respiration Variation');

save([out_path, ['/RV_filt_ds/', sub_id, '_', scan_id, '_rv_filt_ds.mat']], 'rv_filt_ds');



 
