%% auto_physio_qa_criteria
% Specific to HCP Young Adult dataset

% test if RV is clipped
flag_bad_RV = 0; flag_bad_HR = 0;
if (length(find(resp_raw == 4095))>500)
    flag_bad_RV = 1;
    disp(['***detected RV clipping(high)*** sub: ',subID, ' scan: ', scan]);
end 

if (length(find(card_raw == 4095))>500)
    flag_bad_HR = 1;
    disp(['***detected PWA clipping(high)*** sub: ',subID, ' scan: ', scan]);
end 

% test if RV clamps at 0 sometimes
if (length(find(resp_raw == 0))>500)
    flag_bad_RV = 1;
    disp(['***detected RV clipping(low)*** sub: ',subID, ' scan: ', scan]);
end

% test if whole thing was empty
if(sum(diff(card_raw))==0)
    flag_bad_HR = 1;
    disp(['***detected HR empty*** sub: ',subID, ' scan: ', scan]);
end

if(sum(diff(resp_raw))==0)
    flag_bad_RV = 1;
    disp(['***detected RV empty*** sub: ',subID, ' scan: ', scan]);
end

%% first calculate heart rate peaks (REGS.hr)

if (~isempty(find(isnan(REGS.hr), 1)))
    flag_bad_HR = 1;
    disp(['***detected HR nans*** sub: ', subID, ' scan: ', scan]);
end

% test if HR is "robot"
if (mode(REGS.hr) == 48)
    flag_bad_HR = 1;
    disp(['***detected HR robot*** sub: ', subID, ' scan: ', scan]);
end

% detect unrealistic heart rates
hr_hi = length(find(REGS.hr>97));
hr_lo = length(find(REGS.hr<30)); 
if (sum([hr_hi,hr_lo])>2)
    flag_bad_HR = 1;
    disp(['***detected HR outliers*** sub: ', subID, ' scan: ', scan]);
end

if (flag_bad_HR || flag_bad_RV)
    % add code here to do something when a scan doesn't meet qa criteria
end
