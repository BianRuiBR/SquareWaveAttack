function results = eeg_msi(eeg, list_freqs, fs, num_harms)
[num_targs, num_channel, num_smpls] = size(eeg);
y_ref = cca_reference(list_freqs, fs, num_smpls, num_harms);
num_class = length(list_freqs);
r=zeros(1,num_class);
results=zeros(num_targs,1);
for targ_i = 1:1:num_targs
    test_tmp = squeeze(eeg(targ_i, :, :));
    for class_i = 1:1:num_class
        refdata = squeeze(y_ref(class_i, :, :));
        
        C11 = test_tmp * test_tmp' / num_smpls;
        C22 = refdata * refdata' / num_smpls;
        C12 = test_tmp * refdata' / num_smpls;
        C21 = refdata * test_tmp' / num_smpls;
        
        R = [eye(num_channel),C11^(-0.5) * C12 * C22^(-0.5);C22^(-0.5)*C21*C11^(-0.5),eye(num_harms*2)];
        [V, D] = eig(R);
        D = diag(D);
        D = D/sum(D);
       
        r(class_i) = 1+sum(D.*log(D)/log(num_channel + 2 * num_harms));
    end % class_i
    [~, tau] = max(r);
    results(targ_i) = tau;
end % targ_i

function [ y_ref ] = cca_reference(list_freqs, fs, num_smpls, num_harms)
if nargin < 3 
    error('stats:cca_reference:LackOfInput',...
        'Not enough input arguments.');
end

if ~exist('num_harms', 'var') || isempty(num_harms), num_harms = 3; end

num_freqs = length(list_freqs);
tidx = (1:num_smpls)/fs;
for freq_i = 1:1:num_freqs
    tmp = [];
    for harm_i = 1:1:num_harms
        stim_freq = list_freqs(freq_i);
        tmp = [tmp;...
            sin(2*pi*tidx*harm_i*stim_freq);...
            cos(2*pi*tidx*harm_i*stim_freq)];
    end % harm_i
    y_ref(freq_i, 1:2*num_harms, 1:num_smpls) = tmp;
end % freq_i