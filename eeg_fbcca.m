function results = eeg_fbcca(eeg, list_freqs, fs, num_harms, num_fbs)
warning off;
if nargin < 3
    error('stats:test_fbcca:LackOfInput', 'Not enough input arguments.'); 
end

if ~exist('num_harms', 'var') || isempty(num_harms), num_harms = 3; end

if ~exist('num_fbs', 'var') || isempty(num_fbs), num_fbs = 5; end

fb_coefs = [1:num_fbs].^(-1.25)+0.25;

[num_targs, ~, num_smpls] = size(eeg);
y_ref = cca_reference(list_freqs, fs, num_smpls, num_harms);
num_class = length(list_freqs);
results=zeros(num_targs,1);
for targ_i = 1:1:num_targs
    temp = eeg(targ_i, :, :);
    sizetemp=size(temp);
    test_tmp = reshape(temp,sizetemp(2:3));
    for fb_i = 1:1:num_fbs
        testdata = filterbank(test_tmp, fs, fb_i);
        for class_i = 1:1:num_class
            refdata = squeeze(y_ref(class_i, :, :));
            [~,~,r_tmp] = canoncorr(testdata', refdata');
            r(fb_i, class_i) = r_tmp(1,1);
        end % class_i
    end % fb_i
    rho = fb_coefs*r;
    [~, tau] = max(rho);
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