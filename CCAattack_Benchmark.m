rng(0);warning off;
freq_phase = load('data/benchmark/Freq_Phase.mat');
list_freqs = freq_phase.freqs;
period=ceil(250./list_freqs);
fs=250;
num_harms=5;
num_fbs = 5;
[f_b,f_a]=butter(8,[0.056 0.72]);
sinall=load('XsinAll.mat');
XsinAll=sinall.XsinAll;  % square signals
datalength=375;
num_trial=240;
start=35;
ASR_CCA=zeros(10,3,35,40);
ASR_FBCCA=zeros(10,3,35,40);
ASR_MSI=zeros(10,3,35,40);
ASR_MEC=zeros(10,3,35,40);
ASR_MCC=zeros(10,3,35,40);
for ch = 1:10  %attack channel
    for am = 1:3 %attack signal amplitude = am / 10
        for data_i =1:1
            data_i
            data = load(['data/benchmark/S' num2str(data_i) '.mat']);
            X = data.X;
            X = X(:,:,126+start:125+start+datalength);
            Y = data.Y;       
            for sin_i = 1:40
                XT = X;
                for trail_i=1:num_trial
                    phase_i=unidrnd(period(sin_i));  % random phase
                    temp=squeeze(XT(trail_i,:,:));
                    if ch<10
                        % attack single channel
                        stdTemp=std(temp(ch,:));
                        temp(ch,:)=temp(ch,:)+stdTemp*am*XsinAll(sin_i,phase_i:datalength+phase_i-1)/10;
                    else
                        % attack all channels
                        stdTemp=mean(std(temp'));
                        temp=temp+stdTemp*am*XsinAll(sin_i,phase_i:datalength+phase_i-1)/10;
                    end
                    temp=filtfilt(f_b,f_a,temp');%滤波
                    temp=temp';
                    XT(trail_i,:,:)=temp;
                end
                yP_cca = eeg_cca(XT, list_freqs, fs, num_harms);
                yP_fbcca = eeg_fbcca(XT, list_freqs, fs, num_harms, num_fbs);
                yP_msi = eeg_msi(XT, list_freqs, fs, num_harms);        
                yP_mec = eeg_mec(XT, list_freqs, fs, num_harms);
                yP_mcc = eeg_mcc(XT, list_freqs, fs, num_harms);
                ASR_CCA(ch, am, data_i,sin_i) = sum(yP_cca==sin_i)/num_trial;
                ASR_FBCCA(ch, am, data_i,sin_i) = sum(yP_fbcca==sin_i)/num_trial;
                ASR_MSI(ch, am, data_i,sin_i) = sum(yP_msi==sin_i)/num_trial;        
                ASR_MEC(ch, am, data_i,sin_i) = sum(yP_mec==sin_i)/num_trial;
                ASR_MCC(ch, am, data_i,sin_i) = sum(yP_mcc==sin_i)/num_trial;
            end
        end
    end
end