% 清空工作区和命令窗口
clear;
clc;

% original fs
fs_original = 48000; % Hz
G = struct();
for row_idx=1:32
    for sp_cur=1:16
        %% Step1: load and observe data      
        
        filename_IR=sprintf('IR_raw_data/Mic_row_index_%d/imp_fs48000_ch%d.mat',row_idx, sp_cur);
        load(filename_IR); % data file

        % data size
        [numChannels,numSamples] = size(ir);% number of mic channels
        impulseResponses=ir.';
        
        %% Step2: extract the likely meaningful part
        icut1=round(30/1000*fs_original); % 最大値から前 30ms まで切り出し（暫定）
        icut2=round(300/1000*fs_original); % 最大値から後 300ms まで切り出し（暫定）
        [~,max_idx]=max(impulseResponses);% get index of the maximum vale, 1x32 double
        start_idx=max_idx-icut1+1;
        end_idx=max_idx+icut2;
        impulseResponses_truncated=zeros(end_idx(1)-start_idx(1)+1,numChannels);
        for mic_idx=1:numChannels
            impulseResponses_truncated(:,mic_idx) = impulseResponses(start_idx(mic_idx):end_idx(mic_idx), mic_idx);
        end
        
        % update samples number
        numSamples_truncated = size(impulseResponses_truncated,1);
        
        % Windowing: Hanning
        window = hann(numSamples_truncated);
        impulseResponses_windowed = impulseResponses_truncated .* window;

        %% Step 3：Anti-Alisling and down-sampling
        downsampleFactor = 3; % down sampling rate
        fs_new = fs_original / downsampleFactor;
        
        % Anti-Alisling filter, FIR low-pass
        % set cut-off frequency
        cutoff_freq = fs_new / 2; % Hz
        normalized_cutoff = cutoff_freq / (fs_original / 2);
        filterOrder = 128; % adjust as need
        % FIR filter design
        b = fir1(filterOrder, normalized_cutoff);
        impulseResponses_filtered = zeros(size(impulseResponses_windowed));
        for ch = 1:numChannels
            impulseResponses_filtered(:, ch) = filtfilt(b, 1, impulseResponses_windowed(:, ch));
        end

        %% Step 4：down sampling
   
        impulseResponses_downsampled = impulseResponses_filtered(1:downsampleFactor:end, :);
        numSamples_downsampled = size(impulseResponses_downsampled, 1);

        %% Step 5：Verify and save results
        %save('impulseResponses_processed.mat', 'impulseResponses_downsampled', 'fs_new');
        %% Step 6: turn IR into transfer function
        Nfft = 2^nextpow2(numSamples_downsampled);
        
        % Zero-padding
        imp_padded = zeros(Nfft, numChannels);
        imp_padded(1:numSamples_downsampled,:)=impulseResponses_downsampled;
        imp_Nfft=imp_padded;
        
        
        %H_positive = zeros(Nfft/2, numChannels);
        f_positive = (1:Nfft/2)*(fs_new/Nfft);% not include 0 Hz
        
        for ch = 1:numChannels
            h = imp_Nfft(:, ch);
            H = fft(h, Nfft);
            %H_positive(:, ch) = H(2:Nfft/2+1);
            h_cur_positive=H(2:Nfft/2+1);
            ch_cur=ch*row_idx;
            G.(sprintf('sp%d_mic%d', sp_cur, ch_cur)) = h_cur_positive;
        end
            
    end
end
save('transferFunction_all.mat', 'G');
clear;
disp("IR pre-processing completed.");