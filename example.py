import os
import numpy as np
import scipy.io
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft

# 读取.mat文件
def load_mat_file(file_path):
    data = scipy.io.loadmat(file_path)
    impulse_responses = data['ir']  # 使用正确的变量名
    return impulse_responses

# 创建文件夹结构
def create_folders(base_folder):
    time_domain_folder = os.path.join(base_folder, 'time_domain')
    freq_domain_folder = os.path.join(base_folder, 'frequency_domain')
    stft_folder = os.path.join(base_folder, 'stft')
    wav_folder = os.path.join(base_folder, 'wav_files')
    
    # 创建不同类型的文件夹
    for folder in [time_domain_folder, freq_domain_folder, stft_folder, wav_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    return time_domain_folder, freq_domain_folder, stft_folder, wav_folder

# 绘制时域图并保存
def plot_time_domain(ir, save_path, mic_id):
    plt.figure()
    plt.plot(ir)
    plt.title(f'Microphone {mic_id + 1} Time Domain')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join(save_path, f'mic_{mic_id + 1}_time.png'))
    plt.close()

# 绘制频域图并保存
def plot_frequency_domain(ir, fs, save_path, mic_id):
    n = len(ir)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_values = np.abs(fft(ir))
    
    plt.figure()
    plt.plot(freqs[:n // 2], fft_values[:n // 2])
    plt.title(f'Microphone {mic_id + 1} Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join(save_path, f'mic_{mic_id + 1}_freq.png'))  # 保存频谱图
    plt.close()

# 绘制STFT频谱图并保存
def plot_stft(tsp, fs, save_path, mic_id, stft_len, stft_win, stft_overlap, tsp_len, nsync):
    plt.figure()
    pxx, stft_freq, stft_bins, stft_time = plt.specgram(tsp, NFFT=stft_len, Fs=fs, window=stft_win, noverlap=stft_overlap)
    plt.axis([0, tsp_len*(nsync+1)/fs, 0, fs/2])
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(f'Microphone {mic_id + 1} STFT')
    plt.savefig(os.path.join(save_path, f'mic_{mic_id + 1}_stft.png'))  # 保存STFT图
    plt.close()

# 保存为.wav文件 (使用wave模块)
def save_as_wav_with_wave(ir, fs, save_path, mic_id):
    file_name = f'mic_{mic_id + 1}.wav'
    file_path = os.path.join(save_path, file_name)

    wf = wave.open(file_path, 'wb')

    nchannels = 1  # 单声道
    sampwidth = 2  # 采样宽度为2字节 (16位)
    nframes = len(ir)
    comptype = 'NONE'
    compname = 'not compressed'

    wf.setparams((nchannels, sampwidth, fs, nframes, comptype, compname))

    ir_int16 = np.int16(ir * 32767)  # 将浮点数转换为16位PCM

    wf.writeframes(ir_int16.tobytes())
    wf.close()

# 主函数
def process_impulse_responses(mat_file, fs, base_folder):
    impulse_responses = load_mat_file(mat_file)
    num_mics = impulse_responses.shape[0]
    
    # 创建文件夹结构
    time_domain_folder, freq_domain_folder, stft_folder, wav_folder = create_folders(base_folder)
    
    for mic_id in range(num_mics):
        ir = impulse_responses[mic_id, :]

        # 绘制并保存时域图
        plot_time_domain(ir, time_domain_folder, mic_id)
        
        # 绘制并保存频域图
        plot_frequency_domain(ir, fs, freq_domain_folder, mic_id)
        
        # 绘制并保存STFT图
        #plot_stft(tsp, fs, stft_folder, mic_id, stft_len, stft_win, stft_overlap, tsp_len, nsync)
        
        # 保存为.wav文件
        save_as_wav_with_wave(ir, fs, wav_folder, mic_id)

# 示例调用
mat_file_path = r"result_IR_mat\imp_fs48000_SP-ch1_MIC_ch1-16.mat"
sampling_rate = 48000
output_folder = 'log_checkIR'

process_impulse_responses(mat_file_path, sampling_rate, output_folder)
