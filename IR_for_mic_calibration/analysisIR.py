import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq

# 1. spectrum distribution. frequency-time
def plot_spectrogram(ir, fs, save_path, s_id, m_id):
    stft_len=256
    stft_overlap = 128
    stft_win = np.hamming(stft_len)
    f, t, Sxx = spectrogram(ir, fs, window=stft_win, noverlap=stft_overlap)#nperseg=256,

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='plasma', vmin=-160, vmax=-60)
    plt.title('Frequency-Time Energy Distribution')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    # Customize colorbar
    cbar = plt.colorbar(label='Energy (dB)')
    cbar.ax.tick_params(labelsize=10)  # Adjust colorbar tick size for readability

    # Optional: set frequency axis limit (adjustable)
    plt.ylim(0, fs // 2)  # Limit to Nyquist frequency

    #plt.show()
    save_file = os.path.join(save_path, f'sp-{s_id}_mic-{m_id}_IR_Spectrogram.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()

# 2. Frequency response
def plot_frequency_response(ir, fs, save_path, m_id):
    N = len(ir)
    freq = fftfreq(N, 1/fs)[:N//2]
    ir_fft = fft(ir)
    magnitude = np.abs(ir_fft)[:N//2]
    
    plt.figure()
    plt.plot(freq, 20 * np.log10(magnitude))
    plt.title('Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    plt.xlim(0, 10000)
    plt.ylim(-30, 30)
    #plt.show()
    plt.savefig(os.path.join(save_path, f'mic_{m_id}_Frequency.png'))
    plt.close()

def plot_all_mic_ir_in_one_figure(ir_list, fs, save_path):
    """
    Plot all 16 mic IR in one figure.
    
    Parameters:
    ir_list (list of np.array): List of IRs for each mic.
    fs (int): Sampling frequency.
    save_path (str): Path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    
    for i, ir in enumerate(ir_list):
        time = np.arange(len(ir)) / fs
        plt.plot(time, ir, label=f'Mic {i+1}')
    
    plt.title('Impulse Responses of All Mics')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'all_mic_ir_in_one_figure.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_mic_ir(ir_list, fs, save_path):
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, ir in enumerate(ir_list):
        time = np.arange(len(ir)) / fs
        axes[i].plot(time, ir)
        axes[i].set_title(f'Mic {i+1}')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid()
        axes[i].set_ylim(-1, 2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'all_mic_ir.png'), dpi=300, bbox_inches='tight')
    plt.close()
def plot_all_mic_ir_freq_in_1_figure(ir_list, fs, save_path):
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, ir in enumerate(ir_list):
        N = len(ir)
        freq = fftfreq(N, 1/fs)[:N//2]
        ir_fft = fft(ir)
        magnitude = np.abs(ir_fft)[:N//2]
        
        axes[i].plot(freq, 20 * np.log10(magnitude), linewidth=1.5)
        axes[i].set_title(f'Mic {i+1}')
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Magnitude (dB)')
        axes[i].grid()
        axes[i].set_xlim(100, 10000)
        axes[i].set_ylim(-30, 20)
        axes[i].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'all_mic_ir_freq_log.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_mic_freq_response(ir_list, fs, save_path):
    plt.figure(figsize=(10, 6))
    
    for i, ir in enumerate(ir_list):
        N = len(ir)
        freq = fftfreq(N, 1/fs)[:N//2]
        ir_fft = fft(ir)
        magnitude = np.abs(ir_fft)[:N//2]
        plt.plot(freq[::10], 20 * np.log10(magnitude[::10]), label=f'Mic {i+1}',linewidth=0.5)
    
    plt.title('Frequency Response of All Mics- interval of 10 data points')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(ncol=2, loc='lower right')
    plt.xlim(100, 10000)
    plt.xscale('log')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'all_mic_freq_response.png'), dpi=300, bbox_inches='tight')
    plt.close()

# initialize
fs = 48000 
tsp_len=65536

"""
log_folder=fr'log_test_data/before_calibration'
if not os.path.exists(log_folder):
      os.makedirs(log_folder)

ir_list=[]
for mic_idx in range(1,17):
      rawfile=fr'test_data/imp_fs{fs}_ch{mic_idx}.mat'
      raw_data=loadmat(rawfile)
      ir_matrix=raw_data['ir']
      ir_target_mic=ir_matrix[mic_idx-1, :]
      #plot_frequency_response(ir_target_mic, fs, log_folder, mic_idx)
      #plot_spectrogram(ir_cur, fs, log_folder, sp_idx, mic_id)
      ir_list.append(ir_target_mic)
# save ir_list as a mat file
#ir_dict = {f'ir_{i+1}': ir for i, ir in enumerate(ir_list)}
#savemat(os.path.join(log_folder, 'ir_list.mat'), ir_dict)

# plot as needed
#plot_all_mic_ir_in_one_figure(ir_list, fs, log_folder)
#plot_all_mic_ir(ir_list, fs, log_folder)
#plot_all_mic_ir_freq_in_1_figure(ir_list, fs, log_folder)
#plot_all_mic_freq_response(ir_list, fs, log_folder)
"""
# doing calibration
mic_ref=13 # select a reference mic

data=loadmat(fr'log_test_data/ir_list.mat')
ir_list=[data[f'ir_{i+1}'].flatten() for i in range(16)]

max_val_ref=max(ir_list[mic_ref-1])
print(f'Max value of reference mic {mic_ref} is {max_val_ref}')
multifactor=[]
ir_list_calibrated=[]
for i in range(16):
      max_val=max(ir_list[i])
      multifactor.append(max_val_ref/max_val)
      ir_list_calibrated.append(ir_list[i]*multifactor[i])

# save calibrated ir_list as a mat file
#ir_dict = {f'ir_{i+1}': ir for i, ir in enumerate(ir_list_calibrated)}
#savemat(os.path.join(fr'log_test_data', 'ir_list_calibrated.mat'), ir_dict)
savemat(os.path.join(fr'log_test_data', 'multifactor.mat'), {'multifactor': multifactor})

# plot as needed
"""
log_folder=fr'log_test_data/after_calibration'
if not os.path.exists(log_folder):
      os.makedirs(log_folder)
plot_all_mic_ir_in_one_figure(ir_list_calibrated, fs, log_folder)
plot_all_mic_ir(ir_list_calibrated, fs, log_folder)
plot_all_mic_ir_freq_in_1_figure(ir_list_calibrated, fs, log_folder)
plot_all_mic_freq_response(ir_list_calibrated, fs, log_folder)
"""      



