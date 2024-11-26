import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
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
    #plt.show()
    plt.savefig(os.path.join(save_path, f'mic_{m_id}_Frequency.png'))
    plt.close()

# plot IR frequency-time spectrum from each sp to each mic
fs = 48000 
tsp_len=65536

for row_idx in range(1,2):
      log_folder=fr'log_RIRs/log_Mic_row_index_{row_idx}'
      if not os.path.exists(log_folder):
            os.makedirs(log_folder)
      for sp_idx in range(1,17):
            rawfile=fr'IR_raw_data/Mic_row_index_{row_idx}/imp_fs{fs}_ch{sp_idx}.mat'
            raw_data=loadmat(rawfile)
            ir_matrix=raw_data['ir']
            num_mic=ir_matrix.shape[0]
            for mic_id in range(1, 2):#num_mic+1):
                  ir_cur=ir_matrix[mic_id-1, :]
                  plot_spectrogram(ir_cur, fs, log_folder, sp_idx, mic_id)

