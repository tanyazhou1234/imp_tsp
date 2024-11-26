import sys
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from imptsp import imptsp

#out_channel = 1
for out_channel in range(2, 3):
    print(f"Measuring impulse response for out_channel {out_channel}")

    # for multiple mic array, define row-idx
    mic_row_idx=21
    # Parameters
    Fs = 48000
    tsp_len = 65536

    in_channel = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    n_in_channel = len(in_channel)
    
    ##########################
    folder_name = fr"IR_raw_data/Mic_row_index_{mic_row_idx}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    ##########################
    # Initialize
    imp = imptsp(Fs,tsp_len,out_channel)

    # Measuring impulse response
    (ir,tsp) = imp.get_imp(in_channel,out_channel)

    # Plot
    plt.figure()
    plt.plot(tsp)
    file_path = os.path.join(folder_name, f"signal_SP{out_channel}_to_MIC1.png")
    plt.savefig(file_path)  # 保存图像
    plt.close()

    plt.figure()
    pxx, stft_freq, stft_bins, stft_time = plt.specgram(tsp, NFFT=imp.stft_len, Fs=Fs, window=imp.stft_win, noverlap=imp.stft_overlap)
    plt.axis([0, tsp_len*(imp.nsync+1)/Fs, 0, Fs/2])
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    file_path = os.path.join(folder_name, f"signal_spectrum_SP{out_channel}_to_MIC1.png")
    plt.savefig(file_path)  # 保存图像
    plt.close()

    plt.figure()
    plt.plot(ir[0,:])
    file_path = os.path.join(folder_name, f"imp_SP{out_channel}_to_MIC1.png")
    plt.savefig(file_path)  # 保存图像
    plt.close()
    #plt.show()

    # Save data
    fname_imp = os.path.join(folder_name, "imp_fs%d_ch%d.mat" % (int(Fs), int(out_channel)))
    #fname_imp = "imp_fs%d_ch%d.mat" % (int(Fs),int(out_channel))
    sio.savemat(fname_imp,{'ir':ir})

    # Terminate
    imp.terminate()

print("All measurements completed.")