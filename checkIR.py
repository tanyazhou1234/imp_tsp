import os
import scipy.io
import matplotlib.pyplot as plt

def load_mat_file(file_path):
    data = scipy.io.loadmat(file_path)
    impulse_responses = data['ir']  # 假设MAT文件中保存的脉冲响应是以这个变量名存储
    return impulse_responses

def plot_time_domain(ir, save_path, mic_id):
    plt.figure()
    plt.plot(ir)
    plt.title(f'Microphone {mic_id + 1} Time Domain')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join(save_path, f'mic_{mic_id + 1}_time.png'))
    plt.close()

def process_impulse_responses(mat_file, base_folder):
    impulse_responses = load_mat_file(mat_file)
    num_mics = impulse_responses.shape[0]
    
    for mic_id in range(num_mics):
        ir = impulse_responses[mic_id, :]
        plot_time_domain(ir, base_folder, mic_id)

# example
sp_id=1 # index of loudspeaker
mic_id_start=1
mic_id_end=32

mat_file_path = fr"result_IR_mat\imp_fs48000_SP-ch{sp_id}_MIC_ch{mic_id_start}-{mic_id_end}.mat"
output_folder = fr"log_checkIR\SP_{sp_id}_MIC_{mic_id_start}-{mic_id_end}"
if not os.path.exists(output_folder):
            os.makedirs(output_folder)

process_impulse_responses(mat_file_path, output_folder)
