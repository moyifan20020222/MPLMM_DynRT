import subprocess
import numpy as np
import pandas as pd

# 定义OpenSMILE可执行文件路径（根据实际安装路径修改）
opensmile_path = "D:/opensmile-3.0.2-windows-x86_64/bin/SMILExtract.exe"

# 定义配置文件路径（这里以IS10_paraling.conf为例，可按需替换）
config_file_path = "D:/opensmile-3.0.2-windows-x86_64/config/is09-13/IS13_ComParE.conf"

# 输入的WAV音频文件路径 Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 44100 Hz, mono, s16, 705 kb/s
input_wav_file = "video_clip/Friends_S1E1_1_2.wav"


def extract_audio_features():
    """
    使用OpenSMILE提取音频文件的声学特征，并存储到numpy数组后保存到文件
    测试提取情况
    """
    # 临时的输出文件路径（以CSV格式作为中间过渡）
    temp_output_csv_file = "feature/temp_feature.csv"
    command = [
        opensmile_path,
        "-C", config_file_path,
        "-I", input_wav_file,
        "-O", temp_output_csv_file
    ]
    command1 = ['ffmpeg', '-i', input_wav_file, '-hide_banner']
    subprocess.run(command1)
    try:
        # 执行OpenSMILE命令
        subprocess.run(command, check=True)
        print("音频特征提取成功，正在处理数据...")

        # # 使用numpy读取CSV文件中的特征数据并存储到数组中
        # features_array = np.genfromtxt(temp_output_csv_file, delimiter=',', skip_header=1)
        #
        # # 删除临时的CSV文件
        # import os
        # os.remove(temp_output_csv_file)
        #
        # # 将numpy数组保存为.npy文件
        # output_npy_file = "features.npy"
        # np.save(output_npy_file, features_array)
        print("音频特征已成功保存到features.npy文件中。")
    except subprocess.CalledProcessError as e:
        print(f"执行OpenSMILE命令出错: {e}")


if __name__ == "__main__":
    # extract_audio_features()
    """
    opensmile 似乎只能处理单声道的信息， 总共6375 列， 第一列是名字， 最后一列是种类， 实际的内容是中间的6373列数据
    """
    df = pd.read_csv("feature/audio_feature.csv", sep=',', skiprows=range(6380), header=None)
    selected_data = df.iloc[:, 1:6374].to_numpy()
    selected_data = np.expand_dims(selected_data, axis=1)
    print(selected_data.shape)
    print(selected_data[308, :, :])
