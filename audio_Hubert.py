import librosa
import torch
from transformers import HubertModel, HubertConfig
import numpy as np
import os

wav_file_path = "video_clip/Friends_S1E1_1_2.wav"


# 1. 使用librosa加载并处理音频文件
def load_and_process_audio(wav_path):
    # 加载音频文件，设置采样率为None表示按原采样率加载，返回音频数据和原始采样率 音频的采样为 44100 Hz
    audio_data, original_sample_rate = librosa.load(wav_path, sr=44100)
    print(audio_data)
    # 通常Hubert模型期望的采样率可能为16000Hz，这里进行重采样（根据Hubert模型实际要求调整）
    target_sample_rate = 16000
    resampled_audio = librosa.resample(audio_data, orig_sr=original_sample_rate, target_sr=target_sample_rate)

    # 将音频数据转换为二维张量，形状为 (1, sequence_length)，假设为单声道音频
    audio_tensor = np.array(resampled_audio, dtype=np.float32).reshape(1, -1)
    print(audio_tensor.shape)
    return audio_tensor


processed_audio = load_and_process_audio(wav_file_path)

# 2. 加载Hubert模型
model_name = "hubert-large-ls960"  # 这里选择一个预训练的Hubert模型，可根据需求换
config = HubertConfig.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name, config=config)
model.eval()  # 设置为评估模式

# 3. 将处理后的音频输入Hubert模型进行编码
with torch.no_grad():
    # 添加批次维度，形状变为 (1, 1, sequence_length)，符合模型输入要求
    input_audio = torch.from_numpy(processed_audio)
    outputs = model(input_audio)
    encoding_result = outputs.last_hidden_state

# 4. 存储Hubert的编码结果
output_dir = "src/audio_Hubert"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file_path = os.path.join(output_dir, "hubert_encoding_result.npy")
# 将编码结果转换为numpy数组并保存
np.save(output_file_path, encoding_result.detach().cpu().numpy())
print(f"Hubert编码结果已保存至 {output_file_path}")
