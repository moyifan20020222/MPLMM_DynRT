import subprocess
import re
import librosa
import torch
from transformers import HubertModel, HubertConfig
import numpy as np
import os

import numpy
import timm as timm
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import timm
from transformers import CLIPVisionModel

"""
视频编码信息
Stream #0:0: Video: hevc (Main 10), yuv420p10le(tv), 1920x1080, SAR 1:1 DAR 16:9, 23.98 fps, 23.98 tbr, 1k tbn (default)
      Metadata:
        BPS             : 5108482
        BPS-eng         : 5108482
        DURATION        : 00:22:49.410000000
        DURATION-eng    : 00:22:49.410000000
        NUMBER_OF_FRAMES: 32833
        NUMBER_OF_FRAMES-eng: 32833
        NUMBER_OF_BYTES : 874450829
        NUMBER_OF_BYTES-eng: 874450829
        _STATISTICS_WRITING_APP: mkvmerge v7.5.0 ('Glass Culture') 64bit built on Jan  4 2015 16:48:44
        _STATISTICS_WRITING_APP-eng: mkvmerge v7.5.0 ('Glass Culture') 64bit built on Jan  4 2015 16:48:44
        _STATISTICS_WRITING_DATE_UTC: 2016-02-24 01:10:32
        _STATISTICS_WRITING_DATE_UTC-eng: 2016-02-24 01:10:32
        _STATISTICS_TAGS: BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES
        _STATISTICS_TAGS-eng: BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES
  Stream #0:1(eng): Audio: ac3, 48000 Hz, 5.1(side), fltp, 640 kb/s (default)
      Metadata:
        BPS             : 640000
        BPS-eng         : 640000
        DURATION        : 00:22:49.408000000
        DURATION-eng    : 00:22:49.408000000
        NUMBER_OF_FRAMES: 42794
        NUMBER_OF_FRAMES-eng: 42794
        NUMBER_OF_BYTES : 109552640
        NUMBER_OF_BYTES-eng: 109552640
        _STATISTICS_WRITING_APP: mkvmerge v7.5.0 ('Glass Culture') 64bit built on Jan  4 2015 16:48:44
        _STATISTICS_WRITING_APP-eng: mkvmerge v7.5.0 ('Glass Culture') 64bit built on Jan  4 2015 16:48:44
        _STATISTICS_WRITING_DATE_UTC: 2016-02-24 01:10:32
        _STATISTICS_WRITING_DATE_UTC-eng: 2016-02-24 01:10:32
        _STATISTICS_TAGS: BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES
        _STATISTICS_TAGS-eng: BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES
  Stream #0:2(eng): Subtitle: hdmv_pgs_subtitle (pgssub) (default)
      Metadata:
        BPS             : 19398
        BPS-eng         : 19398
        DURATION        : 00:21:58.317000000
        DURATION-eng    : 00:21:58.317000000
        NUMBER_OF_FRAMES: 732
        NUMBER_OF_FRAMES-eng: 732
        NUMBER_OF_BYTES : 3196680
        NUMBER_OF_BYTES-eng: 3196680
        _STATISTICS_WRITING_APP: mkvmerge v7.5.0 ('Glass Culture') 64bit built on Jan  4 2015 16:48:44
        _STATISTICS_WRITING_APP-eng: mkvmerge v7.5.0 ('Glass Culture') 64bit built on Jan  4 2015 16:48:44
        _STATISTICS_WRITING_DATE_UTC: 2016-02-24 01:10:32
        _STATISTICS_WRITING_DATE_UTC-eng: 2016-02-24 01:10:32
        _STATISTICS_TAGS: BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES
        _STATISTICS_TAGS-eng: BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES
  Stream #0:3(chi): Subtitle: hdmv_pgs_subtitle (pgssub), 1920x1080
      Metadata:
        BPS             : 23664
        BPS-eng         : 23664
        DURATION        : 00:22:43.612000000
        DURATION-eng    : 00:22:43.612000000
        NUMBER_OF_FRAMES: 714
        NUMBER_OF_FRAMES-eng: 714
        NUMBER_OF_BYTES : 4033723
        NUMBER_OF_BYTES-eng: 4033723
        _STATISTICS_WRITING_APP: mkvmerge v7.5.0 ('Glass Culture') 64bit built on Jan  4 2015 16:48:44
        _STATISTICS_WRITING_APP-eng: mkvmerge v7.5.0 ('Glass Culture') 64bit built on Jan  4 2015 16:48:44
        _STATISTICS_WRITING_DATE_UTC: 2016-02-24 01:10:32
        _STATISTICS_WRITING_DATE_UTC-eng: 2016-02-24 01:10:32
        _STATISTICS_TAGS: BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES
        _STATISTICS_TAGS-eng: BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES
        
依据ECF数据集原先的视频片段格式 Friends_S1E1: 00:01:19.037 - 00:01:25.417 提取  原始视频名字不对的话 记得自已切换
"""
# 定义OpenSMILE可执行文件路径（根据实际安装路径修改）
opensmile_path = "D:/opensmile-3.0.2-windows-x86_64/bin/SMILExtract.exe"

# 定义配置文件路径（这里以IS13_ComParE.conf为例，可按需替换）这里看看以后有没有什么其他的
config_file_path = "D:/opensmile-3.0.2-windows-x86_64/config/is09-13/IS13_ComParE.conf"

# 临时的输出文件路径（以CSV格式作为中间过渡）
temp_output_csv_file = "feature/audio_feature.csv"


def extract_video_name(video_info_str):
    """
    从包含视频信息的字符串中提取视频名
    :param video_info_str: 类似 Friends_S1E1: 00:01:19.037 - 00:01:25.417 的字符串
    :return: 提取出的视频名
    """
    video_name_match = re.search(r'^([^:]+):', video_info_str)
    if video_name_match:
        return video_name_match.group(1)
    return None


def parse_time_segment(time_segment_str):
    """
    解析时间片段字符串，提取出起始时间和持续时间
    :param time_segment_str: 格式类似 Friends_S1E1: 00:01:19.037 - 00:01:25.417 的字符串
    :return: 起始时间字符串（格式如 '00:00:00'），持续时间字符串（格式如 '00:00:00'）
    """
    # 使用正则表达式匹配出时间部分
    time_match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3}) - (\d{2}:\d{2}:\d{2}\.\d{3})', time_segment_str)
    if time_match:
        start_time_str = time_match.group(1)
        end_time_str = time_match.group(2)
        start_time = start_time_str.split(':')
        end_time = end_time_str.split(':')
        start_second = int(start_time[2].split('.')[0])
        start_milliseconds = int(start_time[2].split('.')[1])
        end_second = int(end_time[2].split('.')[0])
        end_milliseconds = int(end_time[2].split('.')[1])
        # 计算持续时间
        start_seconds = int(start_time[0]) * 3600 + int(start_time[1]) * 60 + int(start_second)
        end_seconds = int(end_time[0]) * 3600 + int(end_time[1]) * 60 + int(end_second)
        duration_millisecond_tmp = (end_seconds * 1000 + end_milliseconds) - (start_seconds * 1000 + start_milliseconds)

        duration_milliseconds = duration_millisecond_tmp % 1000
        duration_seconds = int(duration_millisecond_tmp / 1000) % 60
        duration_minutes = int(duration_millisecond_tmp / 60000) % 60
        duration_hours = duration_millisecond_tmp // 3600000

        duration_str = f"{str(duration_hours).zfill(2)}:{str(duration_minutes).zfill(2)}:{str(int(duration_seconds)).zfill(2)}.{str(duration_milliseconds).zfill(3)}"

        return start_time_str, duration_str, duration_seconds, duration_milliseconds
    return None, None, None, None


def clip_video_subprocess(input_video, start_time, duration, output_video):
    """
    似乎源文件有些错误，导致一些视频片段无法获取 所以可以执行直接截取 对应帧的图片
    """
    command = ['ffmpeg', '-analyzeduration', '10M', '-probesize', '10M', '-i', input_video, '-ss', start_time, '-t',
               duration, '-vcodec', 'copy', '-an', output_video]
    print(command)
    subprocess.run(command)


def clip_audio_subprocess(input_video, start_time, duration, output_audio):
    # Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 44100 Hz, mono, s16, 705 kb/s
    command = ['ffmpeg', '-analyzeduration', '10M', '-probesize', '10M', '-i', input_video, '-ss', start_time, '-t',
               duration, '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1', '-vn', output_audio]
    print(command)
    subprocess.run(command)


def clip_jpg_subprocess(input_video, start_time, duration, output_audio):
    command = ['ffmpeg', '-analyzeduration', '10M', '-probesize', '10M', '-i', input_video, '-ss', start_time, '-vf',
               'fps=1', '-t', duration, f'{output_audio}frame%d.jpg']
    print(command)
    subprocess.run(command)


def audio_feature(input_wav_file):
    # 输入的WAV音频文件路径 Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 44100 Hz, mono, s16, 705 kb/s
    command1 = [
        opensmile_path,
        "-C", config_file_path,
        "-I", input_wav_file,
        "-O", temp_output_csv_file
    ]
    subprocess.run(command1, check=True)


class CustomVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if
                          f.endswith('.jpg') or f.endswith('.png')]  # 获取文件夹所有的图片
        print(len(self.image_dir))
        # 只要能获取到截取的图片就ok了

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        image_file = self.image_dir[idx]
        image_file = Image.open(image_file).convert('RGB')
        if self.transform:
            image_file = self.transform(image_file)
        return image_file


transform = transforms.Compose([
    # 这里添加你的预处理步骤，如调整大小、归一化等 这部分保持常规方法
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
dataset = CustomVideoDataset(root_dir='video_clip/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
# [49,768]
vit_model = "vit_base_patch32_224"  # 填写下载好的ViT模型
pretrained_vit_model = timm.create_model(vit_model).to(device)
pretrained_vit_model.eval()


def vit_forward(x):
    x = pretrained_vit_model.patch_embed(x).to(device)
    # print("x", x.shape)
    # print("cls_token", pretrained_vit_model.cls_token.shape)
    cls_token = pretrained_vit_model.cls_token.expand(x.shape[0], -1, -1).to(device)
    x = torch.cat([cls_token, x], dim=1)
    x = pretrained_vit_model.pos_drop(x + pretrained_vit_model.pos_embed).to(device)
    # print("pretrained_vit_model.pos_embed", pretrained_vit_model.pos_embed.shape, pretrained_vit_model.pos_embed.dtype)
    x = pretrained_vit_model.blocks(x).to(device)
    x = pretrained_vit_model.norm(x).to(device)
    return x[:, 1:].to(device)


# 服务器模型 这个比较大，会输出[196,768]的大小 先用卷积操作缩回49的大小，先保持原先的代码不变吧 ，如果有修改再说
# vit_model = "/storage/nfs2/ModelHub/openai/clip-vit-base-patch16"
# pretrained_vit_model = CLIPVisionModel.from_pretrained(vit_model)
# pretrained_vit_model.eval()
#
# def vit_forward(x):
#     x = pretrained_vit_model.vision_model.embeddings.patch_embedding(x)
#     print(x.shape)
#     class_embedding = pretrained_vit_model.vision_model.embeddings.class_embedding
#     print(class_embedding)
#     cls_token = class_embedding.expand(x.shape[0], 1, -1)
#     x = torch.cat([cls_token, x], dim=1)
#     x = pretrained_vit_model.vision_model.embeddings.dropout(x + pretrained_vit_model.vision_model.embeddings.position_embedding(x))
#     x = pretrained_vit_model.vision_model.encoder(x)
#     x = pretrained_vit_model.vision_model.layernorm_embedding(x)
#     return x[:, 1:]


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


# 2. 加载Hubert模型
model_name = "hubert-large-ls960"  # 这里选择一个预训练的Hubert模型，可根据需求换
config = HubertConfig.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name, config=config)
model.eval()  # 设置为评估模式


def audio_feature_hubert(wav_file_path, j):
    processed_audio = load_and_process_audio(wav_file_path)

    # 3. 将处理后的音频输入Hubert模型进行编码
    with torch.no_grad():
        # 添加批次维度，形状变为 (1, 1, sequence_length)，符合模型输入要求
        input_audio = torch.from_numpy(processed_audio)
        outputs = model(input_audio)
        encoding_result = outputs.last_hidden_state

    # 4. 存储Hubert的编码结果
    output_dir = f"src/audio_Hubert/output_{j}.npy"
    np.save(output_dir, encoding_result.detach().cpu().numpy())


if __name__ == '__main__':
    # test = "Friends_S1E1: 00:00:19.417 - 00:01:25.307"
    # print(parse_time_segment(test))
    # print(extract_video_name(test))
    # 先依据片段截取全部视频片段， 然后再按照顺序逐个提取特征，
    image_len = []
    data = open("data/all_data_pair.txt", 'r', encoding='utf-8')
    # command = ['ffmpeg', '-i', "original_video/Friends_S1E1.mkv"]
    # subprocess.run(command)
    # Stream #0:0: Video: hevc (Main 10), yuv420p10le(tv), 1920x1080, SAR 1:1 DAR 16:9, 23.98 fps, 23.98 tbr, 1k tbn (default)
    flag = 1
    st = 0
    while True:
        line = data.readline()
        if line == '':
            break
        line = line.strip().split()
        d_id, d_len = line[0], int(line[1])
        line = data.readline()
        for i in range(d_len):
            utter_data = data.readline().strip().split(' | ')
            origin_clip = utter_data[4]
            clip_name = extract_video_name(origin_clip)
            clip_time_st, clip_time_dur, clip_time_second, clip_time_milliseconds = parse_time_segment(origin_clip)
            output_clip_video = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + ".mkv")
            output_clip_audio = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + ".wav")
            output_clip_name = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + "_")
            input_video = os.path.join("original_video/", clip_name + ".mkv")
            # if os.path.exists(output_clip_name): continue
            # if os.path.exists(output_clip_audio): continue
            print(len(image_len), clip_time_st, clip_time_dur, clip_time_second, clip_time_milliseconds)

            if clip_time_milliseconds / 100 >= 5:
                clip_time_second += 1
            else:
                if clip_time_second == 0:
                    clip_time_second += 1
            st += clip_time_second
            image_len.append(st)
            if os.path.exists(output_clip_audio):
                continue
            if not os.path.exists(input_video):
                flag = 0
                break
            clip_jpg_subprocess(input_video, clip_time_st, clip_time_dur, output_clip_name)
            # clip_video_subprocess(input_video, clip_time_st, clip_time_dur, output_clip_name)
            clip_audio_subprocess(input_video, clip_time_st, clip_time_dur, output_clip_audio)

        if flag == 0:
            break
    print("成功执行视频截取")
    # 如果使用Hubert截取

    flag = 1
    data = open("data/all_data_pair.txt", 'r', encoding='utf-8')
    j = 1
    while True:
        line = data.readline()
        if line == '':
            break
        line = line.strip().split()
        d_id, d_len = line[0], int(line[1])
        line = data.readline()
        for i in range(d_len):
            utter_data = data.readline().strip().split(' | ')
            origin_clip = utter_data[4]
            clip_name = extract_video_name(origin_clip)
            clip_time_st, clip_time_dur, clip_time_second, clip_time_milliseconds = parse_time_segment(origin_clip)
            output_clip_video = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + ".mkv")
            audio_file = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + ".wav")
            output_clip_name = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + "_")

            if not os.path.exists(audio_file):
                flag = 0
                break
            # if os.path.exists(output_clip_name): continue
            # if os.path.exists(output_clip_audio): continue
            if os.path.exists(f"src/audio_Hubert/output_{j}.npy"):
                continue
            audio_feature_hubert(audio_file, j)
            j += 1
        if flag == 0:
            break

    # 拆完 视频片段后 进行编码
    flag = 1
    data = open("data/all_data_pair.txt", 'r', encoding='utf-8')
    while True:
        line = data.readline()
        if line == '':
            break
        line = line.strip().split()
        d_id, d_len = line[0], int(line[1])
        line = data.readline()
        for i in range(d_len):
            utter_data = data.readline().strip().split(' | ')
            origin_clip = utter_data[4]
            clip_name = extract_video_name(origin_clip)
            clip_time_st, clip_time_dur, clip_time_second, clip_time_milliseconds = parse_time_segment(origin_clip)
            output_clip_video = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + ".mkv")
            audio_file = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + ".wav")
            output_clip_name = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + "_")
            if not os.path.exists(audio_file):
                flag = 0
                break
            # if os.path.exists(output_clip_name): continue
            # if os.path.exists(output_clip_audio): continue
            audio_feature(audio_file)
        if flag == 0:
            break

    import pandas as pd

    df = pd.read_csv(temp_output_csv_file, sep=',', skiprows=range(6380), header=None)  # 处理完了所有音频信息 可以用此方法提取全部内容。
    selected_data = df.iloc[:, 1:6374].to_numpy()
    selected_data = numpy.expand_dims(selected_data, axis=1)
    numpy.save("src/audio_feature.npy", selected_data)
    print(selected_data.shape)
    print("成功执行音频编码")
    print(image_len)
    print(len(image_len))
    # tensor_list = []
    image_tensor = []
    j = 0
    for idx, batch_images in enumerate(dataloader):
        # print("输入的维度", batch_images.size())
        # if idx >= image_len[len(image_len)-1]: break
        image_tensor.append(batch_images.to(device))
        # print("处理后的维度", vit_forward(batch_images).size())
        if idx == image_len[j] - 1:
            num = 0
            image_tensors = None
            print(len(image_tensor))
            file_name = f"src/video_feature/output_{j+1}.pt"
            if os.path.exists(file_name):
                j += 1
                image_tensor = []
                continue
            for t in image_tensor:
                print("当前处理内容", idx, j, image_len[j])
                result = vit_forward(t.to(device)).cpu()
                # 调整patch个数部分
                bs, grid_num, dim = result.shape
                if grid_num != 49:
                    conv1d = nn.Conv1d(grid_num, 49, kernel_size=4, stride=4)
                    # 需要将visual_output的维度进行转换，以符合Conv1d的输入要求
                    visual_output_transposed = result.transpose(1, 2)
                    # 进行一维卷积操作
                    compressed_output_transposed = conv1d(visual_output_transposed)
                    # 将输出张量的副本维度转换回原始格式，最终输出形状为[batch_size, 49, patch_dim]
                    result = compressed_output_transposed.transpose(1, 2)
                num += 1
                if image_tensors is None:
                    image_tensors = result
                else:
                    image_tensors = torch.add(image_tensors, result)
            # print(image_tensors.size())
            if num != 0: image_tensors = image_tensors / num
            j += 1
            torch.save(image_tensors, file_name)
            image_tensor = []
            print("完成一句话的处理", j)
    # numpy_array_list = numpy.array([t.detach().numpy() for t in tensor_list])
    # print(numpy_array_list.shape)
    # numpy.save("src/video_feature.npy", numpy_array_list)
    print("成功执行视频编码")
