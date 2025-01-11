import os
import re

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


class CustomVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if
                          f.endswith('.jpg') or f.endswith('.png')]  # 获取文件夹所有的图片
        # print(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        image_file = self.image_dir[idx]
        image_file = Image.open(image_file).convert('RGB')
        if self.transform:
            image_file = self.transform(image_file)
        return image_file


transform = transforms.Compose([
    # 这里添加你的预处理步骤，如调整大小、归一化等
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
dataset = CustomVideoDataset(root_dir='video_clip/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
# 先使用和原论文一样的 vit模型，之后可以考虑扩展到 clip-vit这个本身就是多模态训练的模型上，或是增加vit的参数量，比如large 14这些
vit_model = "vit_base_patch32_224"
pretrained_vit_model = timm.create_model(vit_model).to(device)
pretrained_vit_model.eval()
#
# vit_model = "/storage/nfs2/ModelHub/openai/clip-vit-base-patch16"
# pretrained_vit_model = CLIPVisionModel.from_pretrained(vit_model).to(device)
# pretrained_vit_model.eval()


# def vit_forward(x):
#     x = pretrained_vit_model.vision_model.embeddings.patch_embedding(x)
#     print(x.shape, x.dtype)  # [bs, dim, 14, 14] -> [bs, 196, dim] 按照vit-32-224的维度大小进行操作
#     x = x.contiguous().view(x.shape[0], 196, -1).to(device)
#     class_embedding = pretrained_vit_model.vision_model.embeddings.class_embedding
#     print(class_embedding.shape, class_embedding.dtype)  # [dim] -> [bs, 1, dim]
#     class_embedding.unsqueeze(0).unsqueeze(1)
#     cls_token = class_embedding.expand(x.shape[0], 1, -1).to(device)
#     x = torch.cat([cls_token, x], dim=1)
#     print("pretrained_vit_model.vision_model.embeddings.position_embedding(x)",
#           pretrained_vit_model.vision_model.embeddings.position_embedding(x.long()).dtype,
#           pretrained_vit_model.vision_model.embeddings.position_embedding(x.long()).shape)
#     x = nn.Dropout(x + pretrained_vit_model.vision_model.embeddings.position_embedding(x.long()).float32())
#     x = pretrained_vit_model.vision_model.encoder(x)
#     x = pretrained_vit_model.vision_model.layernorm_embedding(x)
#     return x[:, 1:].to(device)


image_len = []
st = 0
flag = 1
data = open("data/all_data_pair.txt", 'r', encoding='utf-8')
# idx = 0
while True:
    line = data.readline()
    if line == '':
        break
    line = line.strip().split()
    d_id, d_len = line[0], int(line[1])
    line = data.readline()
    for i in range(d_len):
        if d_id == '32':
            flag = 0
            break
        utter_data = data.readline().strip().split(' | ')
        origin_clip = utter_data[4]
        clip_name = extract_video_name(origin_clip)
        clip_time_st, clip_time_dur, clip_time_second, clip_time_milliseconds = parse_time_segment(origin_clip)
        output_clip_video = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + ".mkv")
        output_clip_audio = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + ".wav")
        output_clip_name = os.path.join("video_clip/", clip_name + "_" + str(d_id) + "_" + str(i + 1) + "_")
        input_video = os.path.join("original_video/", clip_name + ".mkv")
        # if not os.path.exists(input_video):
        #     flag = 0
        #     break
        #  爆论， 好像ffmpeg的分割采用四舍五入的原则，我们采用1秒一帧的话，如果是6380ms 只截取6帧， 6923ms则是7帧， 如果有问题再说吧
        if clip_time_milliseconds / 100 >= 5:
            clip_time_second += 1
        else:
            if clip_time_second == 0:
                clip_time_second += 1
        st += clip_time_second
        image_len.append(st)
    if flag == 0:
        break

#
def vit_forward(x):
    x = pretrained_vit_model.patch_embed(x).to(device)
    cls_token = pretrained_vit_model.cls_token.expand(x.shape[0], -1, -1).to(device)
    x = torch.cat([cls_token, x], dim=1).to(device)
    x = pretrained_vit_model.pos_drop(x + pretrained_vit_model.pos_embed).to(device)
    x = pretrained_vit_model.blocks(x).to(device)
    x = pretrained_vit_model.norm(x).to(device)
    return x[:, 1:].to(device)


# 直接单个写入吧，内容太大了， 49*768 = 37632 >> 4096
# tensor_list = []
image_tensor = []
j = 0
for idx, batch_images in enumerate(dataloader):
    # print("输入的维度", batch_images.size())
    image_tensor.append(batch_images.to(device))
    # print("处理后的维度", vit_forward(batch_images.to(device)).size())
    if idx == image_len[j] - 1:
        j += 1
        num = 0
        image_tensors = None
        for t in image_tensor:
            result = vit_forward(t.to(device)).cpu()
            bs, grid_num, dim = result.shape
            if grid_num != 49:
                # 如果切换了模型，让grid_num数量增加，也许需要调整一下大小 ? 保留意见
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
        image_tensors = image_tensors / num
        file_name = f"src/video_feature/output_{j}.pt"
        torch.save(image_tensors, file_name)
        # tensor_list.append(image_tensors)
        image_tensor = []
        print("完成一句话的处理", j)
# numpy_array_list = numpy.array([t.detach().numpy() for t in tensor_list])
# print(numpy_array_list.shape)
# numpy.save("src/video_feature.npy", numpy_array_list)
print("成功执行视频编码")
