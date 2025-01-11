import torch.nn as nn
from model.TRAR.layer_norm import LayerNorm
import torch
import torch.nn.functional as F


"""
可以产生一下在这里 修正一下权重啥的。 而不是直接相加
"""


class cls_layer_video(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_video, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat, audio_feat):
        proj_feat = self.proj_norm(img_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


class cls_layer_text(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_text, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat, audio_feat):
        proj_feat = self.proj_norm(lang_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


class cls_layer_audio(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_audio, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat, audio_feat):
        proj_feat = self.proj_norm(audio_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


class cls_layer_text_video(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_text_video, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat, audio_feat):
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


class cls_layer_text_audio(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_text_audio, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat, audio_feat):
        proj_feat = lang_feat + audio_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


class cls_layer_video_audio(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_video_audio, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat, audio_feat):
        proj_feat = img_feat + audio_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


class cls_layer_all(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_all, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat, audio_feat):
        proj_feat = img_feat + audio_feat + lang_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat
