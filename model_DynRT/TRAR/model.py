import torch.nn as nn
import torch
from model_DynRT.TRAR.trar import DynRT_ED
from model_DynRT.TRAR.cls_layer import *


class DynRT(nn.Module):
    def __init__(self, hyp_params):
        super(DynRT, self).__init__()
        self.backbone = DynRT_ED(hyp_params)

        if hyp_params.classifier == 'all':
            self.cls_layer = cls_layer_all(hyp_params.hidden_size, hyp_params.output_size)
        elif hyp_params.classifier == 'text':
            self.cls_layer = cls_layer_text(hyp_params.hidden_size, hyp_params.output_size)
        elif hyp_params.classifier == 'audio':
            self.cls_layer = cls_layer_audio(hyp_params.hidden_size, hyp_params.output_size)
        elif hyp_params.classifier == 'video':
            self.cls_layer = cls_layer_video(hyp_params.hidden_size, hyp_params.output_size)
        elif hyp_params.classifier == 'video_audio':
            self.cls_layer = cls_layer_video_audio(hyp_params.hidden_size, hyp_params.output_size)
        elif hyp_params.classifier == 'text_audio':
            self.cls_layer = cls_layer_text_audio(hyp_params.hidden_size, hyp_params.output_size)
        elif hyp_params.classifier == 'text_video':
            self.cls_layer = cls_layer_text_video(hyp_params.hidden_size, hyp_params.output_size)

    def forward(self, img_feat, lang_feat, audio_feat, conversation_mask, missing_mod, x_v_mask, attention_mask, audio_mask):
        """
        img_feat = [bs, max_conversation_len, 49, 768]
        conversation_mask 只用于指示batch对话中，每组对话的个数的填充项 需要提供的是 [bs, conversation_len, max_len] 然后以相同方式扩充
        lang_feat = [bs, max_conversation_len, max_len, 768] 填充项为0
        audio_feat = [bs, max_conversation_len, 6373]
        """
        img_feat_mask = []
        conversation_mask = conversation_mask.unsqueeze(2).unsqueeze(3)
        audio_mask = audio_mask.unsqueeze(2).unsqueeze(3)
        if img_feat is not None:
            img_feat_mask = torch.zeros([img_feat.shape[0], img_feat.shape[1], 1, 1, img_feat.shape[2]], dtype=torch.bool,
                                        device=img_feat.device)
        # (bs, conversation, 1, 1, grid_num)
        #  经过动态路由的计算
        lang_feat, img_feat, audio_feat = self.backbone(
            lang_feat,
            img_feat,
            audio_feat,
            conversation_mask,
            img_feat_mask,
            missing_mod,
            x_v_mask,
            attention_mask,
            audio_mask
        )
        #  这个部分是最后的分类部分 各个参数经过平均化， 然后先层归一化 执行一个线性层 如果我们的 MECPE的话 ，现在就先以 情绪句子 直接连接 原因句子的各个模态表示先。
        # 这里才处理长度的问题，取平均得到每个句子的表示信息。 需不需要乘以mask来屏蔽可能存在的问题呢
        # print("lang_feat", lang_feat)

        lang_feat = torch.mean(lang_feat, dim=2)
        img_feat = torch.mean(img_feat, dim=2)
        audio_feat = torch.mean(audio_feat, dim=2)

        proj_feat = self.cls_layer(lang_feat, img_feat, audio_feat)  # 模态整合后的修改
        # 处理完每一个句子的特征信息
        return proj_feat, lang_feat, img_feat, audio_feat
