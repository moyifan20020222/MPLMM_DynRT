import os.path

import numpy as np
import torch
import timm
import model_DynRT
from transformers import RobertaModel
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


class DynRT(torch.nn.Module):
    # define model elements
    def __init__(self, hyp_params):
        """
        整体输入还是采用MECPE类似的架构， 文本不预先编码， 每一句话语的音频和视频模态编码信息 由 numpy数组中对应取出。不微调编码信息吧
        每个batch改成于每组对话长度最大值一致
        参数设计于 MuLT保持一致 每个句子的操作不做区分， 最后判断原因句子的时候，再结合情绪句子信息。
        """
        super(DynRT, self).__init__()
        self.utter_dim = hyp_params.hidden_size
        encoder_path = './roberta-base'
        self.encoder = RobertaModel.from_pretrained(encoder_path)
        self.mapping = nn.Sequential(nn.Linear(768, self.utter_dim, True), nn.LeakyReLU(negative_slope=0.01),
                                     nn.Dropout(0.1))
        self.hyp_params = hyp_params
        if not self.hyp_params.finetune:
            freeze_layers(self.encoder)

        self.trar = model_DynRT.TRAR.DynRT(hyp_params)
        self.sigm = torch.nn.Sigmoid()
        self.classifier_cause = torch.nn.Sequential(
            torch.nn.Dropout(0.6),
            torch.nn.Linear(hyp_params.output_size, 2)
        )
        self.classifier_emotion = torch.nn.Sequential(
            torch.nn.Dropout(0.6),
            torch.nn.Linear(hyp_params.output_size, 7)
        )
        # DynRT的一维卷积处理部分
        self.orig_d_l, self.orig_d_a, self.orig_d_v = (
            hyp_params.orig_d_l,
            hyp_params.orig_d_a,
            hyp_params.orig_d_v,
        )
        # 我们在输入之前就执行维度变换了， 统一成768的维度。
        self.d_l, self.d_a, self.d_v = (
            hyp_params.hidden_size,
            hyp_params.hidden_size,
            hyp_params.hidden_size,
        )
        self.proj_l = nn.Conv1d(
            self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False
        )
        self.proj_v = nn.Conv1d(
            self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False
        )
        self.cause = 0.5
        self.emotion = 0.5

    def roberta_forward(self, conv_utterance, attention_mask, conv_len):
        # conv_utterance: [[conv_len1, max_len1], [conv_len2, max_len2], ..., [conv_lenB, max_lenB]] 其中每一个内容是经过预处理的token id
        processed_output = []
        for conv_utterance, attention_mask in zip(conv_utterance, attention_mask):
            print("对话部分和掩码部分", conv_utterance.size(), attention_mask.size())
            # 需要稍微调整一下输出， DynRT的 文本编码在输入的时候仍然保留着整个句子长度的token编码。也就是不在此处执行池化操作

            # output_data = self.encoder(conv_utterance, attention_mask=attention_mask).last_hidden_state[:, -1, :]
            # 而是直接在编码完成后的第一层就可以获取了
            # print("测试输出内容", self.encoder(conv_utterance, attention_mask=attention_mask)[0].size())
            # TODO: 原始论文使用了第一层的输出内容， 这可以多实验一下 看看那一层比较好，总之只要保证长度这一维度信息别去掉就ok
            output_data = self.encoder(conv_utterance, attention_mask=attention_mask)[0]
            # mapped_output = self.mapping(output_data)  # [conv_len, token_dim] -> [conv_len, utter_dim]

            # 如果需要调整在此处修改，不过你也不会修改Roberta和ViT 的隐藏层维度的吧 ~(￣▽￣)~*
            output_data = output_data.transpose(1, 2)
            output_data = output_data if self.orig_d_l == self.d_l else self.proj_l(output_data)
            output_data = output_data.transpose(1, 2)

            print("output_data", output_data.size())
            processed_output.append(output_data)
        # [batch_size, max_conv_len, max_len, utter_dim]
        conv_output = pad_sequence(processed_output, batch_first=True)  # 对于一组对话中的话语。进行补足 补充不同对话的句子个数不同的情况
        # 稍微留意一下有没有需要修改的部分。在Dataloader部分
        return conv_output

    # forward propagate input
    def forward(self, token_ids, attention_mask, conversation_len, x_v, x_v_mask, audio_embedding, sentence_mask):
        """
        此处的输入为修改的batch输入， token_ids, attention_mask, conversation_len 指示了每一组batch里面 每一组对话的信息
        x_v_mask才是对应于每个batch中每组对话个数的mask信息 就是记录那些是对于batch中 对话长度的填充项
        需要提供的是 [bs, conversation_len, max_len] 不只是x_v_mask了
        embedding 记录预处理好的其他模态信息。
        x_v 记录每一个对话对应的其他模态的id

        在这里执行模态缺失的补充操作 此处
        """
        bert_text = self.roberta_forward(token_ids, attention_mask, conversation_len)
        """
        处理后的结果已经得到 每一个batch（对话个数） batch的对话中最大对话长度 话语编码信息。 所有为0 的表示填充项
        各个模态的大小 依照原论文方法 dim = 768 grid_num = 49
        """
        #   模态缺失的指示信息
        """
        0、文本生成 ， 1、音频生成 2、视频生成， 3、文本音频生成 4、文本视频生成 5、 视频音频生成 6、全部都存在
        可以随机生成这个值用于测试效果， 也可以单一设置，验证一种情况下的值
        """
        missing_mod = torch.zeros(x_v_mask.size(0), x_v_mask.size(1), dtype=torch.int32).to(device)
        if self.hyp_params.missing_model == 9:
            missing_mod = torch.randint(0, 7, [x_v_mask.size(0), x_v_mask.size(1)])
        else:
            for i, id in enumerate(x_v_mask):
                for j, ind in enumerate(id):
                    ind = ind.item()
                    if int(ind) != 0:
                        missing_mod[i, j] = self.hyp_params.missing_model
        # print(missing_mod)
        video_feature = torch.zeros(x_v_mask.size(0), x_v_mask.size(1), 49, 768, dtype=torch.float32).to(device)
        audio_feature = torch.zeros(x_v_mask.size(0), x_v_mask.size(1), 1, 768, dtype=torch.float32).to(device)
        audio_mask = torch.ones(x_v_mask.size(0), x_v_mask.size(1), 1, dtype=torch.bool).to(device)
        if self.hyp_params.dataset in ["ECF", "ECF_cause"]:
            if os.path.exists("src/video_feature/output_1.pt"):
                video_feature = torch.zeros(x_v_mask.size(0), x_v_mask.size(1), 49, 768, dtype=torch.float32).to(device)
                for i, id in enumerate(x_v):
                    for j, ind in enumerate(id):
                        ind = ind.item()
                        if int(ind) != -1:
                            # 先在三维上调整 再输入进四维张量
                            file = f"src/video_feature/output_{int(ind)}.pt"  # [49, 768]
                            tmp = torch.load(file)
                            tmp = tmp.transpose(0, 1)
                            tmp = tmp if self.orig_d_v == self.d_v else self.proj_v(tmp)
                            tmp = tmp.transpose(0, 1)
                            video_feature[i, j, ] = tmp
            # 这里再做一个改进，opensmile处理的音频特征和 Hubert处理的特征可以进行两次实验， Hubert是带长度的编码信息
            # 所以需要多指定一个参数，指定音频特征的选取，Hubert需要指定长度， 话说这里应该可以再改进一下， 就是我们的参数不是在
            # 每个部分都在传递吗，我们直接把每一个batch 的所需的最大长度 替换掉 a_len, l_len v_len这三个量，不就能减少一部分计算量了吗。
            # 讲道理，好像img_mask只是指示了每一个token的检测范围，好像并不是让每个token都与图片的每个中心点的所有mask计算的。maybe
            # 那这样的话，按理来说，文本长度 不足49 也是可以使用的呢。。
            if audio_embedding is not None and self.hyp_params.audio_type == "opensmile":
                audio_feature = torch.zeros(x_v_mask.size(0), x_v_mask.size(1), 1, 768, dtype=torch.float32).to(device)
                for i, id in enumerate(x_v):
                    for j, ind in enumerate(id):
                        ind = ind.item()
                        if int(ind) != -1 & int(ind) < audio_embedding.shape[0]:
                            # 先在三维上调整 再输入进四维张量
                            tmp = audio_embedding[int(ind), ]
                            tmp = tmp.transpose(0, 1)
                            tmp = tmp if self.orig_d_a == self.d_a else self.proj_a(tmp)
                            tmp = tmp.transpose(0, 1)
                            audio_feature[i, j, ] = tmp  # 注意赋值tensor的类型一致才行
            elif self.hyp_params.audio_type == "Hubert" and os.path.exists("src/audio_Hubert/output_1.npy"):
                audio_feature = torch.zeros(x_v_mask.size(0), x_v_mask.size(1), self.hyp_params.a_len, 768, dtype=torch.float32).to(device)
                audio_mask = torch.zeros(x_v_mask.size(0), x_v_mask.size(1), self.hyp_params.a_len, dtype=torch.bool).to(device)
                for i, id in enumerate(x_v):
                    for j, ind in enumerate(id):
                        ind = ind.item()
                        if int(ind) != -1:
                            file = f"src/audio_Hubert/output_{int(ind)}.npy"
                            tmp = torch.from_numpy(np.load(file)).squeeze(0)
                            len = tmp.size(0)
                            tmp = tmp.transpose(0, 1)
                            tmp = tmp if self.orig_d_a == self.d_a else self.proj_a(tmp)
                            tmp = tmp.transpose(0, 1)
                            ttmp = torch.zeros(self.hyp_params.a_len, self.d_a, dtype=torch.float32)
                            ttmp[:len, ] = tmp
                            audio_mask[i, j, :len] = 1
                            audio_feature[i, j, ] = ttmp
        print("维度检查")
        print(video_feature.size())
        print(bert_text.size())
        print(audio_feature.size())
        print(sentence_mask.size())
        print(conversation_len)
        if self.hyp_params.audio_type == "Hubert":
            print(audio_mask.size())
        (out1, lang_emb, img_emb, audio_emb) = self.trar(video_feature, bert_text, audio_feature, sentence_mask,
                                                         missing_mod, x_v_mask, attention_mask, audio_mask)

        # 原因的最后的ffn 可以再考虑一些别的想法， 总之前面的对于每个句子的 编码信息可以算是结束了
        # 这里可以在加上宽度学习，改掉这个ffn的部分。 而且用了这个原因句子识别也能更简单的融合情绪句子的隐藏层状态了， 不过注意情绪自已分析原因就不用加一份了 ..吧？

        result = None
        # print("out1",out1.size())
        if self.hyp_params.dataset in ["ECF_cause", "RECCON"]:
            """
            修正嵌入信息， 综合情绪和当前句子。 权重可调
            """
            for i, ind in enumerate(conversation_len):
                num = ind.item()
                # print("num", num)
                emo_embed = out1[i, num-1, ]
                for j in range(num):
                    out1[i, j, ] = out1[i, j, ] * self.cause + emo_embed * self.emotion
            result = self.classifier_cause(out1)
            # result = self.sigm(out)
        if self.hyp_params.dataset == "ECF":
            result = self.classifier_emotion(out1)
            # result = self.sigm(out)
        del bert_text, out1
        return result, lang_emb, img_emb, audio_emb


def build_DynRT(hyp_params):
    return DynRT(hyp_params)
