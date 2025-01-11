import argparse
import json
import random

import numpy
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

from prepare_data import *
# from bert import modeling, optimization, tokenization
import config
import train

parser = argparse.ArgumentParser(description="Missing modality with emotion cause prediction")

# Tasks  missing_model 存在缺失模态的话，使用PromptDynRTModel 预训练的基准模型是通过在ECF上，手动训练六种缺失情况得到的 然后用RECCON验证
# 当然如果只是单纯的检验DynRT的结果，或者说不缺失模态，单单验证DynRT的性能，使用DynRTMulTModel 并将missing_model置为6
"""
    0 -> 文本补充
    1 -> 音频补充
    2 -> 视频补充
    3 -> 文本 音频 补充
    4 -> 文本 视频 补充 
    5 -> 音频 视频 补充
    6 -> 全部存在
    7 -> 不使用promptDynRT的方法 保持特征不被添加任何prompt信息
    8 -> 联合学习，将0-6的所有情况一起学习。
    9 -> 随机赋值，于原论文的方法保持一致，每个输入数据的缺失模态情况随机赋值
"""
parser.add_argument(
    "--pretrained_model",
    type=str,
    default="DynRTMulTModel",
    help="name of the model to use (PromptDynRTModel, DynRTMulTModel)",
)
parser.add_argument(
    "--audio_type",
    type=str,
    default="opensmile",
    help="dataset to use (opensmile, Hubert)",
)
# Plus 在情绪识别上还可以用上原先代码使用的数据集， 不过为了保持统一 可能需要多一个维度 赋值为1，保持代码格式不变
parser.add_argument(
    "--dataset",
    type=str,
    default="ECF",
    help="dataset to use (ECF,ECF_cause,RECCON)",
)
parser.add_argument("--missing_model", type=int, default=6, help="missing_modal type")

parser.add_argument(
    "--data_path",
    type=str,
    default="result/",
    help="path for storing the dataset",
)

# Dropouts
parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout")
parser.add_argument(
    "--attn_dropout_a", type=float, default=0.1, help="attention dropout (for audio)"
)
parser.add_argument(
    "--attn_dropout_v", type=float, default=0.1, help="attention dropout (for visual)"
)
parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
parser.add_argument(
    "--embed_dropout", type=float, default=0.25, help="embedding dropout"
)
parser.add_argument(
    "--res_dropout", type=float, default=0.1, help="residual block dropout"
)
parser.add_argument(
    "--out_dropout", type=float, default=0.1, help="output layer dropout"
)

# Architecture
parser.add_argument(
    "--nlevels", type=int, default=5, help="number of layers in the network"
)
# 取这么小吗？？
parser.add_argument(
    "--proj_dim", type=int, default=30, help="projection dimension of the network"
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=5,
    help="number of heads for the transformer network",
)
parser.add_argument(
    "--attn_mask", action="store_false", help="use attention mask for Transformer"
)
# 此值需要和hidden_size一致即可 原文数值是30.。
parser.add_argument("--prompt_dim", type=int, default=768)
parser.add_argument("--prompt_length", type=int, default=16)
parser.add_argument("--path", type=str, default="data/")

# Tuning  batch_size 先小一点 最后看着增加就好。
parser.add_argument(
    "--batch_size", type=int, default=2, metavar="N", help="batch size"
)
parser.add_argument("--clip", type=float, default=0.8, help="gradient clip value")
parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--optim", type=str, default="AdamW", help="optimizer to use")
parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--when", type=int, default=10, help="when to decay learning rate")
parser.add_argument("--drop_rate", type=float, default=0.6)


# Dyn_RT
parser.add_argument("--hidden_size", type=int, default=768, help="Model's hidden size")
parser.add_argument("--output_size", type=int, default=768, help="DynRT Result")
parser.add_argument("--classifier", type=str, default="all", help="How to use Modality")
parser.add_argument("--tau_max", type=int, default=10, help="Gumble Softmax ")
parser.add_argument("--layer", type=int, default=4, help="Dyn_RT layers")
parser.add_argument("--ORDERS", type=int, default=[0, 1, 2, 3], help="Dyn_RT layers")
parser.add_argument("--orders", type=int, default=4, help="Dyn_RT layers")
parser.add_argument("--IMG_SCALE", type=int, default=7, help="ViT Scale")
parser.add_argument("--len", type=int, default=100, help="len of sentences")
parser.add_argument("--dropout", type=float, default=0.6, help="DynRT dropout")
parser.add_argument("--routing", type=str, default="hard", help="The way use in Multi-Head Co-Attention")
parser.add_argument("--pooling", type=str, default="avg", help="The way use in Multi-Head Co-Attention")
parser.add_argument("--BINARIZE", type=str, default="false", help="The way use in Multi-Head Co-Attention")
parser.add_argument("--multihead", type=int, default=4, help="")
parser.add_argument("--ffn_size", type=int, default=768, help="FFN ffn_Size")

# Logistics
parser.add_argument(
    "--log_interval",
    type=int,
    default=30,
    help="frequency of result logging (default: 30)",
)
parser.add_argument("--seed", type=int, default=3407, help="random seed")
parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
parser.add_argument("--name", type=str, default="result/DynRT_ECF", help="name of the trial")
args = parser.parse_args()

dataset = args.dataset.strip()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


torch.set_default_tensor_type("torch.FloatTensor")
use_cuda = False
if torch.cuda.is_available():
    if args.no_cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably not run with --no_cuda"
        )
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        use_cuda = True

setup_seed(args.seed)
emotion_idx = dict(zip(['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'], range(7)))

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


hyp_params = args
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.log_interval = args.log_interval
hyp_params.clip = args.clip
hyp_params.criterion = "MaskedCrossEntropyLoss"
hyp_params.hidden_size = args.hidden_size
hyp_params.output_size = args.output_size
hyp_params.dropout = args.dropout
hyp_params.classifier = args.classifier
hyp_params.IMG_SCALE = args.IMG_SCALE
hyp_params.len = args.len
if args.pretrained_model == "PromptDynRTModel":
    hyp_params.len = args.len + args.prompt_length
hyp_params.routing = args.routing
hyp_params.pooling = args.pooling
hyp_params.BINARIZE = args.BINARIZE
hyp_params.multihead = args.multihead
hyp_params.ffn_size = args.ffn_size
hyp_params.path = args.path
hyp_params.batch_size = args.batch_size
hyp_params.name = args.name
hyp_params.missing_model = args.missing_model
hyp_params.num_epochs = args.num_epochs
hyp_params.video_idx_file = "data/video_id_mapping.npy"
hyp_params.audio_emb_file = "src/audio_feature.npy"
hyp_params.audio_type = args.audio_type
# 暂使用ECF提供的数据
hyp_params.video_emb_file = "D:\Desktop\研一内容\论文对应代码\\2023-MECPE 多模态情绪原因对提取\MECPE-main\data\\video_embedding_4096.npy"

hyp_params.v_len = 49
hyp_params.orig_d_l = 768
if args.audio_type == "opensmile":
    hyp_params.orig_d_a = 6373
    hyp_params.a_len = 1
elif args.audio_type == "Hubert":
    # 先遍历一次Hubert的结果得到最大长度先吧， 好像没有什么好的截取方法，
    hyp_params.orig_d_a = 1024
    hyp_params.a_len = 769
hyp_params.orig_d_v = 768
hyp_params.finetune = "None"
hyp_params.prompt_dim = args.prompt_dim
hyp_params.prompt_length = args.prompt_length


class CauseDataset(Dataset):
    def __init__(self, data_file_name, tokenizer, video_idx, spe_idx, hyp_params):
        super(CauseDataset, self).__init__()
        self.tokenizer = tokenizer
        self.utterance = []
        self.speaker = []
        self.token_ids = []
        self.attention_mask = []
        self.sentence_mask = []
        self.cause = []
        self.doc_id = []
        self.y_pairs = []
        self.adj_index = []
        self.conversation_len = []
        self.x_v = []
        self.x_v_mask = []  # 用于指示一个batch中的每一个对话实际包含的句子个数
        if hyp_params.dataset == "ECF_cause":
            raw_data_path = "data/ECF_all.json"
        if hyp_params.dataset == "RECCON":
            raw_data_path = "data/RECCON_all_data.json"
        with open(raw_data_path, 'r', encoding="utf-8") as f:
            self.raw_data_dict = json.load(f)
        data = self.raw_data_dict[data_file_name]
        dia_ids = list(data.keys())
        for dia_id in dia_ids:
            dia_dict = data[dia_id]  # 每一组的内容
            d_len = dia_dict['num_utt']
            y_emotion_tmp, y_cause_tmp = np.zeros((hyp_params.len, 7)), np.zeros((hyp_params.len, 2))
            x_v_tmp, sen_len_tmp, spe_tmp, x_v_mask_tmp = [np.zeros(hyp_params.len, dtype=np.int32) for _ in
                                                           range(4)]
            d_id = int(re.search(r'_(.*?)_', dia_id).group(1))
            self.doc_id.append(d_id)
            self.conversation_len.append(d_len)
            utter_tmp = []
            speak_tmp = []
            tk_id = []
            at_mk = []
            st_mk = []
            cau = []
            spk = []
            x_v_temp = []
            x_v_mask_temp = []
            y_pairs_tmp = []
            for i in range(d_len):
                if hyp_params.dataset == "ECF_cause":
                    x_v_tmp[i] = dia_dict['correspondence'][i]
                x_v_mask_tmp[i] = 1
                utter_data = dia_dict['content_list'][i]
                x_v_temp.append(x_v_tmp[i])
                x_v_mask_temp.append(x_v_mask_tmp[i])
                speaker = dia_dict['speaker_list'][i]
                speak_tmp.append(speaker)
                if hyp_params.dataset == "ECF_cause":
                    if speaker in spe_idx:
                        spe_tmp[i] = int(spe_idx[speaker])  # 通过之前构建的说话人字典，获取当前说话人的代表数字
                    else:
                        print('speaker {} error!'.format(speaker))
                spk.append(spe_tmp[i])

                if i+1 in dia_dict['pos_cause_utts']:
                    cau.append(1)
                    y_pairs_tmp.append([d_len, i])
                else:
                    cau.append(0)

                utter_tmp.append(utter_data)
                encoder_utter = tokenizer(utter_data)
                tid = encoder_utter.input_ids
                atm = encoder_utter.attention_mask
                tk_id.append(torch.tensor(tid, dtype=torch.long))
                at_mk.append(torch.tensor(atm, dtype=torch.long))

            spk = torch.tensor(spk)
            x_v_temp = torch.tensor(x_v_temp, dtype=torch.long)
            x_v_mask_temp = torch.tensor(x_v_mask_temp, dtype=torch.long)
            same_spk = spk.unsqueeze(1) == spk.unsqueeze(0)
            other_spk = same_spk.eq(False).float().tril(0)
            same_spk = same_spk.float().tril(0)

            spker = torch.stack([same_spk, other_spk], dim=0)
            # 包含两个二维数组 第一个指示是不是同一个说话人，所以它的对角线是1 第二个指示不同说话人的 这里的句子id 从1开始哦

            tk_id = pad_sequence(tk_id, batch_first=True, padding_value=1)
            at_mk = pad_sequence(at_mk, batch_first=True, padding_value=0)
            st_mk = at_mk
            cau = torch.tensor(cau, dtype=torch.long)
            self.y_pairs.append(y_pairs_tmp)
            self.utterance.append(utter_tmp)
            self.speaker.append(speak_tmp)
            self.adj_index.append(spker)
            self.token_ids.append(tk_id)
            self.attention_mask.append(at_mk)
            self.sentence_mask.append(st_mk)
            self.cause.append(cau)
            self.x_v.append(x_v_temp)
            self.x_v_mask.append(x_v_mask_temp)


    def __getitem__(self, item):
        utter = self.utterance[item]
        speaker = self.speaker[item]
        # [conv_len, utter_len]
        token_id = self.token_ids[item]
        att_mask = self.attention_mask[item]
        sentence_mask = self.sentence_mask[item]
        cause = self.cause[item]
        # [conv_len, conv_len]
        adj_idx = self.adj_index[item]
        conversation_len = self.conversation_len[item]
        y_pairs = self.y_pairs[item]
        doc_id = self.doc_id[item]
        x_v = self.x_v[item]
        x_v_mask = self.x_v_mask[item]
        return utter, token_id, att_mask, cause, adj_idx, conversation_len, y_pairs, doc_id, x_v, x_v_mask, speaker, sentence_mask

    def __len__(self):
        return len(self.cause)

def collate_fn_cause(data):
    sentence_mask = []
    utter = []
    speaker = []
    cause_label = []
    adj_index = []
    conversation_len = []
    y_pairs = []
    x_v = []
    x_v_mask = []
    batch_token_ids = []
    batch_attention_mask = []
    doc_id = []
    max_doc_token_len = 0
    max_doc_mask_len = 0
    max_conversation = 0
    # 先找出整个batch中最长的句子长度是多少 ，然后再次弄到一样长
    # 对于token中，一开始roberta预处理的时候，每一组对话的tokenid 已经变成一样长了，batch就是 只需要将每次的弄到共有的最大句子长度，每组对话
    # 的句子个数不需要怪，之后传给roberta处理到向量之后 会变成 一样的值
    for i, d in enumerate(data):
        if i == 0:
            max_doc_token_len = d[1].shape[1]
            max_doc_mask_len = d[2].shape[1]
            max_conversation = d[5]
        else:
            max_doc_mask_len = max(max_doc_token_len, d[2].shape[1])
            max_doc_token_len = max(max_doc_token_len, d[1].shape[1])
            max_conversation = max(max_conversation, d[5])
    # 为了和原始论文统一， 设定100作为最大长度，就不依据每一个batch调整了 ECF数据集最长的分词为93 就不考虑截取了，如果新的数据集超出了此限制
    # 就扩大最大长度， 不做裁剪吧 好像就算放开长度也没啥吧，maybe。
    max_doc_token_len = max(hyp_params.len, max_doc_token_len)
    max_doc_mask_len = max(hyp_params.len, max_doc_mask_len)
    for i, d in enumerate(data):
        # 这里遍历的是八组对话 他这个方法没有更新完全啊。
        token_ids = d[1]
        attention_mask = d[2]
        if token_ids.shape[1] < max_doc_token_len:
            token_ids = torch.cat(
                [token_ids, torch.ones(token_ids.shape[0], max_doc_token_len - token_ids.shape[1], dtype=torch.long)],
                dim=1)
        batch_token_ids.append(token_ids.clone().detach().requires_grad_(False).to(dtype=torch.long))
        if attention_mask.shape[1] < max_doc_mask_len:
            attention_mask = torch.cat([attention_mask,
                                        torch.zeros(attention_mask.shape[0], max_doc_mask_len - attention_mask.shape[1],
                                                    dtype=torch.long)], dim=1)

        batch_attention_mask.append(attention_mask.clone().detach().requires_grad_(False).to(dtype=torch.long))
        sentence_mask_tmp = attention_mask
        if d[5] < max_conversation:
            sentence_mask_tmp = torch.cat([sentence_mask_tmp, torch.zeros(max_conversation - d[5], max_doc_mask_len,
                                                                          dtype=torch.int32)], dim=0)
        sentence_mask.append(sentence_mask_tmp)
        cause_label.append(d[3])
        # adj_index -> [2, conv_len, conv_len]
        adj_index.append(d[4])
        conversation_len.append(d[5])
        #  dev数据中有一个特殊的，他没有情绪-原因对，然后会让batch却一部分，进而没有计算这部分。。
        y_pairs.append(d[6])
        doc_id.append(d[7])
        x_v.append(d[8])
        x_v_mask.append(d[9])
        speaker.append(d[10])
        utter.append(d[0])
    cause_label = pad_sequence(cause_label, batch_first=True, padding_value=-1)
    x_v = pad_sequence(x_v, batch_first=True, padding_value=-1)
    x_v_mask = pad_sequence(x_v_mask, batch_first=True, padding_value=0)
    max_len = max(conversation_len)
    # x_v = [torch.cat([torch.cat([yp, torch.zeros(max_len-yp.shape[0], yp.shape[1])], dim=0),
    #                        torch.zeros(max_len, max_len-yp.shape[1])], dim=1) for yp in x_v]
    # x_v = torch.stack(x_v, dim=0)
    adj_index = [torch.cat([torch.cat([a, torch.zeros(2, max_len - a.shape[1], a.shape[2])], dim=1),
                            torch.zeros(2, max_len, max_len - a.shape[2])], dim=2) for a in adj_index]
    # [batch_size, 2, conv_len, conv_len]
    adj_index = torch.stack(adj_index, dim=0)
    sentence_mask = torch.stack(sentence_mask, dim=0)
    doc_id = torch.tensor(doc_id, dtype=torch.long)
    conversation_len = torch.tensor(conversation_len, dtype=torch.long)
    # ece_pair 的二维数组从0 开始计数，而数据集中的句子从1 开始， 如果要用记得加1
    return batch_token_ids, batch_attention_mask, conversation_len, adj_index, cause_label, y_pairs, doc_id, x_v, x_v_mask, utter, speaker, sentence_mask


class BaseDataset(Dataset):
    def __init__(self, data_file_name, tokenizer, video_idx, spe_idx, hyp_params):
        super(BaseDataset, self).__init__()
        # 缺少再来补吧 内容还包含了 情绪原因对一起预测的部分，即自已预测情绪和原因的方法， 但是现在只用在情绪的识别上
        self.tokenizer = tokenizer
        self.utterance = []
        self.speaker = []
        self.token_ids = []
        self.attention_mask = []
        self.sentence_mask = []
        self.emotion = []
        self.cause = []
        self.adj_index = []
        self.evidence = []
        self.mask = []
        self.doc_id = []
        self.y_pairs = []
        self.conversation_len = []
        self.x_v = []
        self.x_v_mask = []  # 用于指示一个batch中的每一个对话实际包含的句子个数
        data = open(hyp_params.path + data_file_name + '_test.txt', 'r', encoding='utf-8')
        # num_emo, num_emo_cause, num_pairs = [0 for _ in range(6)]
        while True:
            line = data.readline()
            if line == '':
                break
            line = line.strip().split()
            d_id, d_len = line[0], int(line[1])
            d_id = int(d_id)
            self.doc_id.append(d_id)
            self.conversation_len.append(d_len)

            pairs = eval('[' + data.readline().strip() + ']')
            pair_emo, cause = [], []
            ev_vic = numpy.zeros((d_len, d_len))
            if pairs != []:
                if len(pairs[0]) > 2:
                    # 处理异常情况， 如果序列的一个元素中长度大于2，也就是 （1，2，3）的情况 只取前面两个值
                    pairs = [(p[0], p[1]) for p in pairs]
                    pairs = sorted(list(set(pairs)))  # 然后排序去重
                pair_emo, cause = zip(*pairs)  # 最后得到情绪所在的话语，额 原因所在的话语
            for p in pairs:
                ev_vic[p[0] - 1][p[1] - 1] = 1  # 注意ev_vic的值因为句子id从1 开始 所以先提前减去1了 后续还要用需要把1加回来

            pairs = torch.tensor(pairs)
            self.y_pairs.append(pairs)  # 存储所有的情绪-原因对，
            # num_pairs += len(pairs)  # 个数
            # num_emo_cause += len(list(set(pair_emo)))  # 情绪话语共有几句
            y_emotion_tmp, y_cause_tmp = np.zeros((hyp_params.len, 7)), np.zeros((hyp_params.len, 2))

            # if config.choose_emocate:  # 是否选择更细致的情绪分类方式
            #     y_emotion_tmp = np.zeros((config.max_doc_len, 7))

            x_v_tmp, sen_len_tmp, spe_tmp, x_v_mask_tmp = [np.zeros(hyp_params.len, dtype=np.int32) for _ in
                                                           range(4)]
            # 暂存每组对话的token id 和注意力掩码
            utter_tmp = []
            speak_tmp = []
            tk_id = []
            at_mk = []
            st_mk = []
            emo = []
            cau = []
            spk = []
            x_v_temp = []
            x_v_mask_temp = []
            for i in range(d_len):
                x_v_tmp[i] = video_idx['dia{}utt{}'.format(int(self.doc_id[-1]), i + 1)]
                x_v_mask_tmp[i] = 1
                # video_idx 是对应一组对话中的一个话语的下标，doc_id最后一个元素记录当前对话是第几组，i+1表示第几句对话 进而确定维度对应的行数
                utter_data = data.readline().strip().split(' | ')
                x_v_temp.append(x_v_tmp[i])
                x_v_mask_temp.append(x_v_mask_tmp[i])
                speaker = utter_data[1]
                speak_tmp.append(speaker)
                if speaker in spe_idx:
                    spe_tmp[i] = int(spe_idx[speaker])  # 通过之前构建的说话人字典，获取当前说话人的代表数字
                else:
                    print('speaker {} error!'.format(speaker))
                spk.append(spe_tmp[i])

                if i + 1 in cause:
                    cau.append(1)
                else:
                    cau.append(0)

                emo_id = emotion_idx[utter_data[2]]  # 获取情绪所对应的数字
                # if emo_id > 0:
                #     num_emo += 1
                # if config.choose_emocate:
                y_emotion_tmp[i][emo_id] = 1  # 如果详细记录，就指定7个元素中的一个变为1
                # else:
                #     y_emotion_tmp[i] = [1, 0] if utter_data[2] == 'neutral' else [0, 1]
                # 如果只是需要确定是否有无情绪，就判断是否为neutral中立，
                y_cause_tmp[i][int(i + 1 in cause)] = 1
                # cause 是原因从句所在的话语 用于记录一个话语中那几句是原因从句

                utter = utter_data[3].replace('|', '')
                utter_tmp.append(utter)
                encoder_utter = tokenizer(utter)
                tid = encoder_utter.input_ids
                atm = encoder_utter.attention_mask
                tk_id.append(torch.tensor(tid, dtype=torch.long))
                at_mk.append(torch.tensor(atm, dtype=torch.long))

                emo.append(emo_id)

            ev_vic = torch.tensor(ev_vic, dtype=torch.long)
            msk = torch.ones_like(ev_vic, dtype=torch.long).tril(0)
            spk = torch.tensor(spk)
            x_v_temp = torch.tensor(x_v_temp, dtype=torch.long)
            x_v_mask_temp = torch.tensor(x_v_mask_temp, dtype=torch.long)
            same_spk = spk.unsqueeze(1) == spk.unsqueeze(0)
            other_spk = same_spk.eq(False).float().tril(0)
            same_spk = same_spk.float().tril(0)

            spker = torch.stack([same_spk, other_spk], dim=0)
            # 包含两个二维数组 第一个指示是不是同一个说话人，所以它的对角线是1 第二个指示不同说话人的 这里的句子id 从1开始哦

            tk_id = pad_sequence(tk_id, batch_first=True, padding_value=1)
            at_mk = pad_sequence(at_mk, batch_first=True, padding_value=0)
            st_mk = at_mk
            emo = torch.tensor(emo, dtype=torch.long)
            cau = torch.tensor(cau, dtype=torch.long)

            self.utterance.append(utter_tmp)
            self.speaker.append(speak_tmp)
            self.token_ids.append(tk_id)
            self.attention_mask.append(at_mk)
            self.sentence_mask.append(st_mk)
            self.emotion.append(emo)
            self.cause.append(cau)
            self.evidence.append(ev_vic)
            self.mask.append(msk)
            self.adj_index.append(spker)
            self.x_v.append(x_v_temp)
            self.x_v_mask.append(x_v_mask_temp)

    def __getitem__(self, item):
        utter = self.utterance[item]
        speaker = self.speaker[item]
        # [conv_len, utter_len]
        token_id = self.token_ids[item]
        att_mask = self.attention_mask[item]
        sentence_mask = self.sentence_mask[item]
        emo_label = self.emotion[item]
        cause = self.cause[item]
        # [conv_len, conv_len]
        adj_idx = self.adj_index[item]
        evidence = self.evidence[item]
        msk = self.mask[item]
        conversation_len = self.conversation_len[item]
        y_pairs = self.y_pairs[item]
        doc_id = self.doc_id[item]
        x_v = self.x_v[item]
        x_v_mask = self.x_v_mask[item]
        return utter, token_id, att_mask, emo_label, cause, adj_idx, evidence, msk, conversation_len, y_pairs, doc_id, x_v, x_v_mask, speaker, sentence_mask

    def __len__(self):
        return len(self.emotion)


def collate_fn(data):
    """
    token_id, att_mask, emo_label, adj_idx, evidence, msk, conversation_len, y_pair 只处理这些
    evidence 是以二维数组指示的情绪-原因对 y_pair则是具体数字对应的，暂时保留两者内容
    规定  一个batch出来的内容 是什么
    """
    token_ids = []
    attention_mask = []
    sentence_mask = []
    utter = []
    speaker = []
    emo_label = []
    cause_label = []
    adj_index = []
    ece_pair = []
    mask = []
    conversation_len = []
    y_pairs = []
    x_v = []
    x_v_mask = []
    batch_token_ids = []
    batch_attention_mask = []
    doc_id = []
    max_doc_token_len = 0
    max_doc_mask_len = 0
    max_conversation = 0
    # 先找出整个batch中最长的句子长度是多少 ，然后再次弄到一样长
    # 对于token中，一开始roberta预处理的时候，每一组对话的tokenid 已经变成一样长了，batch就是 只需要将每次的弄到共有的最大句子长度，每组对话
    # 的句子个数不需要怪，之后传给roberta处理到向量之后 会变成 一样的值
    for i, d in enumerate(data):
        if i == 0:
            max_doc_token_len = d[1].shape[1]
            max_doc_mask_len = d[2].shape[1]
            max_conversation = d[8]
        else:
            max_doc_mask_len = max(max_doc_token_len, d[2].shape[1])
            max_doc_token_len = max(max_doc_token_len, d[1].shape[1])
            max_conversation = max(max_conversation, d[8])
    # 为了和原始论文统一， 设定100作为最大长度，就不依据每一个batch调整了 ECF数据集最长的分词为93 就不考虑截取了，如果新的数据集超出了此限制
    # 就扩大最大长度， 不做裁剪吧
    max_doc_token_len = max(hyp_params.len, max_doc_token_len)
    max_doc_mask_len = max(hyp_params.len, max_doc_mask_len)
    for i, d in enumerate(data):
        # 这里遍历的是八组对话 他这个方法没有更新完全啊。
        token_ids = d[1]
        attention_mask = d[2]
        if token_ids.shape[1] < max_doc_token_len:
            token_ids = torch.cat(
                [token_ids, torch.ones(token_ids.shape[0], max_doc_token_len - token_ids.shape[1], dtype=torch.long)],
                dim=1)
        batch_token_ids.append(token_ids.clone().detach().requires_grad_(False).to(dtype=torch.long))
        if attention_mask.shape[1] < max_doc_mask_len:
            attention_mask = torch.cat([attention_mask,
                                        torch.zeros(attention_mask.shape[0], max_doc_mask_len - attention_mask.shape[1],
                                                    dtype=torch.long)], dim=1)

        batch_attention_mask.append(attention_mask.clone().detach().requires_grad_(False).to(dtype=torch.long))
        sentence_mask_tmp = attention_mask
        if d[8] < max_conversation:
            sentence_mask_tmp = torch.cat([sentence_mask_tmp, torch.zeros(max_conversation - d[8], max_doc_mask_len,
                                                                          dtype=torch.int32)], dim=0)
        sentence_mask.append(sentence_mask_tmp)
        emo_label.append(d[3])
        cause_label.append(d[4])
        # adj_index -> [2, conv_len, conv_len]
        adj_index.append(d[5])
        ece_pair.append(d[6])
        mask.append(d[7])
        conversation_len.append(d[8])
        #  dev数据中有一个特殊的，他没有情绪-原因对，然后会让batch却一部分，进而没有计算这部分。。
        y_pairs.append(d[9])
        doc_id.append(d[10])
        x_v.append(d[11])
        x_v_mask.append(d[12])
        speaker.append(d[13])
        utter.append(d[0])
    emo_label = pad_sequence(emo_label, batch_first=True, padding_value=-1)
    cause_label = pad_sequence(cause_label, batch_first=True, padding_value=-1)
    x_v = pad_sequence(x_v, batch_first=True, padding_value=-1)
    x_v_mask = pad_sequence(x_v_mask, batch_first=True, padding_value=0)
    max_len = max(conversation_len)

    # 每个部分的是一个二维数组 所以需要 在两个维度上进行填充操作 所以填充方法需要改变
    mask = [torch.cat([torch.cat([m, torch.zeros(max_len - m.shape[0], m.shape[1])], dim=0),
                       torch.zeros(max_len, max_len - m.shape[1])], dim=1) for m in mask]
    mask = torch.stack(mask, dim=0)
    ece_pair = [torch.cat([torch.cat([ep, torch.zeros(max_len - ep.shape[0], ep.shape[1])], dim=0),
                           torch.zeros(max_len, max_len - ep.shape[1])], dim=1) for ep in ece_pair]
    ece_pair = torch.stack(ece_pair, dim=0)

    # x_v = [torch.cat([torch.cat([yp, torch.zeros(max_len-yp.shape[0], yp.shape[1])], dim=0),
    #                        torch.zeros(max_len, max_len-yp.shape[1])], dim=1) for yp in x_v]
    # x_v = torch.stack(x_v, dim=0)
    adj_index = [torch.cat([torch.cat([a, torch.zeros(2, max_len - a.shape[1], a.shape[2])], dim=1),
                            torch.zeros(2, max_len, max_len - a.shape[2])], dim=2) for a in adj_index]
    # [batch_size, 2, conv_len, conv_len]
    adj_index = torch.stack(adj_index, dim=0)
    sentence_mask = torch.stack(sentence_mask, dim=0)
    doc_id = torch.tensor(doc_id, dtype=torch.long)
    conversation_len = torch.tensor(conversation_len, dtype=torch.long)
    # ece_pair 的二维数组从0 开始计数，而数据集中的句子从1 开始， 如果要用记得加1
    return batch_token_ids, batch_attention_mask, conversation_len, mask, adj_index, emo_label, cause_label, ece_pair, y_pairs, doc_id, x_v, x_v_mask, utter, speaker, sentence_mask


def get_dataloaders(tokenizer, video_idx, spe_idx, batch_size, hyp_params, dataset_type='ECF'):
    if dataset_type == 'ECF':
        train_set = BaseDataset('train', tokenizer, video_idx, spe_idx, hyp_params)
        dev_set = BaseDataset('dev', tokenizer, video_idx, spe_idx, hyp_params)
        test_set = BaseDataset('test', tokenizer, video_idx, spe_idx, hyp_params)
        # 因为预测结果还需要按照同样的顺序放回去，所以都不能打乱了
        hyp_params.n_train = len(train_set.doc_id)
        hyp_params.n_test = len(test_set.doc_id)
        hyp_params.n_valid = len(dev_set.doc_id)
        train_loader = DataLoader(train_set, batch_size, False, collate_fn=collate_fn)
        dev_loader = DataLoader(dev_set, batch_size, False, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size, False, collate_fn=collate_fn)
    else:
        train_set = CauseDataset('train', tokenizer, video_idx, spe_idx, hyp_params)
        dev_set = CauseDataset('dev', tokenizer, video_idx, spe_idx, hyp_params)
        test_set = CauseDataset('test', tokenizer, video_idx, spe_idx, hyp_params)
        hyp_params.n_train = len(train_set.doc_id)
        hyp_params.n_test = len(test_set.doc_id)
        hyp_params.n_valid = len(dev_set.doc_id)
        train_loader = DataLoader(train_set, batch_size, False, collate_fn=collate_fn_cause)
        dev_loader = DataLoader(dev_set, batch_size, False, collate_fn=collate_fn_cause)
        test_loader = DataLoader(test_set, batch_size, False, collate_fn=collate_fn_cause)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    word_idx_rev, word_idx, spe_idx_rev, spe_idx = load_w2v(hyp_params.hidden_size,
                                                            hyp_params.hidden_size,
                                                            hyp_params.path + 'all_data_pair.txt'
                                                            )

    video_idx, audio_embedding = load_embedding_from_npy(hyp_params.video_idx_file, hyp_params.audio_emb_file)
    audio_embedding = torch.tensor(audio_embedding, dtype=torch.float32)

    audio_embedding.to(device)

    tokenizer = RobertaTokenizer.from_pretrained('./roberta-base')
    train_loader, dev_loader, test_loader = get_dataloaders(tokenizer, video_idx, spe_idx, hyp_params.batch_size, hyp_params,
                                                            hyp_params.dataset)
    log_name = 'log_' + hyp_params.dataset + "_" + hyp_params.missing_model + '.txt'
    log_path = os.path.join("result/", log_name)

    log = open(log_path, 'w')
    log.write(str(hyp_params) + '\n\n')
    #  最后一个emotion 表示 是情绪句子的识别，  换成cause 需要调整相关数据集，并用于判别原因句子
    if hyp_params.missing_model != 6:
        train.initiate(hyp_params, train_loader, dev_loader, test_loader, audio_embedding, log)
    else:
        train.initiate(hyp_params, train_loader, dev_loader, test_loader, audio_embedding, log)