from model.TRAR.fc import MLP
import copy

from model.TRAR.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


class SoftRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(SoftRoutingBlock, self).__init__()
        self.pooling = pooling

        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)
        #  reduction 对于通道数进行一个缩减操作
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        # x 视频编码信息 img (bs, conversation_len, grid_num, dim) x_mask (bs,conversation_len, 1, max_len, grid_num) tau 10
        logits = []
        if self.pooling == 'avg':
            #   四维张量的处理，需要单独使用后三维度  直接处理长度
            output = torch.zeros_like(x)  # [(bs, conversation_len, grid_num, dim)]
            for i in range(x.shape[0]):
                tmp = x[i]  # (conversation_len, grid_num, dim)
                tmp = tmp.transpose(1, 2)
                tmp = self.pool(tmp)  # ( conversation_len, dim, 1)
                output[i] = tmp
            # x = x.transpose(2, 3)  # (bs, conversation_len, dim, max_len)
            # x = self.pool(x)
            logits = self.mlp(output.squeeze(-1))
            # (bs, conversation_len, dim, 1) -> # (bs, conversation_len, dim) -> # (bs, conversation_len, output_dim)
        elif self.pooling == 'fc':
            output = torch.zeros([x.shape[0], x.shape[1], x.shape[2], self.out_channel])
            for i in range(x.shape[0]):
                tmp = x[i]
                b, _, c = tmp.size()
                mask = self.make_mask(tmp).squeeze(1).squeeze(1).unsqueeze(2)  # (conversation_len, grid_num, 1)
                scores = self.pool(tmp)  # (conversation_len, grid_num, dim) -> (conversation_len, grid_num, 1)
                scores = scores.masked_fill(mask, -1e9)
                scores = F.softmax(scores, dim=1)
                _x = tmp.mul(scores)  # (conversation_len, grid_num, grid_num)
                tmp = torch.sum(_x, dim=1)  # (conversation_len, grid_num)
                logits = self.mlp(tmp)  # (conversation_len, grid_num, out_channel)
                output[i] = logits
            logits = output

        alpha = F.softmax(logits, dim=-1)  # 就是路由权重
        return alpha

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(2).unsqueeze(3)
        # (bs, conversation_len, max_len, dim) -> (bs, conversation_len, max_len) - > (bs, conversation_len, 1, 1, max_len) 对于绝对值之和为0 的张量记为0 不遮蔽


class HardRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(HardRoutingBlock, self).__init__()
        self.pooling = pooling
        self.out_channel = out_channel
        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        # x 视频编码信息 img (bs, conversation_len, grid_num, dim) x_mask (bs,conversation_len, 1, max_len, grid_num) tau 10
        # 其余内容和 Soft的一致
        logits = []
        if self.pooling == 'avg':
            #   四维张量的处理，需要单独使用后三维度  直接处理长度
            output = torch.zeros([x.shape[0], x.shape[1], x.shape[3], 1])  # [(bs, conversation_len, dim, 1)]
            for i in range(x.shape[0]):
                tmp = x[i]  # (conversation_len, grid_num, dim)
                tmp = tmp.transpose(1, 2)
                tmp = self.pool(tmp)  # ( conversation_len, dim, 1)
                output[i] = tmp
            # x = x.transpose(2, 3)  # (bs, conversation_len, dim, max_len)
            # x = self.pool(x)
            logits = self.mlp(output.squeeze(-1))
            # (bs, conversation_len, dim, 1) -> # (bs, conversation_len, dim) -> # (bs, conversation_len, output_dim)
        elif self.pooling == 'fc':
            #  处理维度
            output = torch.zeros([x.shape[0], x.shape[1], x.shape[2], self.out_channel])
            for i in range(x.shape[0]):
                tmp = x[i]
                b, _, c = tmp.size()
                mask = self.make_mask(tmp).squeeze(1).squeeze(1).unsqueeze(2)  # (conversation_len, grid_num, 1)
                scores = self.pool(tmp)  # (conversation_len, grid_num, dim) -> (conversation_len, grid_num, 1)
                scores = scores.masked_fill(mask, -1e9)
                scores = F.softmax(scores, dim=1)
                _x = tmp.mul(scores)  # (conversation_len, grid_num, grid_num)
                tmp = torch.sum(_x, dim=1)  # (conversation_len, grid_num)
                logits = self.mlp(tmp)  # (conversation_len, out_channel)
                output[i] = logits
            logits = output
        alpha = self.gumbel_softmax(logits, -1, tau)  # 修改了最后的函数权重的计算部分 tau指示了本方法的温度参数 值越小分类结果更离散 越大则越接近softmax结果
        #  [bs, conversation_len, out_channel]
        return alpha

    def gumbel_softmax(self, logits, dim=-1, temperature=0.1):
        '''
        Use this to replace argmax
        My input is probability distribution, multiply by 10 to get a value like logits' outputs.
        '''
        gumbels = -torch.empty_like(logits).exponential_().log()
        logits = (logits.log_softmax(dim=dim) + gumbels) / temperature
        return F.softmax(logits, dim=dim)

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(2).unsqueeze(3)  # (conversation_len, grid_num, dim) -> (conversation_len, 1, 1, grid_num)


class mean_Block(nn.Module):
    """
    Self-Attention Routing Block
    orders = 4 对应与使用的动态路由个数吧
    """

    def __init__(self, hidden_size, orders):
        super(mean_Block, self).__init__()
        self.len = orders
        self.hidden_size = hidden_size

    def forward(self, x, tau, masks):
        alpha = (1 / self.len) * torch.ones(x.shape[0], x.shape[1], self.len).to(x.device)  # (bs, 4)
        return alpha


class SARoutingBlock(nn.Module):
    """
    Self-Attention Routing Block
    """

    def __init__(self, opt):
        super(SARoutingBlock, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_k = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_q = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_merge = nn.Linear(opt.hidden_size, opt.hidden_size)
        #  默认使用 avg 方式
        if opt.routing == 'hard':
            self.routing_block = HardRoutingBlock(opt.hidden_size, opt.orders, opt.pooling)
        elif opt.routing == 'soft':
            self.routing_block = SoftRoutingBlock(opt.hidden_size, opt.orders, opt.pooling)
        elif opt.routing == 'mean':
            self.routing_block = mean_Block(opt.hidden_size, opt.pooling)

        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, v, k, q, masks, tau, training):
        # mask 是视频模态使用的掩码的列表 是 [bs, conversation_len, 1, max_len, grid_num] 的一个列表
        # v,k 是视频模态，q是文本模态 需要处理两者模态长度不一致的问题 ，考虑到原始模型 给出了视频掩码适配文本掩码的方法，故采用同样的方式
        # 将整体的长度拉长到max_len 即 100
        n_batches = q.size(0)
        conversation_len = q.size(1)  # MECPE数据集多出来的一维
        x = v

        alphas = self.routing_block(x, tau, masks)
        # (bs, conversation_len, 4) 对应于论文中的路由权重部分， 将图片的经过ViT后得到的维度信息进行 其个数与DynRT的层数一致
        # 其输出维度依照不同层的情况 分别是 1 2 3 4

        if self.opt.BINARIZE:
            if not training:
                alphas = self.argmax_binarize(alphas)

        v = self.linear_v(v).view(
            n_batches,
            conversation_len,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(2, 3)  # (bs, conversation_len, 4, 49, 192)

        k = self.linear_k(k).view(
            n_batches,
            conversation_len,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(2, 3)  # (bs, conversation_len, 4, 49, 192)

        q = self.linear_q(q).view(
            n_batches,
            conversation_len,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(2, 3)  # (bs, conversation_len, 4, 100, 192)

        att_list = self.routing_att(v, k, q, masks)  # (bs, order_num, head_num, grid_num, grid_num) (bs, 4, 4, 49, 49)
        # (bs, conversation_len, 4, 4, 100, 49)
        att_map = torch.einsum('bvl,bvlcnm->bvcnm', alphas, att_list)
        # 此处对应论文中的路由权重 * 模态交叉注意力计算结果 得到当前层的 权重大小
        # (bs, conversation_len, 4), (bs, conversation_len, 4, 4, 100, 49) - > (bs, conversation_len, 4, 100, 49)

        atted = torch.matmul(att_map, v)
        # (bs, conversation_len, 4, 100, 49) * (bs, conversation_len, 4, 49, 192) - > (bs, conversation_len, 4, 100, 192) mul [100, 49]*[49, 192],

        atted = atted.transpose(2, 3).contiguous().view(
            n_batches,
            conversation_len,
            -1,
            self.opt.hidden_size
        )  # (bs, conversation_len, 100, 768)  多头注意力再次回来

        atted = self.linear_merge(atted)  # (bs, conversation_len, 100, 768)

        return atted

    def routing_att(self, value, key, query, masks):
        """
        计算qkv的值，并乘上图片mask所表示的可见范围
        """
        d_k = query.size(-1)  # masks [[bs, 1, 1, 49], [bs, 1, 49, 49], [bs, 1, 49, 49], [bs, 1, 49, 49]]
        scores = torch.matmul(
            query, key.transpose(-2, -1)
            # query(bs, conversation_len, 4, 100, 192) , key.transpose(-2, -1)  (bs, conversation_len, 4, 192, 49)
        ) / math.sqrt(d_k)  # (bs, 4, 49, 49) (2, 4, 360, 49)
        # k q v [4, 4, 49, 192] key (2, 4, 49, 192) query [2, 4, 360, 192]
        # masks是get_image_Masks得到的 保证了前三维不变的情况下， [max_len, grid_num]
        # Masks 只是指示了每个token可以看到的图片位置， 这样的话，文本模态的填充值有影响吗？那为什么文本长度在源代码里面需要大于grid_num呢
        # print("scores", scores)
        att_list = []
        for i in range(len(masks)):
            mask = masks[i]  # (bs, conversation_len, 1, 100, 49)
            scores_temp = scores.masked_fill(mask, -1e9)
            att_map = F.softmax(scores_temp, dim=-1)
            att_map = self.dropout(att_map)
            # print("att_map", att_map)
            if i == 0:
                att_list = att_map.unsqueeze(2)  # (bs, conversation_len, 1, 4, 100, 49)
            else:
                att_list = torch.cat((att_list, att_map.unsqueeze(2)),
                                     2)  # (bs, conversation_len, 2, 4, 100, 49) -> (bs, conversation_len, 3, 4, 100, 49)
            #  个数同样与layers 即 当前层数挂钩 可以将其视为论文中的qkv计算结果
        return att_list

    def argmax_binarize(self, alphas):
        # n = alphas.size()[0]
        # out = torch.zeros_like(alphas)
        # indexes = alphas.argmax(-1)
        # out[torch.arange(n), indexes] = 1
        n, m, _ = alphas.size()
        out = torch.zeros_like(alphas)
        indexes = alphas.argmax(-1)
        # 将 torch.arange(n) 扩展为形状 [n, m] 以匹配 indexes 的形状
        out[torch.arange(n).view(n, 1).expand(n, m), indexes] = 1
        return out


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, opt):
        super(FFN, self).__init__()

        self.mlp = MLP(
            input_dim=opt.hidden_size,
            hidden_dim=opt.ffn_size,
            output_dim=opt.hidden_size,
            dropout=opt.dropout,
            activation="ReLU"
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, opt):
        super(MHAtt, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_k = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_q = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_merge = nn.Linear(opt.hidden_size, opt.hidden_size)

        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, v, k, q, mask):
        # mask 是视频模态使用的掩码的列表 是 [bs, conversation_len, 1, 1, max_len] 的一个列表
        # 这里按照论文里说的话，询问永远是文本为主， 所以此处的自注意力机制只在整合后的文本模态信息上执行
        # v,k 是视频模态，q是文本模态 需要处理两者模态长度不一致的问题 ，考虑到原始模型 给出了视频掩码适配文本掩码的方法，故采用同样的方式
        # 将整体的长度拉长到max_len 即 100
        n_batches = q.size(0)
        conversation_len = q.size(1)  # MECPE数据集多出来的一维

        #  路由权重只在交叉注意力中使用
        v = self.linear_v(v).view(
            n_batches,
            conversation_len,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(2, 3)  # (bs, conversation_len, 4, 49/100, 192)

        k = self.linear_k(k).view(
            n_batches,
            conversation_len,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(2, 3)  # (bs, conversation_len, 4, 49/100, 192)

        q = self.linear_q(q).view(
            n_batches,
            conversation_len,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(2, 3)  # (bs, conversation_len, 4, 49/100, 192)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(2, 3).contiguous().view(
            n_batches,
            conversation_len,
            -1,
            self.opt.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        # (bs, conversation_len, 4, 49/100, 192)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        # (bs, conversation_len, 4, 49/100, 49/100)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # (bs, conversation_len, 4, 49/100, 49/100)
            # 文本模态的mask 是正常的 1代表有，0表示填充
        att_map = F.softmax(scores, dim=-1)  # 计算整个序列上 与 每个token之间的相似度
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
        # (bs, conversation_len, 4, 49/100, 192)


class MHAttAudio(nn.Module):
    """
    先暂定 文本与音频的交叉注意力机制 就直接计算 就好， 暂时还没有什么眼前一亮的方法 等整体训练完成之后再看看
    """
    def __init__(self, opt):
        super(MHAttAudio, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_k = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_q = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_merge = nn.Linear(opt.hidden_size, opt.hidden_size)

        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, v, k, q, mask):
        # mask 是视频模态使用的掩码的列表 是 [bs, conversation_len, 1, 1, max_len] 的一个列表
        # 这里按照论文里说的话，询问永远是文本为主， 所以此处的自注意力机制只在整合后的文本模态信息上执行
        # v,k 是音频模态，q是文本模态 需要处理两者模态长度不一致的问题 ，考虑到原始模型 给出了视频掩码适配文本掩码的方法，故采用同样的方式
        # 将整体的长度拉长到max_len 即 100 这里的音频使用opensmile对整体编码 所以就只能假定长度为1了
        n_batches = q.size(0)
        conversation_len = q.size(1)  # MECPE数据集多出来的一维

        #  路由权重只在交叉注意力中使用
        v = self.linear_v(v).view(
            n_batches,
            conversation_len,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(2, 3)  # (bs, conversation_len, 4, 49/100, 192)

        k = self.linear_k(k).view(
            n_batches,
            conversation_len,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(2, 3)  # (bs, conversation_len, 4, 49/100, 192)

        q = self.linear_q(q).view(
            n_batches,
            conversation_len,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(2, 3)  # (bs, conversation_len, 4, 49/100, 192)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(2, 3).contiguous().view(
            n_batches,
            conversation_len,
            -1,
            self.opt.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted  # (bs, conversation_len, 100, 768)

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        # (bs, conversation_len, 4, 49/100, 192)
        """
        (bs, conversation_len, 4, 100, 192) (bs, conversation_len, 4, 1, 192)
        (bs, conversation_len, 4, 100, 1)
        (bs, conversation_len, 4, 1, 192)
        """
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        print("mask", mask.size())
        # 因为音频模态长度为1, 所以直接在文本的mask上添加一维即可 正好对应
        # mask = mask.transpose(-1, -2) 更新了， 不用在这里转了
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # (bs, conversation_len, 4, 49/100, 49/100)
            # 文本模态的mask 是正常的 1代表有，0表示填充
        att_map = F.softmax(scores, dim=-1)  # 计算整个序列上 与 每个token之间的相似度
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
        # (bs, conversation_len, 4, 100, 192)


class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        if is_Fusion:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        else:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class multiTRAR_SA_block(nn.Module):
    """
    也许可以尝试 扩展同样的方法到音频模态上， 需要注意， 这里的最后输出是经过了注意力调整的 文本模态信息， 视频模态信息仍原路返回
    """

    def __init__(self, opt):
        super(multiTRAR_SA_block, self).__init__()
        """
        对应于论文的DynRT 部分 包括 1、多头交叉注意力， 2、多头自注意力 3、线性层， 每层都包含残差和归一化操作。
        """
        self.mhatt1 = SARoutingBlock(opt)
        self.mhatt2 = MHAtt(opt)
        self.ffn = FFN(opt)
        self.mha_audio = MHAttAudio(opt)
        self.opt = opt
        self.dropout1 = nn.Dropout(opt.dropout)
        self.norm1 = LayerNorm(opt.hidden_size)

        self.dropout2 = nn.Dropout(opt.dropout)
        self.norm2 = LayerNorm(opt.hidden_size)

        self.dropout3 = nn.Dropout(opt.dropout)
        self.norm3 = LayerNorm(opt.hidden_size)

        self.dropout4 = nn.Dropout(opt.dropout)
        self.norm4 = LayerNorm(opt.hidden_size)

        self.a = 0.7
        self.b = 0.3

        # modality-signal prompts m表示模态存在 nm表示模态缺失 为了配合一维卷积的输入格式， 先维度再长度
        self.promptl_m = nn.Parameter(torch.zeros(self.opt.prompt_dim, self.opt.len))
        self.prompta_m = nn.Parameter(torch.zeros(self.opt.prompt_dim, self.opt.a_len))
        self.promptv_m = nn.Parameter(torch.zeros(self.opt.prompt_dim, self.opt.v_len))
        self.promptl_nm = nn.Parameter(torch.zeros(self.opt.prompt_dim, self.opt.len))
        self.prompta_nm = nn.Parameter(torch.zeros(self.opt.prompt_dim, self.opt.a_len))
        self.promptv_nm = nn.Parameter(torch.zeros(self.opt.prompt_dim, self.opt.v_len))

        generative_prompt = torch.zeros(3, self.opt.prompt_dim, self.opt.prompt_length)
        self.generative_prompt = nn.Parameter(generative_prompt)
        #  整个部分表示了多模态生成的部分， 第一组是用单个模态恢复缺失模态的prompt，第二组是修改长度 保持输入长度一致
        #  通过一维卷积处理存在的模态 来为缺失模态提供提示信息 这里和原论文有出入，我们先处理好了维度统一的问题 所以这里都是隐藏层维度大小了
        self.l2a = MLPLayer(self.opt.hidden_size, self.opt.prompt_dim)
        self.l2v = MLPLayer(self.opt.hidden_size, self.opt.prompt_dim)
        self.v2a = MLPLayer(self.opt.hidden_size, self.opt.prompt_dim)
        self.v2l = MLPLayer(self.opt.hidden_size, self.opt.prompt_dim)
        self.a2v = MLPLayer(self.opt.hidden_size, self.opt.prompt_dim)
        self.a2l = MLPLayer(self.opt.hidden_size, self.opt.prompt_dim)
        # 所有缺失情况和填补情况 前面的是缺失的， 后面是补充得到的模态
        self.l_ap = MLPLayer(self.opt.prompt_length + self.opt.a_len, self.opt.len, True)
        self.l_vp = MLPLayer(self.opt.prompt_length + self.opt.v_len, self.opt.len, True)
        self.l_avp = MLPLayer(
            self.opt.prompt_length + self.opt.a_len + self.opt.v_len, self.opt.len, True
        )

        self.a_lp = MLPLayer(self.opt.prompt_length + self.opt.len, self.opt.a_len, True)
        self.a_vp = MLPLayer(self.opt.prompt_length + self.opt.v_len, self.opt.a_len, True)
        self.a_lvp = MLPLayer(
            self.opt.prompt_length + self.opt.len + self.opt.v_len, self.opt.a_len, True
        )

        self.v_ap = MLPLayer(self.opt.prompt_length + self.opt.a_len, self.opt.v_len, True)
        self.v_lp = MLPLayer(self.opt.prompt_length + self.opt.len, self.opt.v_len, True)
        self.v_alp = MLPLayer(
            self.opt.prompt_length + self.opt.a_len + self.opt.len, self.opt.v_len, True
        )

        # missing-type prompts 同样的，因为我们只使用 文本作为q矩阵，所以只有单一的有用，其余部分等着之后增加再说。
        self.missing_type_prompt = nn.Parameter(
            torch.zeros(3, self.opt.prompt_length, self.opt.prompt_dim)
        )
        # 最后的缺失missing-type的prompt的结果 会加入到模型的单个模态的自注意力部分，因为会融合其他两个模态 进行交叉注意力机制，
        #  所以维度是 2 * prompt_dim  保证投影矩阵的维度符合要求 比如原始论文采用拼接最后一维的方法 所以是2 * dim
        # 而我们采用了权值相加， 所以采用dim  注意保持和你所需要的维度相同
        # self.m_a = nn.Parameter(torch.zeros(self.opt.a_len, 2 * self.prompt_dim))
        # self.m_v = nn.Parameter(torch.zeros(self.opt.v_len, 2 * self.prompt_dim))
        # self.m_l = nn.Parameter(torch.zeros(self.opt.len, 2 * self.prompt_dim))

        self.m_a = nn.Parameter(torch.zeros(self.opt.a_len, self.opt.prompt_dim))
        self.m_v = nn.Parameter(torch.zeros(self.opt.v_len, self.opt.prompt_dim))
        self.m_l = nn.Parameter(torch.zeros(self.opt.len, self.opt.prompt_dim))

    def get_complete_data(self, x_l, x_a, x_v, missing_mode):
        """
        遍历每一个batch中的数据，默认全部模态都存在， 但是假设模态缺失的情况就是不使用这个模态 然后用单个模态的缺失填充值生成缺失模态
        然后修改填补的模态的长度变成与正常使用的长度一致， 最后为每个模态附加上missing-signal的prompt信息
        需要注意，所有维度修改都是用一维卷积操作得到的， 而操作的维度需要特定在 -2 维度上， 所以这里所有的输入都要在一开始交换维度后在进行输入
        全部操作完成后再重新交换回去
        因为我们一开始操作的时候已经执行了维度变换， 这里去除了此操作 只保留了对于生成特征的操作
        0 -> 文本补充
        1 -> 音频补充
        2 -> 视频补充
        3 -> 文本 音频 补充
        4 -> 文本 视频 补充
        5 -> 音频 视频 补充
        6 -> 全部存在
        """
        x_l, x_a, x_v = x_l.unsqueeze(dim=0), x_a.unsqueeze(dim=0), x_v.unsqueeze(dim=0)
        # print("x_l, x_a, x_v", x_l.size(), x_a.size(), x_v.size())
        if missing_mode == 0:
            # 获取文本模态， 音频和 视频补充
            x_l = torch.cat(
                [self.generative_prompt[0, :, :], self.a2l(x_a)[0], self.v2l(x_v)[0]],
                dim=1,
            ).unsqueeze(dim=0)
            #  调整维度，和MuLT的输入保持一致  并且在最后加上指示模态是否缺失的missing-signal的prompt
            x_l = self.l_avp(x_l.transpose(1, 2)).transpose(1, 2) + self.promptl_m
            x_a = x_a + self.prompta_nm
            x_v = x_v + self.promptv_nm
        elif missing_mode == 1:
            # 获取音频模态，文本和 视频补充
            x_a = torch.cat(
                [self.generative_prompt[1, :, :], self.l2a(x_l)[0], self.v2a(x_v)[0]],
                dim=1,
            ).unsqueeze(dim=0)
            x_a = self.a_lvp(x_a.transpose(1, 2)).transpose(1, 2) + self.prompta_m
            x_v = x_v + self.promptv_nm
            x_l = x_l + self.promptl_nm
        elif missing_mode == 2:
            # 获取视频模态， 音频和文本补充
            x_v = torch.cat(
                [self.generative_prompt[2, :, :], self.l2v(x_l)[0], self.a2v(x_a)[0]],
                dim=1,
            ).unsqueeze(dim=0)
            x_v = self.v_alp(x_v.transpose(1, 2)).transpose(1, 2) + self.promptv_m
            x_l = x_l + self.promptl_nm
            x_a = x_a + self.prompta_nm
        elif missing_mode == 3:
            # 获取文本， 音频模态， 视频补充
            x_l = torch.cat(
                [self.generative_prompt[0, :, :], self.v2l(x_v)[0]], dim=1
            ).unsqueeze(dim=0)
            x_a = torch.cat(
                [self.generative_prompt[1, :, :], self.v2a(x_v)[0]], dim=1
            ).unsqueeze(dim=0)
            x_l = self.l_vp(x_l.transpose(1, 2)).transpose(1, 2) + self.promptl_m
            x_a = self.a_vp(x_a.transpose(1, 2)).transpose(1, 2) + self.prompta_m
            x_v = x_v + self.promptv_nm
        elif missing_mode == 4:
            # 获取文本， 视频模态， 音频补充
            x_l = torch.cat(
                [self.generative_prompt[0, :, :], self.a2l(x_a)[0]], dim=1
            ).unsqueeze(dim=0)
            x_v = torch.cat(
                [self.generative_prompt[2, :, :], self.a2v(x_a)[0]], dim=1
            ).unsqueeze(dim=0)
            x_l = self.l_ap(x_l.transpose(1, 2)).transpose(1, 2) + self.promptl_m
            x_v = self.v_ap(x_v.transpose(1, 2)).transpose(1, 2) + self.promptv_m
            x_a = x_a + self.prompta_nm
        elif missing_mode == 5:
            # 获取视频， 音频模态， 文本补充
            x_a = torch.cat(
                [self.generative_prompt[1, :, :], self.l2a(x_l)[0]], dim=1
            ).unsqueeze(dim=0)
            x_v = torch.cat(
                [self.generative_prompt[2, :, :], self.l2v(x_l)[0]], dim=1
            ).unsqueeze(dim=0)
            x_a = self.a_lp(x_a.transpose(1, 2)).transpose(1, 2) + self.prompta_m
            x_v = self.v_lp(x_v.transpose(1, 2)).transpose(1, 2) + self.promptv_m
            x_l = x_l + self.promptl_nm
        else:
            # 全部都存在， 不做生成操作 ， 只执行missing-signal的指示操作
            x_a = x_a + self.prompta_nm
            x_l = x_l + self.promptl_nm
            x_v = x_v + self.promptv_nm

        return x_l, x_a, x_v

    def get_proj_matrix(self):
        """
        构建投影矩阵， 对应于论文的用 missing-signal构建出的 missing-type 的矩阵， 通过对三个模态的缺失情况*对应mask矩阵
        [1, self.prompt_dim, self.prompt_dim]
        -> [7, self.prompt_dim, self.prompt_dim] 相乘得到的结果 对应于 missing-type的prompt的修正项
        """
        a_v_l = (
            self.prompta_nm @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(dim=0)
        am_v_l = (
            self.prompta_m @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(dim=0)
        a_vm_l = (
            self.prompta_nm @ self.m_a
            + self.promptv_m @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(dim=0)
        a_v_lm = (
            self.prompta_nm @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_m @ self.m_l
        ).unsqueeze(dim=0)
        am_vm_l = (
            self.prompta_m @ self.m_a
            + self.promptv_m @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(dim=0)
        am_v_lm = (
            self.prompta_m @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_m @ self.m_l
        ).unsqueeze(dim=0)
        a_vm_lm = (
            self.prompta_nm @ self.m_a
            + self.promptv_m @ self.m_v
            + self.promptl_m @ self.m_l
        ).unsqueeze(dim=0)
        self.mp = torch.cat(
            [a_v_lm, am_v_l, a_vm_l, am_v_lm, a_vm_lm, am_vm_l, a_v_l], dim=0
        )

    def forward(self, x, y, z, y_masks, x_mask, tau, training, missing_mod, x_v_mask, attention_mask, audio_mask):
        """
        attention_mask用于指示后续prompt添加到文本模态后面的情况
        y_mask 是视频模态使用的掩码，  x_mask 是文本的指示掩码
        y 是视频模态，x是文本模态 z是音频模态
        此处的xyz 分别表示了文本，视频和音频模态的特征向量
        lang_feat = [bs, max_conversation_len, max_len+prompt_length, 768] 填充项为0
        audio_feat = [bs, max_conversation_len, 1, 6373]
        img_feat = [bs, max_conversation_len, 49, 768]
        missing_mod = [bs, max_conversation_len] 指示模态缺失情况
        audio_mask 指示Hubert后出现的长度维度
        """
        tmp_x = x
        tmp_y = y
        tmp_z = z
        tmp_x = tmp_x.transpose(-1, -2)
        tmp_y = tmp_y.transpose(-1, -2)
        tmp_z = tmp_z.transpose(-1, -2)
        xx_l = torch.zeros_like(x).to(device)
        xx_a = torch.zeros_like(z).to(device)
        xx_v = torch.zeros_like(y).to(device)
        for i, id in enumerate(x_v_mask):
            for j, ind in enumerate(id):
                ind = ind.item()
                if int(ind) != 0:
                    temp_x = tmp_x[i, j]
                    temp_y = tmp_y[i, j]
                    temp_z = tmp_z[i, j]
                    temp_missing_mod = missing_mod[i, j]
                    if temp_missing_mod.item() != 7:
                        x_l_temp, x_a_temp, x_v_temp = self.get_complete_data(
                            temp_x, temp_z, temp_y, temp_missing_mod
                        )
                    else:
                        x_l_temp = temp_x
                        x_a_temp = temp_z
                        x_v_temp = temp_y
                    x_l_temp = x_l_temp.transpose(-1, -2)
                    x_a_temp = x_a_temp.transpose(-1, -2)
                    x_v_temp = x_v_temp.transpose(-1, -2)
                    xx_l[i, j, :] = x_l_temp
                    xx_a[i, j, :] = x_a_temp
                    xx_v[i, j, :] = x_v_temp

        video_x = self.norm1(xx_l + self.dropout1(
            self.mhatt1(v=xx_v, k=xx_v, q=xx_l, masks=y_masks, tau=tau, training=training)
        ))  # (64, 49, 512) # (bs, 49, 768) 文本模态先经过与视频模态的交叉注意力融合
        # 多添加一个音频的注意力机制计算
        tmp_x_mask = x_mask.transpose(-1, -2)
        audio_mask1 = tmp_x_mask * audio_mask
        audio_x = self.norm4(xx_l + self.dropout4(self.mha_audio(v=xx_a, k=xx_a, q=xx_l, mask=audio_mask1)))
        # 结合两者内容后 用权重整合吧 暂定七三开
        x = self.a * video_x + self.b * audio_x

        self.get_proj_matrix()
        # 对于每一句话都有一个模态补充信息，并且获取他的用于自注意力计算时的 missing-type的prompt信息
        # 但是我们的方法都是给予文本开始计算的， 所以其实并没有用上原论文的方法， 现在只是保留这个内容存在
        # batch_prompt = torch.zeros(x.shape[0], x.shape[1], 3, self.opt.prompt_length, self.opt.prompt_dim)
        # 修改x中的内容，将prompt信息填入其中
        for i, id in enumerate(x_v_mask):
            tmp_att = attention_mask[i]
            for j, ind in enumerate(id):
                ind = ind.item()
                if int(ind) != 0 and missing_mod[i, j].item() != 7:
                    # print("missing_model", missing_mod[i, j].item())
                    # 每一个附加信息的维度 [prompt_length, prompt_dim]  因为暂定只有文本做自注意力, 所以只有一个需要计算
                    tmp = torch.matmul(self.missing_type_prompt[0], self.mp[missing_mod[i, j].item()])
                    num_tmp = torch.sum(tmp_att[j])
                    x[i, j, num_tmp:num_tmp+self.opt.prompt_length, :] = tmp
                    x_mask[i, j, num_tmp:num_tmp+self.opt.prompt_length] = 1
                    # batch_prompt[i, j] = tmp
        # 再走一个自注意力机制结束  在这里附加上missing-type的prompt的信息 100 mask  + 16  111 000  1111
        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# --------------------------------
# ---- img Local Window Generator ----
# --------------------------------
def getImgMasks(scale=16, order=2):
    """
    :param scale: Feature Map Scale
    :param order: Local Window Size, e.g., order=2 equals to windows size (5, 5)
    :return: masks = (scale**2, scale**2)
    """
    masks = []
    _scale = scale
    assert order < _scale, 'order size be smaller than feature map scale'

    for i in range(_scale):
        for j in range(_scale):
            mask = np.ones([_scale, _scale], dtype=np.float32)
            for x in range(i - order, i + order + 1, 1):
                for y in range(j - order, j + order + 1, 1):
                    if (0 <= x < _scale) and (0 <= y < _scale):
                        mask[x][y] = 0
            mask = np.reshape(mask, [_scale * _scale])
            masks.append(mask)
    masks = np.array(masks)
    masks = np.asarray(masks, dtype=np.bool)  # 0, 1 -> False True (True mask)  和论文的图示反过来， 0表示掩码关注的部分 反而是不关注的部分
    return masks


def getMasks_img_multimodal(x_mask, hyp_params):
    mask_list = []  # x_mask [bs, conversation_len, 1, 1, 49] len 100 Img Scale 7 保持源论文的参数信息
    # 如果采用复杂度较高的ViT模型， 会出现grid_num 大于 文本长度的情况，
    # 指示了每一个token对应的 各级的可以进行交叉注意力计算的 图片位置
    ORDERS = hyp_params.ORDERS
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)  # 不做任何操作的话，全部都是0
        else:  # 定义不同步长的掩码矩阵 因为 ViT的基本方法就是7*7 所以可选的掩码矩阵大小就是0 - 3 之间的值
            mask_img = torch.from_numpy(getImgMasks(hyp_params.IMG_SCALE, order)).byte().to(
                x_mask.device)  # (49, 49)  以每个可选点作为中心点的 图片掩码矩阵
            mask = torch.concat([mask_img] * (hyp_params.len // (hyp_params.IMG_SCALE * hyp_params.IMG_SCALE)), dim=0)
            mask = torch.concat([mask, mask_img[:(hyp_params.len % (hyp_params.IMG_SCALE * hyp_params.IMG_SCALE)), :]])
            mask = torch.logical_or(x_mask, mask)
        # (64, conversation_len, 1, max_len, grid_num)  将 (49, 49) 的第一维扩展到 len的大小 总之指示来 文本和视频模态各自真实值
        #     print("扩展后的mask维度", mask)
            mask_list.append(mask)
    return mask_list


class DynRT_ED(nn.Module):
    def __init__(self, hyp_params):
        super(DynRT_ED, self).__init__()
        self.hyp_params = hyp_params
        self.tau = hyp_params.tau_max
        opt_list = []
        for i in range(hyp_params.layer):
            hyp_params_copy = copy.deepcopy(hyp_params)
            hyp_params_copy.ORDERS = hyp_params.ORDERS[:len(hyp_params.ORDERS) - i]
            hyp_params_copy.orders = len(hyp_params.ORDERS) - i
            opt_list.append(copy.deepcopy(hyp_params_copy))
        self.dec_list = nn.ModuleList([multiTRAR_SA_block(opt_list[-(i + 1)]) for i in range(hyp_params.layer)])

    def forward(self, lang_feat, img_feat, audio_feat, conversation_mask, img_feat_mask, missing_mod, x_v_mask,
                attention_mask, audio_mask):
        """
        attention_mask用于指示后续prompt添加到文本模态后面的情况
        missing_mod = [bs, max_conversation_len] 指示模态缺失情况
        img_feat = [bs, max_conversation_len, 49, 768]
        conversation_mask 只用于指示batch对话中，每组对话的个数的填充项 [bs, conversation_len]
        如果保持一致， 需要提供的是 [bs, conversation_len, max_len]
        lang_feat = [bs, max_conversation_len, max_len, 768] 填充项为0
        audio_feat = [bs, max_conversation_len, 1, 6373]
        img_feat_mask (bs, conversation_len, 1, 1, grid_num)
        audio_mask 是使用Hubert后 出现了长度维度，进而给出了另一个mask

        模态如果缺失了 视频模态， 好像要单独做一个 只有音频和文本的一种方法才行， 还是继续保持三个都做为输入，再接受的时候进行判断 调用不同函数吧
        比如img_feat_mask 为空 说明输入的模态没有视频模态，相关操作全部更换
        如果使用RECCON也保持同样的输入信息吧， 差异化体现在模态特征是否为空， conversation_mask 帮助指示每组的真实信息位置
        RECCON只有文本模态 所以只能定义missing-mod的值为 5
        ECF数据集包含全部模态，可以在所有 缺失情况下进行训练， 保留模型用于验证

        此外，文本模态的信息还关系着指示每一组内容的责任， 所以在模态缺失中的 文本模态缺失部分， 我们只是把文本模态特征全部置0而已。
        所以实际操作就是， 假设所有模态都存在的前提下， 根据missing_mod的不同，将相应模态特征值置0

        对应音频与 其他模态的注意力计算，仍然使用基本的 交叉注意力得到 就值保留 文本 和 音频的融合
        而在 原因识别的时候，除了每个句子自身的 句子表示计算， 如何融合呢？
        """
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        img_feat_masks = getMasks_img_multimodal(img_feat_mask, self.hyp_params)  # 视频模态的大小没有变，到是把掩码矩阵的大小与文本长度保持了一致、
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors 不同轮次可以使用的掩码矩阵大小不同
        for i, dec in enumerate(self.dec_list):
            lang_feat = dec(lang_feat, img_feat, audio_feat, img_feat_masks[:i + 1], conversation_mask, self.tau,
                            self.training, missing_mod, x_v_mask, attention_mask, audio_mask)  # (4, 360, 768)
        return lang_feat, img_feat, audio_feat

    def set_tau(self, tau):
        self.tau = tau
