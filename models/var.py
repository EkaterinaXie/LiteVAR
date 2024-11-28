import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE, # vae_local:VQVAE模型的实例,用于量化和解量化
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False, # depth:transformer块的数量16，embed_dim: 嵌入向量的维度1024，mlp_ratio: 前馈网络的中间维度与嵌入维度的比值4
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        # 推理部分，做十次forward来把Latent生成出来
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size # 从vae_local中获取码本大小(Cvae)和词典大小(V)
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads # 保存depth、嵌入维度(C)、前馈网络中间维度(D)和头数
        
        self.cond_drop_rate = cond_drop_rate # 条件嵌入的dropout率
        self.prog_si = -1   # progressive training 初始化prog_si为-1(用于渐进式训练)
        
        self.patch_nums: Tuple[int] = patch_nums # 之前设置的元组(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.L = sum(pn ** 2 for pn in self.patch_nums) # L：multi-scale所有token的数量
        self.first_l = self.patch_nums[0] ** 2 # 计算第一个尺度的token数first_l，即1x1
        self.begin_ends = [] # 初始化begin_ends列表,记录每个尺度的token索引范围，比如3x3的就是5-13
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        # breakpoint()
        self.rng = torch.Generator(device=dist.get_device()) # 初始化一个随机数生成器rng,用于推理过程中的随机采样
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize # 获取vae_local中的量化器quant
        self.vae_proxy: Tuple[VQVAE] = (vae_local,) # 将vae_local和quant保存为元组
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C) # 定义一个线性层word_embed,将VQ码本中的向量映射到嵌入空间
                                                     # VQ词典里面每个vector长度转化为image的embedding每个token长度？Cvae到底是码本大小还是向量本身？？
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3) # 嵌入空间的初始化标准差
        self.num_classes = num_classes
        # 创建一个均匀分布的概率向量uniform_prob,用于无条件采样
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        # class embedding: imagenet1k类，有1k个不同的emb作为guidance来提示生成什么类别的图像，1是无条件
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C) # 将类别映射到嵌入空间
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std) # 用截断正态分布初始化class emb的权重
        # 第一个token生成之前的占位符，是一个learnable的token（可学习的位置嵌入），表示第一个token的位置信息
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C)) # Init learnable PE
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std) # 用截断正态分布初始化pos_start
        
        # 3. absolute position embedding 绝对位置编码
        pos_1LC = []
        # 对于每个尺度,创建一个形状为(1, pn*pn, self.C)的张量,使用截断正态分布初始化
        # 将所有尺度的位置嵌入连接起来,形成(1, L, C)的张量
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C 这里这里
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC) # absolute learnable PE，pos_1LC是可学习的
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        '''下一句代码初始化了级别嵌入lvl_embed。它是一个嵌入层,将不同尺度(scale)的索引映射到嵌入空间。嵌入的大小为(len(self.patch_nums), self.C)。
        级别嵌入类似于GPT中的段嵌入,用于区分不同尺度的token金字塔。最后,使用截断正态分布初始化lvl_embed的权重。'''
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C) # Level PE 将scale的个数映射到嵌入空间????????????????????????
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        # 定义了一个shared_ada_lin,如果shared_aln为True,则它是一个包含SiLU激活函数和SharedAdaLin线性层的序列;否则,它是一个恒等映射。
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn( # AdaLNSelfAttn块,数量为depth
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks] # 是否使用了融合的加法和规范化函数fused_add_norm_fn
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1] 返回重建图像
        """
        only used for inference, on autoregressive mode
        :param B: batch size          整数，批次大小
        :param label_B: imagenet label; if None, randomly sampled 可选整数 
        :param g_seed: random seed    整数
        :param cfg: classifier-free guidance ratio CFG
        :param top_k: top-k sampling  用于Top-K采样的K值,默认为0(不使用Top-K采样)
        :param top_p: top-p sampling  用于Top-P(nucleus)采样的P值,默认为0.0(不使用Top-P采样)。
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
                                      一个布尔值,指示是否使用Gumbel Softmax平滑预测结果,默认为False。在可视化时使用,但在计算FID/IS时不使用。
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng # rng是一个随机数生成器,用于控制随机采样过程
        
        if label_B is None: # 如果没设ImageNet标签，则从均匀分布中随机采样B个标签
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int): # 如果设了标签整数，则创建一个形状为(B,)的张量,填充为该整数（这八个？）(如果大于0)或self.num_classes(如果小于0)。这个操作是为了处理标签的输入。demo中B=8
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        # 带condition的和不带condition的，batchsize直接x2，classification embedding和代表没有类的embedding
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        '''
        sos这一行代码计算了条件embedding (cond_BD)。
        首先,它将label_B和一个表示"没有类别"的填充张量(torch.full_like(label_B, fill_value=self.num_classes))连接起来,
        形成一个新的张量。然后,它将这个新张量传递给self.class_emb函数,得到条件embedding cond_BD。
        sos也被赋予相同的值。这里的cond_BD代表了带有标签和没有标签的条件embedding,批次大小被直接乘以2。
        '''
        # positional encoding,pos_1LC表示绝对的在sequance位置上的encoding。
        # 位置编码lvl_pos,表示每个patch在序列中的绝对位置编码和所在scale的编码？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        # 初始token，sos.是构造好的condition，pos_start是第一个learnable的头，加上第一个token的positional encoding，self.first_l=1
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        # 初始化零向量f_hat,用于存储每个scale估计的值
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]) # 每个scale估计的值堆起来
        
        for b in self.blocks: b.attn.kv_caching(True) # kv cache???
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn # 累加当前token的平方
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD) # 对条件embed线性变换
            x = next_token_map
            '''对每个块(self.blocks)中的AdaLNSelfAttn模块进行前向传播计算,输入是：x当前token映射、cond_BD_or_gss(线性变换后的条件嵌入)和None(注意力bias)。计算结果会覆盖x'''
            # for b in self.blocks: # 这里调用的似乎是AdaLN的自注意力forward，后面var的forward似乎没找到调用点？？？？？？？？？？？？？
            #     x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            for block_idx, b in enumerate(self.blocks):  # 引入block序号的参数block_idx
                # breakpoint()
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None, block_idx=block_idx)
            logits_BlV = self.get_logits(x, cond_BD) # 将处理后的x和原始条件嵌入cond_BD作为输入,得到logits张量logits_BlV。
            
            t = cfg * ratio # t是CFG的强度
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:] # 线性组合带标签的和不带标签的emb，算logits
            '''使用sample_with_top_k_top_p_函数从logits_BlV中采样,得到索引张量 idx_Bl
            采样过程受top_k和top_p超参数的影响,使用rng作为随机数生成器。'''
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case   # 默认直接使用索引idx_Bl查找embedding
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            # 查找得到的h_BChw嵌入，调整形状，与当前对于残差的估计f_hat合并，得到下一阶段的输入next_token_map
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            # 这里用了quant.py里面的def get_next_autoregressive_input，当前估计的残差缩放给到下一个scale的技巧
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            '''如果当前不是最后一个阶段,这部分代码会准备下一阶段的next_token_map。
            首先调整形状,然后通过self.word_embed进行词嵌入,再加上相应的位置编码lvl_pos。
            最后,由于CFG,会将next_token_map在第一维度上重复2次。
            '''
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False) # 禁用注意力机制中的键值缓存
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]将f_hat转换为图像张量
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size V是词汇表大小
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L) # 获取当前阶段的起始和结束位置(bg和ed)
        B = x_BLCv_wo_first_l.shape[0] # 从x_BLCv_wo_first_l的形状中获取B:批次大小
        '''
        这部分代码计算初始状态(sos)和条件embedding(cond_BD)。
        首先,根据self.cond_drop_rate的概率,将一部分标签替换为self.num_classes(表示无条件)。
        然后,使用self.class_emb将标签映射为条件embedding cond_BD。sos被赋予相同的值。
        接下来,sos在第二维度上被扩展为(B, self.first_l, -1)的形状,并加上self.pos_start(表示初始位置编码)。
        '''
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            '''
            这部分代码准备输入x_BLC。
            如果self.prog_si为0,则x_BLC直接等于sos。
            否则,x_BLC是将sos和通过self.word_embed映射后的x_BLCv_wo_first_l连接而成。
            最后,x_BLC会加上scale embedding(self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)))和位置编码(self.pos_1LC[:, :ed])。
            '''
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        '''
        这部分代码是一个hack,用于获取混合精度计算时的数据类型(main_type)。
        它通过创建一个临时张量,执行矩阵乘法,并获取结果的数据类型来实现。
        然后,将x_BLC、cond_BD_or_gss和attn_bias转换为该数据类型。
        '''
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        '''
        这部分代码对输入x_BLC进行处理。它遍历self.blocks中的每个块(b)。
        对于每个块,它将x_BLC、cond_BD_or_gss和attn_bias作为输入,执行前向传播计算,并将结果覆盖到x_BLC上。
        '''
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        '''
        这部分代码是一个特殊处理,当self.prog_si为0时执行。
        它修改x_BLC的第一个元素(x_BLC[0, 0, 0])。如果self.word_embed是一个线性层,则将self.word_embed的权重和偏置的第一个元素乘以0后加到x_BLC[0, 0, 0]上。
        否则,它会遍历self.word_embed
        '''
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
