# -*- encoding: utf-8 -*-
# @File         : model.py
# @Date         : 2024/10/02 17:12:41
# @Author       : Eliwii_Keeya
# @Modified from: yuunnn-w, et al., 2024 -- RWKV_Pytorch

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from typing import Tuple


class RWKV_Block(nn.Cell):
    """
    RWKV模型的块结构。

    Args:
        block_w (dict): 权重字典。
        n_embd (int): 嵌入维度。
        n_head (int): 头数。
        i (int): 时间索引。
    """
    def __init__(self, block_w: dict, n_embd: int, n_head: int, i: int):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        # 初始化时间索引
        self.i0 = (2 + self.head_size) * i + 0
        self.i1 = (2 + self.head_size) * i + 1
        self.i2 = (2 + self.head_size) * i + 2
        self.i3 = (2 + self.head_size) * (i + 1)
        
        # 初始化层归一化
        self.ln1 = nn.LayerNorm((n_embd,),
                                gamma_init=mindspore.Parameter(block_w['ln1.weight']),
                                beta_init=mindspore.Parameter(block_w['ln1.bias']),
                                epsilon=1e-5)
        self.ln2 = nn.LayerNorm((n_embd,),
                                gamma_init=mindspore.Parameter(block_w['ln2.weight']),
                                beta_init=mindspore.Parameter(block_w['ln2.bias']),
                                epsilon=1e-5)

        # 初始化激活函数
        self.silu = nn.SiLU()
        
        # 初始化注意力参数
        self.att_time_maa_x = mindspore.Parameter(block_w['att.time_maa_x'])
        self.att_time_maa_w = mindspore.Parameter(block_w['att.time_maa_w'])
        self.att_time_maa_k = mindspore.Parameter(block_w['att.time_maa_k'])
        self.att_time_maa_v = mindspore.Parameter(block_w['att.time_maa_v'])
        self.att_time_maa_r = mindspore.Parameter(block_w['att.time_maa_r'])
        self.att_time_maa_g = mindspore.Parameter(block_w['att.time_maa_g'])
        self.att_time_maa_w1 = mindspore.Parameter(block_w['att.time_maa_w1'])
        self.att_time_maa_w2 = mindspore.Parameter(block_w['att.time_maa_w2'])
        self.att_time_decay = mindspore.Parameter(block_w['att.time_decay'])
        self.att_time_decay_w1 = mindspore.Parameter(block_w['att.time_decay_w1'])
        self.att_time_decay_w2 = mindspore.Parameter(block_w['att.time_decay_w2'])
        self.att_time_faaaa = mindspore.Parameter(block_w['att.time_faaaa'])
        self.att_receptance = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.att_receptance.weight = mindspore.Parameter(block_w['att.receptance.weight'])
        self.att_key = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.att_key.weight = mindspore.Parameter(block_w['att.key.weight'])
        self.att_value = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.att_value.weight = mindspore.Parameter(block_w['att.value.weight'])
        self.att_output = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.att_output.weight = mindspore.Parameter(block_w['att.output.weight'])
        self.att_gate = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.att_gate.weight = mindspore.Parameter(block_w['att.gate.weight'])
        self.att_group_norm = nn.GroupNorm(num_groups=n_head,
                                            num_channels=n_embd,
                                            eps=1e-5,
                                            affine=True,
                                            gamma_init=mindspore.Parameter(block_w['att.ln_x.weight']),
                                            beta_init=mindspore.Parameter(block_w['att.ln_x.bias']))
            
        # 初始化前馈参数
        self.ffn_time_maa_k = mindspore.Parameter(block_w['ffn.time_maa_k'])
        self.ffn_time_maa_r = mindspore.Parameter(block_w['ffn.time_maa_r'])
        self.ffn_key = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.ffn_key.weight = mindspore.Parameter(block_w['ffn.key.weight'])
        self.ffn_receptance = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.ffn_receptance.weight = mindspore.Parameter(block_w['ffn.receptance.weight'])
        self.ffn_value = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.ffn_value.weight = mindspore.Parameter(block_w['ffn.value.weight'])

    def channel_mixing(self, x: mindspore.Tensor, state: mindspore.Tensor) -> mindspore.Tensor:
        """
        通道混合函数。

        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, 2048]。
            state (mindspore.Tensor): 时间状态张量，形状为[Batch, State Size, 2048]。

        Returns:
            mindspore.Tensor: 混合后的张量，形状与输入的x相同。
        """
        sx = state[:, self.i0] - x
        state[:, self.i0] = x
        xk = x + sx * self.ffn_time_maa_k
        xr = x + sx * self.ffn_time_maa_r
        r = ops.sigmoid(self.ffn_receptance(xr))
        k = ops.relu(self.ffn_key(xk)).pow(2)
        output = r * self.ffn_value(k)
        return output

    def time_mixing(self, x: mindspore.Tensor, state: mindspore.Tensor) -> mindspore.Tensor:
        """
        时间混合函数。

        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, 2048]。
            state (mindspore.Tensor): 时间状态张量，形状为[Batch, State Size, 2048]。
        Returns:
            mindspore.Tensor: 混合后的时间状态张量，形状与输入的state相同。
        """
        batch_size, H, S = x.shape[0], self.n_head, self.head_size

        sx = state[:, self.i1] - x
        state[:, self.i1] = x
        
        xxx = x + sx * self.att_time_maa_x
        xxx = ops.tanh(xxx @ self.att_time_maa_w1).view(batch_size, 5, 1, -1)
        xxx = ops.matmul(xxx, self.att_time_maa_w2).view(batch_size, 5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=1)

        xw = x + sx * (self.att_time_maa_w + mw)
        xk = x + sx * (self.att_time_maa_k + mk)
        xv = x + sx * (self.att_time_maa_v + mv)
        xr = x + sx * (self.att_time_maa_r + mr)
        xg = x + sx * (self.att_time_maa_g + mg)

        w = (self.att_time_decay + (ops.tanh(xw @ self.att_time_decay_w1) @ self.att_time_decay_w2))
        
        # 计算注意力机制的权重
        w = ops.exp(-ops.exp(w.view(batch_size, H, S, 1)))

        # 计算注意力机制的组件
        r = self.att_receptance(xr).view(batch_size, H, 1, S)
        k = self.att_key(xk).view(batch_size, H, S, 1)
        v = self.att_value(xv).view(batch_size, H, 1, S)
        g = self.silu(self.att_gate(xg))

        # 使用注意力机制更新状态
        s = state[:, self.i2: self.i3, :].view(batch_size, H, S, S)
        a = k @ v
        x = r @ (self.att_time_faaaa * a + s)
        s = a + w * s
        state[:, self.i2: self.i3, :] = s.view(batch_size, S, -1)

        # 展平x并应用组归一化和门控
        x = self.att_group_norm(x.flatten(start_dim=1)) * g

        # 应用输出层并返回结果
        return self.att_output(x)

    def construct(self, x: mindspore.Tensor, state: mindspore.Tensor) -> mindspore.Tensor:
        """
        模型的前向传播。
        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, N_embd]。
            state (mindspore.Tensor): 隐藏状态张量，形状为[Batch, State Size, N_embd]。
        Returns:
            mindspore.Tensor: 前向传播结果张量，形状与输入的x相同。
        """
        x = x + self.time_mixing(self.ln1(x), state)
        x = x + self.channel_mixing(self.ln2(x), state)
        return x
        

class RWKV_RNN(nn.Cell):
    """
    RWKV模型的RNN结构。

    Args:
        args (dict): 参数字典。
    """
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.set_train(False)

        # 加载权重
        w = mindspore.load_checkpoint(args['MODEL_NAME'] + '.ckpt')
        
        # 将所有权重转换为float32
        self.num_layer = 0
        for k in w.keys():
            w[k] = w[k].float()
            if '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)
            if "blocks" in k: self.num_layer = max(self.num_layer, int(k.split(".")[1]))
        
        self.num_layer += 1

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.n_embd = w['blocks.0.ln1.weight'].shape[0]
        self.head_size = self.n_embd // self.n_head
        self.state_size = [self.num_layer * (2 + self.head_size), self.n_embd]

        print(f"state_size: {self.state_size}") # 这里打印状态的形状
        
        # 初始化模型参数
        self.emb = nn.Embedding(w['emb.weight'].shape[0], w['emb.weight'].shape[1], embedding_table=w['emb.weight'])
        self.ln0 = nn.LayerNorm((self.n_embd,),
                                gamma_init=mindspore.Parameter(w['blocks.0.ln0.weight']),
                                beta_init=mindspore.Parameter(w['blocks.0.ln0.bias']),
                                epsilon=1e-5)
        self.blocks = nn.CellList()
        
        for i in range(self.num_layer):
            # 提取当前块的权重
            block_w = {k[len(f'blocks.{i}.'):]: v for k, v in w.items() if f'blocks.{i}.' in k}
            self.blocks.append(RWKV_Block(block_w, self.n_embd, self.n_head, i))
            print(f"Loading blocks...[{i + 1}/{self.num_layer}]", end='\r')
        print()

        self.ln_out = nn.LayerNorm((self.n_embd,),
                                    gamma_init=mindspore.Parameter(w['ln_out.weight']),
                                    beta_init=mindspore.Parameter(w['ln_out.bias']),
                                    epsilon=1e-5)
        self.head = nn.Dense(self.n_embd, args['vocab_size'], has_bias=False)
        self.head.weight = mindspore.Parameter(w['head.weight'])

    def construct(self, token: mindspore.Tensor, state: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        模型的前向传播。
        Args:
            token (mindspore.Tensor): 输入的令牌张量。[Batch_size]
            state (mindspore.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]
        Returns:
            mindspore.Tensor: 模型输出。
        """
        x = self.emb(token)
        x = self.ln0(x)
        for block in self.blocks:
            x = block(x, state)
        x = self.ln_out(x)
        x = self.head(x)
        return x, state
