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
    """
    def __init__(self, block_w: dict, n_embd: int, n_head: int, onnx_opset = 16):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.onnx_opset = onnx_opset
        
        # 初始化层归一化
        if self.onnx_opset >= 17:
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln1.weight = mindspore.Parameter(block_w['ln1.weight'])
            self.ln1.bias = mindspore.Parameter(block_w['ln1.bias'])
            self.ln2 = nn.LayerNorm(n_embd)
            self.ln2.weight = mindspore.Parameter(block_w['ln2.weight'])
            self.ln2.bias = mindspore.Parameter(block_w['ln2.bias'])
        else:
            self.ln1_weight = mindspore.Parameter(block_w['ln1.weight'])
            self.ln1_bias = mindspore.Parameter(block_w['ln1.bias'])
            self.ln2_weight = mindspore.Parameter(block_w['ln2.weight'])
            self.ln2_bias = mindspore.Parameter(block_w['ln2.bias'])


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
        
        if self.onnx_opset >= 18:
            self.att_group_norm = nn.GroupNorm(num_groups=n_head, num_channels=n_embd, eps=1e-5, affine=True)
            self.att_group_norm.weight = mindspore.Parameter(block_w['att.ln_x.weight'])
            self.att_group_norm.bias = mindspore.Parameter(block_w['att.ln_x.bias'])
        else:
            self.att_group_norm_weight = mindspore.Parameter(block_w['att.ln_x.weight'])
            self.att_group_norm_bias = mindspore.Parameter(block_w['att.ln_x.bias'])
            
        # 初始化前馈参数
        self.ffn_time_maa_k = mindspore.Parameter(block_w['ffn.time_maa_k'])
        self.ffn_time_maa_r = mindspore.Parameter(block_w['ffn.time_maa_r'])
        self.ffn_key = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.ffn_key.weight = mindspore.Parameter(block_w['ffn.key.weight'])
        self.ffn_receptance = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.ffn_receptance.weight = mindspore.Parameter(block_w['ffn.receptance.weight'])
        self.ffn_value = nn.Dense(self.n_embd, self.n_embd, has_bias=False)
        self.ffn_value.weight = mindspore.Parameter(block_w['ffn.value.weight'])

    def manual_layer_norm(self, x: mindspore.Tensor, weight: mindspore.Tensor, bias: mindspore.Tensor, eps: float = 1e-5) -> mindspore.Tensor:
        """
        人工层归一化函数
        Args:
            x (mindspore.Tensor): 输入张量，形状为 [Batch, *]，* 表示任意维度。
            weight (mindspore.Tensor): 归一化的权重张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            bias (mindspore.Tensor): 归一化的偏置张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            eps (float): 用于数值稳定性的小值，防止除以零。
        Returns:
            mindspore.Tensor: 经过手动层归一化后的张量，形状与输入的 x 相同。
        """
        mean = x.mean(axis=-1, keep_dims=True)
        var = x.var(axis=-1, keepdims=True, ddof=False)
        x_normalized = (x - mean) / ops.sqrt(var + eps)
        x_scaled = x_normalized * weight#.unsqueeze(-1) #这里会自动广播对齐
        x_shifted = x_scaled + bias#.unsqueeze(-1)
        return x_shifted
        
    def manual_group_norm(self, x: mindspore.Tensor, num_groups: int, weight: mindspore.Tensor, bias: mindspore.Tensor, eps: float = 1e-5) -> mindspore.Tensor:
        """
        人工组归一化函数。
        Args:
            x (mindspore.Tensor): 输入张量，形状为 [Batch, 2048]。（或者[Batch*L, 2048]）
            num_groups (int): 分组数，这里为 RWKV 的注意力头数。
            weight (mindspore.Tensor): 归一化的权重张量，形状为 [2048]。
            bias (mindspore.Tensor): 归一化的偏置张量，形状为 [2048]。
            eps (float): 用于数值稳定性的小值，防止除以零。
        Returns:
            mindspore.Tensor: 经过人工组归一化后的张量，形状与输入的 x 相同。
        """
        N, C = x.shape
        #if C % num_groups != 0:
            #raise ValueError("num_channels must be divisible by num_groups")
        #加上这个会有无法推断静态图的警告
        channels_per_group = C // num_groups
        # 重塑x以便于分组
        x = x.view(N, num_groups, channels_per_group)
        mean = x.mean(axis=2, keep_dims=True)
        var = x.var(axis=2, keepdims=True, ddof=False)
        x_normalized = (x - mean) / ops.sqrt(var + eps)
        x_normalized = x_normalized.view(N, C)
        # 应用权重和偏置
        x_scaled = x_normalized * weight
        x_shifted = x_scaled + bias
        return x_shifted

    def channel_mixing(self, x: mindspore.Tensor, state: mindspore.Tensor, i: int) -> mindspore.Tensor:
        """
        通道混合函数。

        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, 2048]。
            state (mindspore.Tensor): 时间状态张量，形状为[Batch, State Size, 2048]。
            i (int): 时间索引。

        Returns:
            mindspore.Tensor: 混合后的张量，形状与输入的x相同。
        """
        i0 = (2 + self.head_size) * i + 0
        sx = state[:, i0] - x
        state[:, i0] = x
        xk = x + sx * self.ffn_time_maa_k
        xr = x + sx * self.ffn_time_maa_r
        r = ops.sigmoid(self.ffn_receptance(xr))
        k = ops.relu(self.ffn_key(xk)).pow(2)
        output = r * self.ffn_value(k)
        return output

    def time_mixing(self, x: mindspore.Tensor, state: mindspore.Tensor, i: int) -> mindspore.Tensor:
        """
        时间混合函数。

        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, 2048]。
            state (mindspore.Tensor): 时间状态张量，形状为[Batch, State Size, 2048]。
            i (int): 时间索引。
        Returns:
            mindspore.Tensor: 混合后的时间状态张量，形状与输入的state相同。
        """
        batch_size, H, S = x.shape[0], self.n_head, self.head_size
        i1 = (2 + S) * i + 1

        sx = state[:, i1] - x
        state[:, i1] = x
        
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
        s = state[:, (2+S)*i+2:(2+S)*(i+1), :].view(batch_size, H, S, S)
        a = k @ v
        x = r @ (self.att_time_faaaa * a + s)
        s = a + w * s
        state[:, (2+S)*i+2:(2+S)*(i+1), :] = s.view(batch_size, S, -1)

        # 展平x并应用组归一化和门控
        if self.onnx_opset >= 18:
            x = self.att_group_norm(x.flatten(start_dim=1)) * g
        else:
            x = x.flatten(start_dim=1) 
            x = self.manual_group_norm(x, num_groups=H, weight=self.att_group_norm_weight, bias=self.att_group_norm_bias) * g
        
        # 应用输出层并返回结果
        return self.att_output(x)

    def construct(self, x: mindspore.Tensor, state: mindspore.Tensor, i: int) -> mindspore.Tensor:
        """
        模型的前向传播。
        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, N_embd]。
            state (mindspore.Tensor): 隐藏状态张量，形状为[Batch, State Size, N_embd]。
            i (int): 时间索引。
        Returns:
            mindspore.Tensor: 前向传播结果张量，形状与输入的x相同。
        """
        if self.onnx_opset >= 17:
            x = x + self.time_mixing(self.ln1(x), state, i)
            x = x + self.channel_mixing(self.ln2(x), state, i)
        else:
            x = x + self.time_mixing(self.manual_layer_norm(x, self.ln1_weight, self.ln1_bias, 1e-5), state, i)
            x = x + self.channel_mixing(self.manual_layer_norm(x, self.ln2_weight, self.ln2_bias, 1e-5), state, i)
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
        try:
            self.onnx_opset = int(args['onnx_opset'])
        except:
            self.onnx_opset = 16 #默认是最低的，op17版本才支持LayerNorm算子，op18版本才支持GroupNorm算子
        print('onnx opset ', self.onnx_opset)
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

        if self.onnx_opset >= 17:
            self.ln0 = nn.LayerNorm(self.n_embd)
            self.ln0.weight = mindspore.Parameter(w['blocks.0.ln0.weight'])
            self.ln0.bias = mindspore.Parameter(w['blocks.0.ln0.bias'])
        else:
            self.ln0_weight = mindspore.Parameter(w['blocks.0.ln0.weight'])
            self.ln0_bias = mindspore.Parameter(w['blocks.0.ln0.bias'])

        self.blocks = nn.CellList()
        
        for i in range(self.num_layer):
            # 提取当前块的权重
            block_w = {k[len(f'blocks.{i}.'):]: v for k, v in w.items() if f'blocks.{i}.' in k}
            self.blocks.append(RWKV_Block(block_w, self.n_embd, self.n_head, self.onnx_opset))
            print(f"Loading blocks...[{i + 1}/{self.num_layer}]", end='\r')
        print()

        if self.onnx_opset >= 17:
            self.ln_out = nn.LayerNorm(self.n_embd)
            self.ln_out.weight = mindspore.Parameter(w['ln_out.weight'])
            self.ln_out.bias = mindspore.Parameter(w['ln_out.bias'])
        else:
            self.ln_out_weight = mindspore.Parameter(w['ln_out.weight'])
            self.ln_out_bias = mindspore.Parameter(w['ln_out.bias'])
        
        self.head = nn.Dense(self.n_embd, args['vocab_size'], has_bias=False)
        self.head.weight = mindspore.Parameter(w['head.weight'])

    def manual_layer_norm(self, x: mindspore.Tensor, weight: mindspore.Tensor, bias: mindspore.Tensor, eps: float = 1e-5) -> mindspore.Tensor:
        """
        人工层归一化函数
        Args:
            x (mindspore.Tensor): 输入张量，形状为 [Batch, *]，* 表示任意维度。
            weight (mindspore.Tensor): 归一化的权重张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            bias (mindspore.Tensor): 归一化的偏置张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            eps (float): 用于数值稳定性的小值，防止除以零。
        Returns:
            mindspore.Tensor: 经过手动层归一化后的张量，形状与输入的 x 相同。
        """
        mean = x.mean(axis=-1, keep_dims=True)
        var = x.var(axis=-1, keepdims=True, ddof=False)
        x_normalized = (x - mean) / ops.sqrt(var + eps)
        x_scaled = x_normalized * weight
        x_shifted = x_scaled + bias
        return x_shifted

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
        if self.onnx_opset >= 17:
            x = self.ln0(x)
        else:
            x = self.manual_layer_norm(x, self.ln0_weight, self.ln0_bias, 1e-5)
        # 开始循环推理RWKV Block    
        for i, block in enumerate(self.blocks):
            x = block(x, state, i)
            
        if self.onnx_opset >= 17:
            x = self.ln_out(x)
        else:
            x = self.manual_layer_norm(x, self.ln_out_weight, self.ln_out_bias, 1e-5) 
        x = self.head(x)
        return x, state
