# -*- encoding: utf-8 -*-
# @File        : convert_weight.py
# @Date        : 2024/10/04 16:28:57
# @Author      : Eliwii_Keeya
# @Usage       : https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html#未支持模型权重转换开发


import os
import torch
import mindspore as ms
from mindformers.utils.convert_utils import pt2ms


def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        model_torch = torch.load(input_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Do not find model in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False
    
    ckpt_list = []
    for name, value in model_torch.items():
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True
