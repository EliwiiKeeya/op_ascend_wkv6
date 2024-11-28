# -*- encoding: utf-8 -*-
# @File        : main.py
# @Date        : 2024/10/04 16:28:43
# @Author      : Eliwii_Keeya
# @Modified from: yuunnn-w, et al., 2024 -- RWKV_Pytorch

import time, os
import mindspore
from mindspore import ops, Profiler
from mindspore._c_expression import _framework_profiler_step_start, _framework_profiler_step_end
from model import RWKV_RNN
from rwkv_tokenizer import RWKV_TOKENIZER
from sampler import sample_logits


if __name__ == '__main__':
    mindspore.set_context(device_target="Ascend")
    mindspore.run_check()
    profiler = Profiler(output_path=f"op_ascend_wkv6/rwkv6_mindspore/log/log_{int(time.time())}")

    args = {
        'MODEL_NAME': 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096/RWKV-x060-World-1B6-v2.1-20240328-ctx4096', #模型文件的名字，ckpt结尾的权重文件。
        'vocab_size': 65536, #词表大小
        'onnx_opset': '12',
    }

    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args)
    tokenizer = RWKV_TOKENIZER("op_ascend_wkv6/rwkv6_mindspore/asset/rwkv_vocab_v20230424.txt")
    print("Done.")

    # 设置续写的初始字符串和参数
    initial_string = "Hello,"
    batch_size = 1
    TEMPERATURE = 2.5  # 温度参数
    TOP_P = 0.1  # Top-p采样参数
    LENGTH_PER_TRIAL = 20  # 生成的长度

    # 编码初始字符串
    encoded_input = tokenizer.encode([initial_string] * batch_size)
    token = mindspore.Tensor(encoded_input, dtype=mindspore.int64)      # 转置以匹配模型输入的形状

    # 初始化状态
    state = ops.zeros((batch_size, model.state_size[0], model.state_size[1]))

    #并行预填充
    token_out, state_out = model.forward_parallel(token, state)

    out = token_out[:,-1] # 取最后一个生成的token

    start_time = time.time() # 开始计时
    
    for step in range(LENGTH_PER_TRIAL):  # 生成指定数量的token
        _framework_profiler_step_start()
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P).type_as(token)
        token = ops.cat((token, token_sampled.unsqueeze(1)), 1)
        out, state = model(token_sampled, state)
        # 清除屏幕并打印结果
        os.system('cls' if os.name == 'nt' else 'clear')
        decoded_sequences = tokenizer.decode(token.tolist())
        for i, seq in enumerate(decoded_sequences):
            print(f"Batch {i+1}: {seq}")
        _framework_profiler_step_end()

    end_time = time.time() # 结束计时

    total_time = end_time - start_time
    tokens_generated = LENGTH_PER_TRIAL * batch_size
    speed = tokens_generated / total_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Token generation speed: {speed:.2f} tokens/second")
