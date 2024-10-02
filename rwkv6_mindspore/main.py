import mindspore
from rwkv6_mindspore.model import RWKV_RNN

if __name__ == '__main__':
    mindspore.set_context(device_target="CPU")
    mindspore.run_check()
    args = {
        'MODEL_NAME': '/mnt/e/Resources/Models/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096', #模型文件的名字，ckpt结尾的权重文件。
        'vocab_size': 65536, #词表大小
        'onnx_opset': '12',
    }

    model = RWKV_RNN(args)
    print(model)
