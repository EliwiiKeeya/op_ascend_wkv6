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
    #     name = name_replace(name)
    #     if name == 'norm.weight':
    #         name = 'norm_out.weight'
    #     if name[:7] == 'layers.':
    #         name = name[7:]

        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True
