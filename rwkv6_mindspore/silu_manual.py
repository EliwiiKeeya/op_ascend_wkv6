# -*- encoding: utf-8 -*-
# @File        : silu_manual.py
# @Date        : 2024/10/04 16:48:30
# @Author      : Eliwii_Keeya

import mindspore
from mindspore import ops, nn, Tensor

class SiLUManual(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = ops.Mul()
        self.sigmoid = ops.Sigmoid()
    
    def construct(self, x: Tensor) -> Tensor:
        return self.mul(x, self.sigmoid(x))

def silu_manual(x: Tensor) -> Tensor:
    return ops.mul(x, ops.sigmoid(x))

if __name__ == '__main__':
    x = mindspore.Tensor([-1, 2, -3, 2, -1], dtype=mindspore.float32)
    silu = SiLUManual()
    x_out = silu(x)
    print(x_out)

    x_out = silu_manual(x)
    print(x_out)