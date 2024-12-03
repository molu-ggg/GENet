"""
STG-NF modules, based on awesome previous work by https://github.com/y0ast/Glow-PyTorch
"""


import math
import torch


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def split_feature(tensor, type="split", imgs=False):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if imgs:
        if type == "split":
            return tensor[:, : C // 2, ...], tensor[:, C // 2:, ...]
        elif type == "cross":
            return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

    if type == "split":
        return tensor[:, : C // 2, ...].squeeze(dim=1), tensor[:, C // 2 :, ...].squeeze(dim=1) #“numpy.squeeze() 这个函数的作用是去掉矩阵里维度为1的维度，应该没有；将特征向量按照通道分成两半，我没记错的话，通道只有两个，所以这里会减少一维度
    elif type == "cross":
        return tensor[:, 0::2, ...].squeeze(dim=1), tensor[:, 1::2, ...].squeeze(dim=1)
