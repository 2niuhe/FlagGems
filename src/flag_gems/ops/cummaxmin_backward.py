import logging

import torch

from .scatter import ScatterFunction

_scatter_func = ScatterFunction()


def cummaxmin_backward(grad, input, indices, dim):
    logging.debug("GEMS CUMMAXMIN_BACKWARD")

    inp = grad.contiguous()
    index = indices.contiguous()
    src = input.contiguous()
    out = torch.zeros_like(inp)

    src_strided = src.as_strided(index.shape, src.stride()).contiguous()

    N = list(index.shape)[index.ndim - 1]
    M = index.numel() // N

    _scatter_func(src_strided, index, inp, out, dim, M, N, "add")
    return out
