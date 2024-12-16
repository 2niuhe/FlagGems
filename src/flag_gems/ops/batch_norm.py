import logging
import math

import torch
import triton
import triton.language as tl

from .. import runtime
from ..utils import libentry
from ..utils import triton_lang_extension as tle
from ..utils.type_utils import get_accumulator_dtype

# --------------------------------------------------------------------------------------------------------------------
#   Kernel implementation
# --------------------------------------------------------------------------------------------------------------------
# TODO.boyue add autotune decorator back
@libentry()
@triton.jit(do_not_specialize=["epsilon"])
def kernel_bn_forward(Y, X_mean, X_stdvar, X_normed, X, weight_ptr, bias_ptr, stride_n, stride_c, stride_s,
                      epsilon: tl.constexpr,
                      SHAPE_N: tl.constexpr, SHAPE_C: tl.constexpr, SHAPE_S: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr):
    """
    Forward of batch norm kernel. Tiling along axis-(H, W, N).
    """
    pid_n, pid_s = tl.program_id(axis=0), tl.program_id(axis=1)

    # pointer arithmetic, and mask calculation
    ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    ofst_c = tl.arange(0, BLOCK_C)
    ofst_s = pid_s*BLOCK_S + tl.arange(0, BLOCK_S)
    offsets = (stride_n*ofst_n)[:, None, None] +\
              (stride_c*ofst_c)[None, :, None] +\
              (stride_s*ofst_s)[None, None, :]

    for idx_c in range(tl.cdiv(SHAPE_C, BLOCK_C)):
        mask_c = ofst_c < SHAPE_C

        # calculate mean of var of input x, for each channel
        accum_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
        accum_xx = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for idx_n in range(tl.cdiv(SHAPE_N, BLOCK_N)):
            mask_n = ofst_n < SHAPE_N

            for idx_s in range(tl.cdiv(SHAPE_S, BLOCK_S)):
                mask_s = ofst_s < SHAPE_S

                mask = mask_n[:, None, None] * mask_c[None, :, None] * mask_s[None, None, :]
                x = tl.load(X + offsets, mask, other=0.0)
                x = tl.permute(x, (1, 0, 2)).reshape(BLOCK_C, BLOCK_N*BLOCK_S)
                accum_x += tl.sum(x, axis=1)
                accum_xx += tl.sum(x*x, axis=1)

                ofst_s += BLOCK_S
                offsets += BLOCK_S*stride_s

            ofst_s = pid_s*BLOCK_S + tl.arange(0, BLOCK_S)
            offsets -= BLOCK_S*tl.cdiv(SHAPE_S, BLOCK_S)*stride_s
            ofst_n += BLOCK_N
            offsets += BLOCK_N*stride_n

        mean_x = accum_x/(SHAPE_N*SHAPE_S)
        mean_xx = accum_xx/(SHAPE_N*SHAPE_S)
        stdvar = tl.sqrt(mean_xx - (mean_x*mean_x))
        tl.store(X_mean + ofst_c, accum_x, mask_c)
        tl.store(X_stdvar + ofst_c, stdvar, mask_c)

        ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
        offsets -= BLOCK_N*tl.cdiv(SHAPE_N, BLOCK_N)*stride_n

        # normalized input x, for each channel
        weight_ = tl.load(weight_ptr + ofst_c, mask_c, other=0.0)
        bias_ = tl.load(bias_ptr + ofst_c, mask_c, other=0.0)
        for idx_n in range(tl.cdiv(SHAPE_N, BLOCK_N)):
            mask_n = ofst_n < SHAPE_N

            for idx_s in range(tl.cdiv(SHAPE_S, BLOCK_S)):
                mask_s = ofst_s < SHAPE_S

                mask = mask_n[:, None, None] * mask_c[None, :, None] * mask_s[None, None, :]
                x = tl.load(X + offsets, mask, other=0.0)
                normed_x =  (x - mean_x[None, :, None])/(stdvar[None, :, None] + epsilon)
                result = normed_x*weight_[None, :, None] + bias_[None, :, None]
                tl.store(X_normed + offsets, normed_x, mask)
                tl.store(Y + offsets, result, mask)

                ofst_s += BLOCK_S
                offsets += BLOCK_S*stride_s

            ofst_s = pid_s*BLOCK_S + tl.arange(0, BLOCK_S)
            offsets -= BLOCK_S*tl.cdiv(SHAPE_S, BLOCK_S)*stride_s
            ofst_n += BLOCK_N
            offsets += BLOCK_N*stride_n

        ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
        offsets -= BLOCK_N*tl.cdiv(SHAPE_N, BLOCK_N)*stride_n

        ofst_c += BLOCK_C
        offsets += BLOCK_C*stride_c

@triton.jit
def _kernel_bn_grad_x(grad_x, normed_y, grad_y, grad_y_mean, prod_mean, gamma,
                      stride_n, stride_c, stride_s,
                      SHAPE_N: tl.constexpr, SHAPE_C: tl.constexpr, SHAPE_S: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_n, pid_s = tl.program_id(0), tl.program_id(1)

    ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    ofst_c = tl.arange(0, BLOCK_C)
    ofst_s = pid_s*BLOCK_S + tl.arange(0, BLOCK_S)
    offsets = (ofst_n*stride_n)[:, None, None] +\
              (ofst_c*stride_c)[None, :, None] +\
              (ofst_s*stride_s)[None, None, :]
    epsilon = 1E-5

    for idx_c in range(tl.cdiv(SHAPE_C, BLOCK_C)):
        mask_c = ofst_c < SHAPE_C

        gamma_tile = tl.load(gamma + ofst_c, mask_c, other=0.0)
        grad_y_mean_tile = tl.load(grad_y_mean + ofst_c, mask_c)
        prod_mean_tile = tl.load(prod_mean + ofst_c, mask_c)
        for idx_n in range(tl.cdiv(SHAPE_N, BLOCK_N)):
            mask_n = ofst_n < SHAPE_N

            for idx_s in range(tl.cdiv(SHAPE_S, BLOCK_S)):
                mask_s = ofst_s < SHAPE_S

                mask = mask_n[:, None, None] * mask_c[None, :, None] * mask_s[None, None, :]
                grad_y_tile = tl.load(grad_y + offsets, mask)
                normed_y_tile = tl.load(normed_y + offsets, mask)

                grad_x_tile = grad_y_tile - grad_y_mean_tile[None, :, None]
                grad_x_tile -= normed_y_tile*prod_mean_tile[None, :, None]
                grad_x_tile = grad_x_tile/(gamma_tile + epsilon)[None, :, None]

                tl.store(grad_x + offsets, grad_x_tile, mask)
                ofst_s += BLOCK_S
                offsets += BLOCK_S*stride_s

            ofst_s = pid_s*BLOCK_S + tl.arange(0, BLOCK_S)
            offsets -= BLOCK_S*tl.cdiv(SHAPE_S, BLOCK_S)*stride_s
            ofst_n += BLOCK_N
            offsets += BLOCK_N*stride_n

        ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
        offsets -= BLOCK_N*tl.cdiv(SHAPE_N, BLOCK_N)*stride_n
        ofst_c += BLOCK_C
        offsets += BLOCK_C*stride_c

@triton.jit
def _kernel_bn_sum_reduce(Y, X, stride_n, stride_c, stride_s,
                    mean: tl.constexpr,
                    SHAPE_N: tl.constexpr, SHAPE_C: tl.constexpr, SHAPE_S: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_n, pid_s = tl.program_id(0), tl.program_id(1)

    ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    ofst_c = tl.arange(0, BLOCK_C)
    ofst_s = pid_s*BLOCK_S + tl.arange(0, BLOCK_S)
    offsets = (ofst_n*stride_n)[:, None, None]
    offsets += (ofst_c*stride_c)[None, :, None]
    offsets += (ofst_s*stride_s)[None, None, :]

    for idx_c in range(tl.cdiv(SHAPE_C, BLOCK_C)):
        mask_c = ofst_c < SHAPE_C

        accum = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for idx_n in range(tl.cdiv(SHAPE_N, BLOCK_N)):
            mask_n = ofst_n < SHAPE_N

            for idx_s in range(tl.cdiv(SHAPE_S, BLOCK_S)):
                mask_s = ofst_s < SHAPE_S

                mask = mask_n[:, None, None] * mask_c[None, :, None] * mask_s[None, None, :]
                x = tl.load(X + offsets, mask, other=0.0)
                x = tl.permute(x, 1, 0, 2).reshape(BLOCK_C, BLOCK_N*BLOCK_S)
                accum += tl.sum(x, axis=1)

                ofst_s += BLOCK_S
                offsets += BLOCK_S*stride_s

            ofst_s = pid_s*BLOCK_S + tl.arange(0, BLOCK_S)
            offsets -= BLOCK_S*tl.cdiv(SHAPE_S, BLOCK_S)*stride_s
            ofst_n += BLOCK_N
            offsets += BLOCK_N*stride_n

        if mean:
            result = accum/(SHAPE_N*SHAPE_S)
            tl.store(Y + ofst_c, result, mask=mask_c)
        else:
            tl.store(Y + ofst_c, accum, mask=mask_c)

        ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
        offsets -= BLOCK_N*tl.cdiv(SHAPE_N, BLOCK_N)*stride_n
        ofst_c += BLOCK_C
        offsets += BLOCK_C*stride_c

# --------------------------------------------------------------------------------------------------------------------
#   Operator class
# -------------------------------------------------------------------------------------------------------------------
class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, epsilon=1e-05):
        assert (len(weight.shape) == len(bias.shape)) and (len(weight.shape) == 1)
        assert (len(x.shape) == 4) and (x.shape[1] == bias.shape[0]) and (weight.shape[0] == bias.shape[0])
        N, C, H, W = x.shape
        tiler = TileAssist()
        tile_shape = tiler.calc_tiling((N, C, H, W))

        _cdiv = lambda a, b: (a + b -1)//b
        grid = lambda meta: (_cdiv(meta["BLOCK_N"], tile_shape[0]),
                         _cdiv(meta["BLOCK_S"], tile_shape[2]))

        x = torch.reshape(x, shape=[N, C, -1])
        x_normed = torch.empty(N, C, H*W, dtype=torch.float32, device="cuda")
        x_mean = torch.empty(C, dtype=torch.float32, device="cuda")
        x_stdvar = torch.empty(C, dtype=torch.float32, device="cuda")
        y = torch.empty(N, C, H*W, dtype=torch.float32, device="cuda")
        kernel_bn_forward[grid](y, x_mean, x_stdvar, x_normed, x, weight, bias, *x.stride(), epsilon=epsilon,
                            SHAPE_N=N, SHAPE_C=C, SHAPE_S=H*W,
                            BLOCK_N=tile_shape[0], BLOCK_C=tile_shape[1], BLOCK_S=tile_shape[2])

        y = torch.reshape(y, shape=[N, C, H, -1])
        x_normed = torch.reshape(x_normed, [N, C, H, -1])
        x = torch.reshape(x, shape=[N, C, H, -1])

        ctx.save_for_backward(x_normed, weight, bias)
        return y

    @staticmethod
    def op_bn_grad_wrt_bias(grad_y: torch.Tensor):
        from tile_calc import TileAssist
        shape = grad_y.shape
        assert (len(shape) == 4)
        N, C, H, W = shape

        # calculation of launch grid, which will be reused by all kernels
        tiler = TileAssist()
        tile_shape = tiler.calc_tiling((N, C, H, W))
        grad_y = torch.reshape(grad_y, shape=[N, C, -1])

        _cdiv = lambda a, b: (a + b -1)//b
        grid = lambda meta: (_cdiv(meta["BLOCK_N"], tile_shape[0]),
                             _cdiv(meta["BLOCK_S"], tile_shape[2]))

        result = torch.empty(C, dtype=torch.float32, device="cuda")
        _kernel_bn_sum_reduce[grid](result, grad_y, *grad_y.stride(), mean=False,
                                    SHAPE_N=N, SHAPE_C=C, SHAPE_S=H*W,
                                    BLOCK_N=tile_shape[0], BLOCK_C=tile_shape[1], BLOCK_S=tile_shape[2])

        grad_y = torch.reshape(grad_y, shape=[N, C, H, W])
        return result

    @staticmethod
    def op_bn_grad_wrt_input(grad_y: torch.Tensor, normed_y: torch.Tensor, gamma: torch.Tensor, epsilon=1E-5):
        from tile_calc import TileAssist
        shape_grad_y = grad_y.shape
        shape_y = normed_y.shape
        assert (shape_grad_y == shape_y) and (len(shape_y) == 4)
        N, C, H, W = shape_y

        # calculation of launch grid, which will be reused by all kernels
        tiler = TileAssist()
        tile_shape = tiler.calc_tiling((N, C, H, W))

        _cdiv = lambda a, b: (a + b -1)//b
        grid = lambda meta: (_cdiv(meta["BLOCK_N"], tile_shape[0]),
                             _cdiv(meta["BLOCK_S"], tile_shape[2]))

        # fuse spatial H, W to S
        grad_y = torch.reshape(grad_y, shape=[N, C, -1])
        normed_y = torch.reshape(normed_y, shape=[N, C, -1])

        # mean of gradient wrt to y
        grad_y_mean = torch.empty(C, device="cuda")
        _kernel_bn_sum_reduce[grid](grad_y_mean, grad_y, *grad_y.stride(),
            mean=True, SHAPE_N=N, SHAPE_C=C, SHAPE_S=H*W,
            BLOCK_N=tile_shape[0], BLOCK_C=tile_shape[1], BLOCK_S=tile_shape[2])

        #mean of normded_y*grad_y
        prod_mean = torch.empty(C, device="cuda")
        _kernel_bn_fused_prodmean[grid](prod_mean, grad_y, normed_y, *grad_y.stride(),
            mean=True,
            SHAPE_N=N, SHAPE_C=C, SHAPE_S=H*W,
            BLOCK_N=tile_shape[0], BLOCK_C=tile_shape[1], BLOCK_S=tile_shape[2])

        # final result
        grad_x = torch.empty(N, C, H*W, device="cuda")
        _kernel_bn_grad_x[grid](grad_x, normed_y, grad_y, grad_y_mean, prod_mean, gamma, *grad_y.stride(),
            SHAPE_N=N, SHAPE_C=C, SHAPE_S=H*W,
            BLOCK_N=tile_shape[0], BLOCK_C=tile_shape[1], BLOCK_S=tile_shape[2])

        # restore spatial axis to H, W
        grad_y = torch.reshape(grad_y, shape=[N, C, H, W])
        normed_y = torch.reshape(normed_y, shape=[N, C, H, W])
        grad_x = torch.reshape(grad_x, shape=[N, C, H, W])

        return grad_y_mean, prod_mean, grad_x

    @staticmethod
    def op_bn_grad_wrt_weight(grad_y: torch.Tensor, normed_x: torch.Tensor):
        assert (grad_y.shape == normed_x.shape)

        N, C, H, W = grad_y.shape
        tiler = TileAssist()
        tile_shape = tiler.calc_tiling((N, C, H, W))

        _cdiv = lambda a, b: (a + b -1)//b
        grid = lambda meta: (_cdiv(meta["BLOCK_N"], tile_shape[0]),
                             _cdiv(meta["BLOCK_S"], tile_shape[2]))

        grad_y = torch.reshape(grad_y, shape=[N, C, -1])
        normed_x = torch.reshape(normed_x, shape=[N, C, -1])

        result = torch.empty(C, dtype=torch.float32, device="cuda")
        _kernel_bn_fused_prodmean[grid](result, grad_y, normed_x, *grad_y.stride(),
                   mean=False,
                   SHAPE_N=N, SHAPE_C=C, SHAPE_S=H*W,
                   BLOCK_N=tile_shape[0], BLOCK_C=tile_shape[1], BLOCK_S=tile_shape[2])

        grad_y = torch.reshape(grad_y, shape=[N, C, H, -1])
        normed_x = torch.reshape(normed_x, shape=[N, C, H, -1])
        return result

    @staticmethod
    def backward(ctx, out_grad, mean_grad, rstd_grad):
        return in_grad, None, weight_grad, bias_grad, None, None

    class TileAssist():
        def __init__(self, smem_size=40*1024, datum_size=4):
            self.smem_size = smem_size
            self.datum_size = datum_size
            self.BATCH_IDX, self.CHANNEL_IDX, self.SPATIAL_IDX = range(0, 3)
            self.trial_prio = [self.CHANNEL_IDX, self.BATCH_IDX, self.SPATIAL_IDX]
            self.dim_minima = [2, 1, 32]
            self.merged_shape = None

        def calc_tiling(self, shape):
            """Calculate tiling for a given shape, assume dim size specified in order of
               (BATCH, CHANNEL, H, W)
            """
            assert(len(shape) == 4)
            next_power2 = lambda x: 2**math.ceil(math.log2(x))
            self.merged_shape = shape[0], shape[1], shape[2]*shape[3]
            tile_shape = list(map(next_power2, self.merged_shape))
            logging.debug(f"Orig shape: {shape}")
            logging.debug(f"Merged shape: {self.merged_shape}")
            logging.debug(f"Tile shape (init): {tile_shape}")

            for dim_idx in self.trial_prio:
               solution = self.trial_split(dim_idx, tile_shape)
               if solution:
                   self.merged_shape = None
                   return solution

            self.merged_shape = None
            logging.debug(f"Tile shape (finial): {tile_shape}")
            return tile_shape

        def trial_split(self, dim_idx, tile_shape):
            assert(dim_idx < len(tile_shape))
            assert(self.merged_shape and (dim_idx < len(self.merged_shape)))
            tile_shape[dim_idx] = self.dim_minima[dim_idx]

            product = lambda a, b: a*b
            tile_size = lambda shape: reduce(product, shape)*self.datum_size
            if tile_size(tile_shape) > self.smem_size:
                return None

            while True:
                if tile_size(tile_shape) > self.smem_size:
                    break
                if tile_shape[dim_idx] >= self.merged_shape[dim_idx]:
                    break
                tile_shape[dim_idx] = tile_shape[dim_idx] * 2

            if tile_size(tile_shape) > self.smem_size:
                tile_shape[dim_idx] = tile_shape[dim_idx] // 2

            return tile_shape

# --------------------------------------------------------------------------------------------------------------------
#   Main entry of the operator
# -------------------------------------------------------------------------------------------------------------------
def batch_norm(x, normalized_shape, weight, bias, eps=1e-5, cudnn_enable=True):
    return BatchNorm.apply(x, normalized_shape, weight, bias, eps, cudnn_enable)
