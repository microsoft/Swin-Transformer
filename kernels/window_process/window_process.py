# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import swin_window_process


def to_pair(value):
    """Accept an int (isotropic) or a (h, w) iterable and return a (h, w) tuple.

    Keeps the public signature of WindowProcess unchanged: `window_size=7`
    behaves exactly as before, `window_size=(4, 8)` selects a rectangular window.
    """
    if isinstance(value, int):
        return value, value
    h, w = value
    return int(h), int(w)


class WindowProcess(torch.autograd.Function):
    """Fused torch.roll + window_partition.

    Args:
        input: (B, H, W, C) contiguous CUDA tensor. float64, float32,
            float16 and bfloat16 all dispatch.
        B, H, W, C: shape of `input`.
        shift_size: int or (shift_h, shift_w). This is the *negated* torch.roll
            shift, matching the existing call sites in models/swin_transformer.py.
        window_size: int or (window_h, window_w). Must divide H and W per axis.

    Returns:
        (B * nH * nW, window_h, window_w, C)
    """

    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        shift_h, shift_w = to_pair(shift_size)
        window_h, window_w = to_pair(window_size)
        output = swin_window_process.roll_and_window_partition_forward(
            input, B, H, W, C, shift_h, shift_w, window_h, window_w)

        ctx.B = B
        ctx.H = H
        ctx.W = W
        ctx.C = C
        ctx.shift_size = (shift_h, shift_w)
        ctx.window_size = (window_h, window_w)
        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W
        C = ctx.C
        shift_h, shift_w = ctx.shift_size
        window_h, window_w = ctx.window_size

        grad_out = swin_window_process.roll_and_window_partition_backward(
            grad_in.contiguous(), B, H, W, C, shift_h, shift_w, window_h, window_w)
        return grad_out, None, None, None, None, None, None


class WindowProcessReverse(torch.autograd.Function):
    """Fused window merge + torch.roll, the inverse of WindowProcess.

    Args:
        input: (B * nH * nW, window_h, window_w, C) contiguous CUDA tensor.
            float64, float32, float16 and bfloat16 all dispatch.
        B, H, W, C: shape of the *output* feature map.
        shift_size: int or (shift_h, shift_w), the torch.roll shift.
        window_size: int or (window_h, window_w). Must divide H and W per axis.

    Returns:
        (B, H, W, C)
    """

    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        shift_h, shift_w = to_pair(shift_size)
        window_h, window_w = to_pair(window_size)
        output = swin_window_process.window_merge_and_roll_forward(
            input, B, H, W, C, shift_h, shift_w, window_h, window_w)

        ctx.B = B
        ctx.H = H
        ctx.W = W
        ctx.C = C
        ctx.shift_size = (shift_h, shift_w)
        ctx.window_size = (window_h, window_w)

        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W
        C = ctx.C
        shift_h, shift_w = ctx.shift_size
        window_h, window_w = ctx.window_size

        grad_out = swin_window_process.window_merge_and_roll_backward(
            grad_in.contiguous(), B, H, W, C, shift_h, shift_w, window_h, window_w)
        return grad_out, None, None, None, None, None, None
