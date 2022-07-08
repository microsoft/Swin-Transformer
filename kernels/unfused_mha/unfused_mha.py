# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import softmax_cuda


class FusedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, relative_pos_bias, attn_mask, batch_size, window_num, num_head, window_len):
        if attn_mask is not None:
            softmax_cuda.softmax_fwd(input, relative_pos_bias, attn_mask, batch_size, window_num, num_head, window_len)
        else:
            softmax_cuda.softmax_nomask_fwd(input, relative_pos_bias, batch_size, window_num, num_head, window_len)
        
        ctx.save_for_backward(input)

        # save for backward for int/float values
        ctx.batch_size = batch_size
        ctx.window_num = window_num
        ctx.num_head = num_head
        ctx.window_len = window_len
        return input

    @staticmethod
    def backward(ctx, grad_out):
        softmax_result = ctx.saved_tensors[0]
        batch_size = ctx.batch_size
        window_num = ctx.window_num
        num_head = ctx.num_head
        window_len = ctx.window_len

        softmax_cuda.softmax_bwd(
            grad_out.contiguous(), softmax_result, batch_size, window_num, num_head, window_len)

        return grad_out, torch.sum(grad_out, dim=0), None, None, None, None, None