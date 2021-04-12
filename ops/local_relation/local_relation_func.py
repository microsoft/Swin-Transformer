# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2019 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Han Hu, Jiarui Xu
# Modified by Ze Liu
# --------------------------------------------------------

import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from . import local_relation_cuda


class LocalRelationFunction(Function):

    @staticmethod
    def forward(ctx,
                query,
                key,
                value,
                pos_weight,
                kernel_size,
                groups,
                stride=1,
                dilation=1,
                scale=1.,
                no_define_value=-100.,
                norm_method=0,
                sim_method=0,
                batch_step=32):
        for input in [query, key, value]:
            if input is not None and input.dim() != 4:
                raise ValueError(
                    "Expected 4D tensor as input, got {}D tensor instead.".format(
                        input.dim()))
        ctx.kernel_size = _pair(kernel_size)
        ctx.groups = groups
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.scale = scale
        ctx.no_define_value = no_define_value
        ctx.norm_method = norm_method
        ctx.sim_method = sim_method
        ctx.batch_step = batch_step

        ctx.save_for_backward(query, key, value, pos_weight)

        output = query.new_empty(
            LocalRelationFunction._output_size(query, value))

        scale_tensor = query.new_tensor([ctx.scale])
        no_define_value_tensor = query.new_tensor([ctx.no_define_value])

        if not input.is_cuda:
            raise NotImplementedError
        else:
            batch_step = min(ctx.batch_step, query.shape[0])
            local_relation_cuda.local_relation_forward_cuda(
                query, key, value, pos_weight, scale_tensor, no_define_value_tensor,
                output, ctx.kernel_size[1], ctx.kernel_size[0], ctx.groups,
                ctx.dilation, ctx.stride, batch_step, ctx.norm_method, ctx.sim_method)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, pos_weight = ctx.saved_tensors

        grad_query = grad_key = grad_value = grad_pos_weight = None

        scale_tensor = query.new_tensor(ctx.scale)
        no_define_value_tensor = query.new_tensor(ctx.no_define_value)

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            batch_step = min(ctx.batch_step, query.shape[0])

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
                grad_query = torch.zeros_like(query)
                grad_key = torch.zeros_like(key)
                grad_value = torch.zeros_like(value)
                grad_pos_weight = torch.zeros_like(pos_weight)
                local_relation_cuda.local_relation_backward_cuda(
                    query, key, value, pos_weight,
                    scale_tensor, no_define_value_tensor, grad_output,
                    grad_query, grad_key, grad_value, grad_pos_weight,
                    ctx.kernel_size[1], ctx.kernel_size[0],
                    ctx.groups, ctx.dilation, ctx.stride, batch_step,
                    ctx.norm_method, ctx.sim_method)

        return (grad_query, grad_key, grad_value, grad_pos_weight, None, None, None,
                None, None, None, None, None, None)

    @staticmethod
    def _output_size(query, value):
        output_size = (query.size(0), value.size(1), query.size(2), query.size(3))
        return output_size


local_relation = LocalRelationFunction.apply
