/*!
 * Copyright (c) 2019 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file local_relation_cuda.cpp
 * \brief
 * \author Han Hu
 * \modified by Jiarui Xu, Ze Liu
*/

#include <torch/extension.h>

#include <cmath>
#include <vector>

void similarity_compute_forward(
	const at::Tensor key, 
	const at::Tensor query, 
	const at::Tensor pos_weight,
	const int batch_size, 
	const int key_channels,
	const int query_channels,
	const int height, 
	const int width,
	const int kernel_height,
	const int kernel_width,
	const int num_group,
	const at::Tensor scale,
	const at::Tensor no_define_value,
	const int dilate,
	const int stride, 
	const int in_height,
	const int in_width,
	const int sim_method,
	at::Tensor output,
	const int key_offset,
	const int query_offset);

void similarity_compute_backward(
	const at::Tensor key, 
	const at::Tensor query, 
	const at::Tensor output_grad,
	const int batch_size, 
	const int key_channels,
	const int query_channels,
	const int height, 
	const int width,
	const int kernel_height,
	const int kernel_width,
	const int num_group,
	const int key_per_group,
	const at::Tensor scale,
	const int dilate,
	const int stride,
	const int in_height,
	const int in_width,
	const int sim_method,
	at::Tensor key_grad,
	at::Tensor query_grad,
	const int key_grad_offset,
	const int query_grad_offset);

void aggregation_forward(
	const at::Tensor value, 
	const at::Tensor softmax_data,
	const int batch_size, 
	const int value_channels,
	const int height, 
	const int width,
	const int kernel_height,
	const int kernel_width,
	const int num_group,
	const int dilate,
	const int stride,
	const int in_height,
	const int in_width,
	at::Tensor output,
	const int value_offset,
	const int output_offset);

void aggregation_value_backward(
	const at::Tensor softmax_data,
	const at::Tensor output_grad,
	const int batch_size, 
	const int value_channels,
	const int height, 
	const int width,
	const int kernel_height,
	const int kernel_width,
	const int num_group,
	const int dilate,
	const int stride,
	const int in_height,
	const int in_width,
	at::Tensor value_grad,
	const int output_grad_offset,
	const int value_grad_offset);

void aggregation_softmax_backward(
	const at::Tensor value,
	const at::Tensor output_grad,
	const int batch_size, 
	const int value_channels,
	const int height, 
	const int width,
	const int kernel_height,
	const int kernel_width,
	const int num_group,
	const int dilate,
	const int stride,
	const int in_height,
	const int in_width,
	at::Tensor softmax_grad,
	const int value_offset,
	const int output_grad_offset);


int local_relation_forward_cuda(
	at::Tensor query,
	at::Tensor key,
	at::Tensor value,
	at::Tensor pos_weight, 
	at::Tensor scale,
	at::Tensor no_define_value,
	at::Tensor output,
	const int kernel_height,
	const int kernel_width,
	const int num_group, 
	const int dilate,
	const int stride,
	const int batch_step,
	const int norm_method,
	const int sim_method)
{
	query = query.contiguous();
	key = key.contiguous();
	value = value.contiguous();
	pos_weight = pos_weight.contiguous();

	const int query_channels = query.size(1);
	const int key_channels = key.size(1);
	const int value_channels = value.size(1);
	const int batch_size = key.size(0);
	const int height = query.size(2);
	const int width = query.size(3);
	const int in_height = key.size(2);
	const int in_width = key.size(3);

	const int batch_step_ = std::min(batch_size, batch_step);
	const int sim_size = batch_step_ * num_group * kernel_height * kernel_width * height * width;

    const int key_step = batch_step_ * key_channels * in_height * in_width;
    const int query_step = batch_step_ * query_channels * height * width;
    const int value_step = batch_step_ * value_channels * in_height * in_width;
    const int output_step = batch_step_ * value_channels * height * width;

    at::Tensor sim_buffer = at::zeros({batch_step_ * num_group, kernel_height * kernel_width, height * width},
                query.options());

    at::Tensor softmax_buffer = at::zeros({batch_step_ * num_group, kernel_height * kernel_width, height * width},
                query.options());

    at::Tensor sum_softmax_buffer = at::zeros({batch_step_ * num_group, height * width});

    int M = (batch_size - 1) / batch_step_ + 1;
    for (int i = 0; i < M; ++i) {
        int cur_batch_step = batch_step_;
        if (i == M - 1) {
            cur_batch_step = batch_size - (M - 1) * batch_step_;
			if (cur_batch_step != batch_step_) {
				sim_buffer = at::zeros({cur_batch_step * num_group, kernel_height * kernel_width, height * width}, query.options());
				softmax_buffer = at::zeros({cur_batch_step * num_group, kernel_height * kernel_width, height * width},query.options());
				sum_softmax_buffer = at::zeros({cur_batch_step * num_group, height * width}, query.options());
			}

            // TORCH_CHECK(cur_batch_step % batch_step_ == 0, "batch_step must be divided by batch_size");
        }
        similarity_compute_forward(key, query, pos_weight, cur_batch_step,
        	key_channels, query_channels, height, width, 
        	kernel_height, kernel_width, num_group, scale, no_define_value, 
        	dilate, stride, in_height, in_width, sim_method, sim_buffer, 
        	key_step * i, query_step * i);

	    // softmax 
        if (norm_method == 0) {
        	softmax_buffer = sim_buffer.softmax(1);
        }
        else {
        	AT_ERROR("Not implemented yet");
        }

        aggregation_forward(value, softmax_buffer, cur_batch_step,
        	value_channels, height, width, kernel_height, kernel_width, 
        	num_group, dilate, stride, in_height, in_width, output, value_step * i, output_step * i);
    }

    return 1;

}

int local_relation_backward_cuda(
	at::Tensor query,
	at::Tensor key,
	at::Tensor value,
	at::Tensor pos_weight, 
	at::Tensor scale,
	at::Tensor no_define_value,
	at::Tensor output_grad,
	at::Tensor query_grad,
	at::Tensor key_grad,
	at::Tensor value_grad,
	at::Tensor pos_weight_grad,
	const int kernel_height,
	const int kernel_width,
	const int num_group, 
	const int dilate,
	const int stride,
	const int batch_step,
	const int norm_method,
	const int sim_method)
{
	query = query.contiguous();
	key = key.contiguous();
	value = value.contiguous();
	pos_weight = pos_weight.contiguous();

	output_grad = output_grad.contiguous();
	query_grad = query_grad.contiguous();
	key_grad = key_grad.contiguous();
	value_grad = value_grad.contiguous();
	pos_weight_grad = pos_weight_grad.contiguous();

	const int query_channels = query.size(1);
	const int key_channels = key.size(1);
	const int value_channels = value.size(1);
	const int batch_size = key.size(0);
	const int height = query.size(2);
	const int width = query.size(3);
	const int in_height = key.size(2);
	const int in_width = key.size(3);
    const int key_per_group = query_channels / num_group;

	const int batch_step_ = std::min(batch_size, batch_step);
	const int sim_size = batch_step_ * num_group * kernel_height * kernel_width * height * width;

    const int key_step = batch_step_ * key_channels * in_height * in_width;
    const int query_step = batch_step_ * query_channels * height * width;
    const int value_step = batch_step_ * value_channels * in_height * in_width;
    const int output_step = batch_step_ * value_channels * height * width;

    at::Tensor sim_buffer = at::zeros({batch_step_ * num_group, kernel_height * kernel_width, height * width},
                query.options());

    at::Tensor softmax_buffer = at::zeros({batch_step_ * num_group, kernel_height * kernel_width, height * width},
                query.options());

    at::Tensor sum_softmax_buffer = at::zeros({batch_step_ * num_group, height * width}, 
    	query.options());

    at::Tensor sim_grad_buffer = at::zeros({batch_step_ * num_group, kernel_height * kernel_width, height * width}, 
    	query.options());

	int M = (batch_size - 1) / batch_step_ + 1;

    const int pos_weight_size = num_group * kernel_height * kernel_width;

    for (int i = 0; i < M; ++i) {
        int cur_batch_step = batch_step_;
        if (i == M - 1) {
            cur_batch_step = batch_size - (M - 1) * batch_step_;
			if (cur_batch_step != batch_step_) {
				sim_buffer = at::zeros({cur_batch_step * num_group, kernel_height * kernel_width, height * width}, query.options());
				softmax_buffer = at::zeros({cur_batch_step * num_group, kernel_height * kernel_width, height * width},query.options());
				sum_softmax_buffer = at::zeros({cur_batch_step * num_group, height * width}, query.options());
				sim_grad_buffer = at::zeros({cur_batch_step * num_group, kernel_height * kernel_width, height * width}, query.options());
			}
            // TORCH_CHECK(cur_batch_step % batch_step_ == 0, "batch_step must be divided by batch_size");
        }

        similarity_compute_forward(key, query, pos_weight, cur_batch_step,
        	key_channels, query_channels, height, width, 
        	kernel_height, kernel_width, num_group, scale, no_define_value, 
        	dilate, stride, in_height, in_width, sim_method, sim_buffer, 
        	key_step * i, query_step * i);

	    // softmax 

        if (norm_method == 0) {
        	softmax_buffer = sim_buffer.softmax(1);
        }
        else {
        	AT_ERROR("Not implemented yet");
        }

        aggregation_value_backward(softmax_buffer, output_grad, cur_batch_step, 
        	value_channels, height, width, kernel_height, kernel_width, 
        	num_group, dilate, stride, in_height, in_width, value_grad, 
        	output_step * i, value_step * i);

        aggregation_softmax_backward(value, output_grad, cur_batch_step, 
        	value_channels, height, width, kernel_height, kernel_width, 
        	num_group, dilate, stride, in_height, in_width, sim_buffer, 
        	value_step * i, output_step * i);

        if (norm_method == 0) {
        	sum_softmax_buffer = (softmax_buffer * sim_buffer).sum(1, true);
        	sim_grad_buffer = softmax_buffer * (sim_buffer - sum_softmax_buffer);
        }
        else {
        	AT_ERROR("Not implemented yet");
        }

        similarity_compute_backward(key, query, sim_grad_buffer, cur_batch_step,
        	key_channels, query_channels, height, width, 
        	kernel_height, kernel_width, num_group, key_per_group, scale,
        	dilate, stride, in_height, in_width, sim_method, key_grad, query_grad,
        	key_step * i, query_step * i);
		
	    pos_weight_grad += sim_grad_buffer.view({cur_batch_step, num_group, kernel_height, kernel_width, height * width}).sum(4).sum(0);
    	
	}

    return 1;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("local_relation_forward_cuda", &local_relation_forward_cuda,
        "local relation forward (CUDA)");
  m.def("local_relation_backward_cuda", &local_relation_backward_cuda,
        "local relation backward (CUDA)");
}
