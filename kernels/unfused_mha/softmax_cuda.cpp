/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#include <torch/torch.h>
#include <torch/extension.h>
 
// if return multiple tensors
// use std::vector<at::Tensor>

// CUDA forward declaration
void softmax_fwd_cuda(
    at::Tensor & input, 
    const at::Tensor & relative_pos_bias,
    const at::Tensor & attn_mask,
    //const at::Tensor & random_tensor,
    //at::Tensor & softmax_result,
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len);

void softmax_nomask_fwd_cuda(
    at::Tensor & input, 
    const at::Tensor & relative_pos_bias,
    //const at::Tensor & random_tensor,
    //at::Tensor & softmax_result,
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len);


// CUDA backward declaration
void softmax_bwd_cuda(
    at::Tensor & grad, 
    const at::Tensor & softmax_result,
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void softmax_fwd(
    at::Tensor & input, 
    const at::Tensor & relative_pos_bias,
    const at::Tensor & attn_mask,
    //const at::Tensor & random_tensor,
    //at::Tensor & softmax_result,
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len) {
    //CHECK_INPUT(input);
    return softmax_fwd_cuda(input, relative_pos_bias, attn_mask, batch_size, window_num, num_head, window_len);
}

void softmax_nomask_fwd(
    at::Tensor & input, 
    const at::Tensor & relative_pos_bias,
    //const at::Tensor & random_tensor,
    //at::Tensor & softmax_result,
    const int batch_size, 
    const int window_num,
    const int num_head,
    const int window_len) {
    //CHECK_INPUT(input);
    return softmax_nomask_fwd_cuda(input, relative_pos_bias, batch_size, window_num, num_head, window_len);
}



void softmax_bwd(
    at::Tensor & grad, 
    const at::Tensor & softmax_result, 
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len) {
    CHECK_INPUT(grad);
    CHECK_INPUT(softmax_result);
    return softmax_bwd_cuda(grad, softmax_result, batch_size, window_num, num_head, window_len);
}


 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_fwd", &softmax_fwd, "Softmax forward pass.");
    m.def("softmax_nomask_fwd", &softmax_nomask_fwd, "Softmax nomask forward pass.");
    m.def("softmax_bwd", &softmax_bwd, "Softmax backward pass.");
}