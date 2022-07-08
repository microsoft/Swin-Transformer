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

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <stdio.h>

int best_block_dim(int feat_dim){
    int best_dim;
    if (feat_dim < 384){
        best_dim = 64;
    }
    else{
        if (feat_dim < 1024){
            best_dim = 128;
        }
        else{
            best_dim = 256;
        }
    }
    return best_dim;
}


template <typename T>
__global__ void roll_and_window_partition_forward_cuda_kernel(
    T* input, 
    T* output, 
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size,
    const int nH,
    const int nW){
    // start
    //bool qual = threadIdx.x < C;
    int index = threadIdx.x;
    int offset;
    for (int i = index; i < C; i += blockDim.x) {
        offset = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * C + i; // C = blocksize
        int input_offset = blockIdx.z / (nH * nW) * H * W * C +
            (blockIdx.z % (nH * nW) / nW * window_size + blockIdx.y - shift_size + H) % H * W * C + 
            (blockIdx.z % nW * window_size + blockIdx.x - shift_size + W) % W * C +
            i;
        output[offset] = (T)(__ldg(input + input_offset));
    }
}


template <typename T>
__global__ void roll_and_window_partition_backward_cuda_kernel(
    T* grad_in, 
    T* grad_out, 
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size,
    const int nH,
    const int nW){
    // start
    int index = threadIdx.x;
    int offset;
    for (int i = index; i < C; i += blockDim.x) {
        offset = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * C + i; // C = blocksize
        int input_offset = 
        (blockIdx.z * nH * nW + (blockIdx.y + shift_size + H) % H / window_size * nW + (blockIdx.x + shift_size + W) % W / window_size) * window_size * window_size * C +
        (blockIdx.y + shift_size + H ) % H % window_size * window_size * C +
        (blockIdx.x + shift_size + W ) % W % window_size * C +
        i;
        grad_out[offset] = (T)(__ldg(grad_in + input_offset));
    }
}


template <typename T>
__global__ void window_merge_and_roll_forward_cuda_kernel(
    T* input, 
    T* output, 
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size,
    const int nH,
    const int nW){
    // start
    int index = threadIdx.x;
    int offset;
    for (int i = index; i < C; i += blockDim.x) {
        offset = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * C + i; // C = blocksize
        int input_offset = 
            (blockIdx.z * nH * nW + (blockIdx.y - shift_size + H) % H / window_size * nH + (blockIdx.x - shift_size + W) % W / window_size) * window_size * window_size * C +
            (blockIdx.y - shift_size + H) % window_size * window_size * C + 
            (blockIdx.x - shift_size + W) % window_size * C +
            i;
        output[offset] = (T)(__ldg(input + input_offset));
    }
}



template <typename T>
__global__ void window_merge_and_roll_backward_cuda_kernel(
    T* grad_in, 
    T* grad_out, 
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size,
    const int nH,
    const int nW){
    // start
    int index = threadIdx.x;
    int offset;
    for (int i = index; i < C; i += blockDim.x) {
        offset = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * C + i; // C = blocksize
        int input_offset = 
        (blockIdx.z / (nH * nW)) * H * W * C +
        (blockIdx.z % (nH * nW) / nW * window_size + blockIdx.y + shift_size + H) % H * W * C +
        (blockIdx.z % nW * window_size + blockIdx.x + shift_size + W) % W * C +
        i;
        grad_out[offset] = (T)(__ldg(grad_in + input_offset));
    }
}

// input: [B, H, W, C]
// output: [B*nH*nW, window_size, window_size, C]
at::Tensor roll_and_window_partition_forward_cuda(
    at::Tensor & input, 
    //at::Tensor & output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size){
    
    int nH = H / window_size;
    int nW = W / window_size;

    dim3 grid(window_size, window_size, B * nH * nW);
    //dim3 block((C + 31) / 32 * 32);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    at::Tensor output;
    if (input.scalar_type() == torch::kFloat16){
        output = torch::empty({B*nH*nW, window_size, window_size, C}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(true));
    }
    else{
        output = torch::empty({B*nH*nW, window_size, window_size, C}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "roll_and_window_partition_forward_cuda_kernel", ([&] {
        roll_and_window_partition_forward_cuda_kernel<scalar_t><<<grid, block, 0>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            B,
            H,
            W,
            C,
            shift_size,
            window_size,
            nH,
            nW);
    }));
    return output;
}


// grad_in: [B*nH*nW, window_size, window_size, C]
// grad_out: [B, H, W, C]
at::Tensor roll_and_window_partition_backward_cuda(
    at::Tensor & grad_in, 
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size){
    
    int nH = H / window_size;
    int nW = W / window_size;

    dim3 grid(W, H, B);
    //dim3 block((C + 31) / 32 * 32);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    at::Tensor grad_out;
    if (grad_in.scalar_type() == torch::kFloat16){
        grad_out = torch::empty({B, H, W, C}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    }
    else{
        grad_out = torch::empty({B, H, W, C}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_in.type(), "roll_and_window_partition_backward_cuda_kernel", ([&] {
        roll_and_window_partition_backward_cuda_kernel<scalar_t><<<grid, block, 0>>>(
            grad_in.data<scalar_t>(),
            grad_out.data<scalar_t>(),
            B,
            H,
            W,
            C,
            shift_size,
            window_size,
            nH,
            nW);
    }));
    return grad_out;
}


// input: [B*nH*nW, window_size, window_size, C]
// output: [B, H, W, C]
at::Tensor window_merge_and_roll_forward_cuda(
    at::Tensor & input, 
    //at::Tensor & output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size){
    
    int nH = H / window_size;
    int nW = W / window_size;

    dim3 grid(W, H, B);
    //dim3 block((C + 31) / 32 * 32);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    //generate output tensor inside
    at::Tensor output;
    if (input.scalar_type() == torch::kFloat16){
        output = torch::empty({B, H, W, C}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(true));
    }
    else{
        output = torch::empty({B, H, W, C}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "window_merge_and_roll_forward_cuda_kernel", ([&] {
        window_merge_and_roll_forward_cuda_kernel<scalar_t><<<grid, block, 0>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            B,
            H,
            W,
            C,
            shift_size,
            window_size,
            nH,
            nW);
    }));
    return output;
}


at::Tensor window_merge_and_roll_backward_cuda(
    at::Tensor & grad_in, 
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size){
    
    int nH = H / window_size;
    int nW = W / window_size;

    dim3 grid(window_size, window_size, B * nH * nW);
    //dim3 block((C + 31) / 32 * 32);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    at::Tensor grad_out;
    if (grad_in.scalar_type() == torch::kFloat16){
        grad_out = torch::empty({B*nH*nW, window_size, window_size, C}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    }
    else{
        grad_out = torch::empty({B*nH*nW, window_size, window_size, C}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_in.type(), "window_merge_and_roll_backward_cuda_kernel", ([&] {
        window_merge_and_roll_backward_cuda_kernel<scalar_t><<<grid, block, 0>>>(
            grad_in.data<scalar_t>(),
            grad_out.data<scalar_t>(),
            B,
            H,
            W,
            C,
            shift_size,
            window_size,
            nH,
            nW);
    }));
    return grad_out;
}