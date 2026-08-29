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
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Read-only cached load: __ldg on NVIDIA, a plain load on AMD.
//
// __ldg covers every dispatched type only under CUDA. c10::Half and
// c10::BFloat16 come from torch/headeronly/util and are not guarded alike:
// BFloat16 sits behind `__CUDACC__ || __HIPCC__` and degrades to *ptr on ROCm,
// Half behind `(__CUDA_ARCH__ >= 350) || (__clang__ && __CUDA__)`, which no
// hipcc build satisfies. Half is in the dispatch, so this file does not compile
// under ROCm as written. The hint is advisory in both languages -- it asks for
// the read-only path and may be ignored -- so dropping it on AMD changes no
// result. A macro rather than an inline function so that the NVIDIA build still
// sees the same __ldg(ptr) tokens as before.
#if defined(__HIP_PLATFORM_AMD__)
#define SWIN_WP_LDG(ptr) (*(ptr))
#else
#define SWIN_WP_LDG(ptr) __ldg(ptr)
#endif

// Width of the block that walks the channel dimension.
//
// No shared memory, no __syncthreads(), no warp-level primitives: this width
// affects occupancy only, never correctness, and nothing here depends on the
// warp or wavefront width -- CDNA's 64 against NVIDIA's 32 is immaterial. All
// three values below are multiples of 64, so neither platform runs a partial
// wave.
//
// The thresholds are NVIDIA-tuned. What CDNA confirms is the width they pick at
// swin-tiny's C, 64 at both stages: on an MI300X 64 beats 256 beats 1024, and
// at 1024 the fused path loses to eager. Both stages land in the first branch,
// so the 384 and 1024 thresholds themselves remain untested there. Override
// with -DSWIN_WP_BLOCK_DIM=N.
int best_block_dim(int feat_dim){
#ifdef SWIN_WP_BLOCK_DIM
    (void)feat_dim;
    return SWIN_WP_BLOCK_DIM;
#else
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
#endif
}


// The four kernels below are pure gathers: each one *is* the input_offset it
// computes. reference.py transcribes those expressions to PyTorch so they can
// be tested on a CPU. Nothing binds the two files -- only a build checks this
// source -- so if you change the arithmetic here, change reference.py too, or
// test_index_math.py will go on passing against the old formula.

// input:  [B, H, W, C]
// output: [B*nH*nW, window_h, window_w, C]
// grid:   (window_w, window_h, B*nH*nW) -- x indexes columns, y indexes rows
template <typename T>
__global__ void roll_and_window_partition_forward_cuda_kernel(
    T* input,
    T* output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w,
    const int nH,
    const int nW){
    // start
    //bool qual = threadIdx.x < C;
    int index = threadIdx.x;
    int offset;
    for (int i = index; i < C; i += blockDim.x) {
        offset = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * C + i; // C = blocksize
        int input_offset = blockIdx.z / (nH * nW) * H * W * C +
            (blockIdx.z % (nH * nW) / nW * window_h + blockIdx.y - shift_h + H) % H * W * C +
            (blockIdx.z % nW * window_w + blockIdx.x - shift_w + W) % W * C +
            i;
        output[offset] = (T)(SWIN_WP_LDG(input + input_offset));
    }
}


// grad_in:  [B*nH*nW, window_h, window_w, C]
// grad_out: [B, H, W, C]
// grid:     (W, H, B)
template <typename T>
__global__ void roll_and_window_partition_backward_cuda_kernel(
    T* grad_in,
    T* grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w,
    const int nH,
    const int nW){
    // start
    int index = threadIdx.x;
    int offset;
    for (int i = index; i < C; i += blockDim.x) {
        offset = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * C + i; // C = blocksize
        int input_offset =
        (blockIdx.z * nH * nW + (blockIdx.y + shift_h + H) % H / window_h * nW + (blockIdx.x + shift_w + W) % W / window_w) * window_h * window_w * C +
        (blockIdx.y + shift_h + H ) % H % window_h * window_w * C +
        (blockIdx.x + shift_w + W ) % W % window_w * C +
        i;
        grad_out[offset] = (T)(SWIN_WP_LDG(grad_in + input_offset));
    }
}


// input:  [B*nH*nW, window_h, window_w, C]
// output: [B, H, W, C]
// grid:   (W, H, B)
template <typename T>
__global__ void window_merge_and_roll_forward_cuda_kernel(
    T* input,
    T* output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w,
    const int nH,
    const int nW){
    // start
    int index = threadIdx.x;
    int offset;
    for (int i = index; i < C; i += blockDim.x) {
        offset = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * C + i; // C = blocksize
        int input_offset =
            (blockIdx.z * nH * nW + (blockIdx.y - shift_h + H) % H / window_h * nW + (blockIdx.x - shift_w + W) % W / window_w) * window_h * window_w * C +
            (blockIdx.y - shift_h + H) % H % window_h * window_w * C +
            (blockIdx.x - shift_w + W) % W % window_w * C +
            i;
        output[offset] = (T)(SWIN_WP_LDG(input + input_offset));
    }
}



// grad_in:  [B, H, W, C]
// grad_out: [B*nH*nW, window_h, window_w, C]
// grid:     (window_w, window_h, B*nH*nW)
template <typename T>
__global__ void window_merge_and_roll_backward_cuda_kernel(
    T* grad_in,
    T* grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w,
    const int nH,
    const int nW){
    // start
    int index = threadIdx.x;
    int offset;
    for (int i = index; i < C; i += blockDim.x) {
        offset = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * C + i; // C = blocksize
        int input_offset =
        (blockIdx.z / (nH * nW)) * H * W * C +
        (blockIdx.z % (nH * nW) / nW * window_h + blockIdx.y + shift_h + H) % H * W * C +
        (blockIdx.z % nW * window_w + blockIdx.x + shift_w + W) % W * C +
        i;
        grad_out[offset] = (T)(SWIN_WP_LDG(grad_in + input_offset));
    }
}

// input: [B, H, W, C]
// output: [B*nH*nW, window_h, window_w, C]
at::Tensor roll_and_window_partition_forward_cuda(
    at::Tensor & input,
    //at::Tensor & output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){

    int nH = H / window_h;
    int nW = W / window_w;

    dim3 grid(window_w, window_h, B * nH * nW);
    //dim3 block((C + 31) / 32 * 32);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    at::Tensor output = at::empty({B*nH*nW, window_h, window_w, C}, input.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "roll_and_window_partition_forward_cuda_kernel", ([&] {
        roll_and_window_partition_forward_cuda_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B,
            H,
            W,
            C,
            shift_h,
            shift_w,
            window_h,
            window_w,
            nH,
            nW);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


// grad_in: [B*nH*nW, window_h, window_w, C]
// grad_out: [B, H, W, C]
at::Tensor roll_and_window_partition_backward_cuda(
    at::Tensor & grad_in,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){

    int nH = H / window_h;
    int nW = W / window_w;

    dim3 grid(W, H, B);
    //dim3 block((C + 31) / 32 * 32);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    at::Tensor grad_out = at::empty({B, H, W, C}, grad_in.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        grad_in.scalar_type(), "roll_and_window_partition_backward_cuda_kernel", ([&] {
        roll_and_window_partition_backward_cuda_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_in.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            B,
            H,
            W,
            C,
            shift_h,
            shift_w,
            window_h,
            window_w,
            nH,
            nW);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_out;
}


// input: [B*nH*nW, window_h, window_w, C]
// output: [B, H, W, C]
at::Tensor window_merge_and_roll_forward_cuda(
    at::Tensor & input,
    //at::Tensor & output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){

    int nH = H / window_h;
    int nW = W / window_w;

    dim3 grid(W, H, B);
    //dim3 block((C + 31) / 32 * 32);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    at::Tensor output = at::empty({B, H, W, C}, input.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "window_merge_and_roll_forward_cuda_kernel", ([&] {
        window_merge_and_roll_forward_cuda_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B,
            H,
            W,
            C,
            shift_h,
            shift_w,
            window_h,
            window_w,
            nH,
            nW);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


// grad_in: [B, H, W, C]
// grad_out: [B*nH*nW, window_h, window_w, C]
at::Tensor window_merge_and_roll_backward_cuda(
    at::Tensor & grad_in,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){

    int nH = H / window_h;
    int nW = W / window_w;

    dim3 grid(window_w, window_h, B * nH * nW);
    //dim3 block((C + 31) / 32 * 32);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    at::Tensor grad_out = at::empty({B*nH*nW, window_h, window_w, C}, grad_in.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        grad_in.scalar_type(), "window_merge_and_roll_backward_cuda_kernel", ([&] {
        window_merge_and_roll_backward_cuda_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_in.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            B,
            H,
            W,
            C,
            shift_h,
            shift_w,
            window_h,
            window_w,
            nH,
            nW);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_out;
}
