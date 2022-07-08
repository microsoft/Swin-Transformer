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


#define FINAL_MASK 0xffffffff
#define ROWS_PER_BLOCK 8


/*******************  warpReduceMax  ***********************/

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

/*******************  blockReduceMax  ***********************/

/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;  // in-warp idx
    int wid = threadIdx.x >> 5;     // warp idx

    val = warpReduceMax(val);  // get maxx in each warp

    if (lane == 0)  // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)-1e20f;
    val = warpReduceMax<T>(val);

    return val;
}

/*******************  warpReduceSum  ***********************/

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

/*******************  blockReduceSum  ***********************/

template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = warpReduceSum(val);
    return val;
}


 
template <typename T>
__global__ void softmax_fwd_cuda_kernel(
    T* qk_buf, 
    const T* relative_pos_bias,
    const T* attn_mask,
    //const T* random_tensor,
    //T* softmax_result, 
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len)
{
    bool qual = threadIdx.x < window_len;
    for (int window_id = blockIdx.x; window_id < window_len; window_id += gridDim.x) {
        float tmp = -1e20f;
        __shared__ float s_mean, s_max;
        int qk_offset;
        if (qual) {
            const int offset_in_window = window_id * window_len + threadIdx.x;
            qk_offset = (blockIdx.z * gridDim.y + blockIdx.y) * window_len * window_len + offset_in_window;
            const int relative_pos_bias_offset = (blockIdx.y % num_head) * window_len * window_len + offset_in_window;
            float mask_val = static_cast<float>(
                        __ldg(attn_mask + ((blockIdx.y / num_head) * window_len * window_len + offset_in_window)));
            tmp = static_cast<float>(qk_buf[qk_offset]) + mask_val
                  + static_cast<float>(__ldg(relative_pos_bias + relative_pos_bias_offset));
        }
        float max_val = blockReduceMax<float>(tmp);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float qk_tmp = qual ? __expf(tmp - s_max) : 0.0f;
        float sum_val = blockReduceSum<float>(qk_tmp);
        
        if (threadIdx.x == 0) {
            s_mean = sum_val; // + 1e-12f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();
        if (qual)
        {
            qk_buf[qk_offset] = (T)(qk_tmp * s_mean);
        }
    }
}


template <>
__global__ void softmax_fwd_cuda_kernel(
    half* qk_buf, 
    const half* relative_pos_bias,
    const half* attn_mask,
    //const half* random_tensor,
    //half* softmax_result, 
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len)
{
    bool qual = threadIdx.x < window_len;
    for (int window_id = blockIdx.x; window_id < window_len; window_id += gridDim.x) {
        float tmp = -1e20f;
        __shared__ float s_mean, s_max;
        int qk_offset;
        if (qual) {
            const int offset_in_window = window_id * window_len + threadIdx.x;
            qk_offset = (blockIdx.z * gridDim.y + blockIdx.y) * window_len * window_len + offset_in_window;
            const int relative_pos_bias_offset = (blockIdx.y % num_head) * window_len * window_len + offset_in_window;
            float mask_val = __half2float(
                        __ldg(attn_mask + ((blockIdx.y / num_head) * window_len * window_len + offset_in_window)));
            tmp = __half2float(qk_buf[qk_offset]) + mask_val
                  + __half2float(__ldg(relative_pos_bias + relative_pos_bias_offset));
        }
        float max_val = blockReduceMax<float>(tmp);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float qk_tmp = qual ? __expf(tmp - s_max) : 0.0f;
        float sum_val = blockReduceSum<float>(qk_tmp);
        
        if (threadIdx.x == 0) {
            s_mean = sum_val; //+ 1e-12f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();
        if (qual)
        {
            qk_buf[qk_offset] = __float2half(qk_tmp * s_mean);
        }
    }
}


template <typename T>
__global__ void softmax_nomask_fwd_cuda_kernel(
    T* qk_buf, 
    const T* relative_pos_bias,
    //const T* attn_mask,
    //const T* random_tensor,
    //T* softmax_result, 
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len)
{
    bool qual = threadIdx.x < window_len;
    for (int window_id = blockIdx.x; window_id < window_len; window_id += gridDim.x) {
        float tmp = -1e20f;
        __shared__ float s_mean, s_max;
        int qk_offset;
        if (qual) {
            const int offset_in_window = window_id * window_len + threadIdx.x;
            qk_offset = (blockIdx.z * gridDim.y + blockIdx.y) * window_len * window_len + offset_in_window;
            const int relative_pos_bias_offset = (blockIdx.y % num_head) * window_len * window_len + offset_in_window;
            //float mask_val =
            //    (attn_mask == nullptr) ?
            //        0.0f :
            //        static_cast<float>(
            //            __ldg(attn_mask + ((blockIdx.y / num_head) * window_len * window_len + offset_in_window)));
            //tmp = static_cast<float>(qk_buf[qk_offset]) + mask_val
            //      + static_cast<float>(__ldg(relative_pos_bias + relative_pos_bias_offset));
            tmp = static_cast<float>(qk_buf[qk_offset]) + static_cast<float>(__ldg(relative_pos_bias + relative_pos_bias_offset));
        }
        float max_val = blockReduceMax<float>(tmp);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float qk_tmp = qual ? __expf(tmp - s_max) : 0.0f;
        float sum_val = blockReduceSum<float>(qk_tmp);
        
        if (threadIdx.x == 0) {
            s_mean = sum_val; // + 1e-12f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();
        if (qual)
        {
            qk_buf[qk_offset] = (T)(qk_tmp * s_mean);
        }
    }
}



template <>
__global__ void softmax_nomask_fwd_cuda_kernel(
    half* qk_buf, 
    const half* relative_pos_bias,
    //const half* random_tensor,
    //half* softmax_result, 
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len)
{
    bool qual = threadIdx.x < window_len;
    for (int window_id = blockIdx.x; window_id < window_len; window_id += gridDim.x) {
        float tmp = -1e20f;
        __shared__ float s_mean, s_max;
        int qk_offset;
        if (qual) {
            const int offset_in_window = window_id * window_len + threadIdx.x;
            qk_offset = (blockIdx.z * gridDim.y + blockIdx.y) * window_len * window_len + offset_in_window;
            const int relative_pos_bias_offset = (blockIdx.y % num_head) * window_len * window_len + offset_in_window;
            tmp = __half2float(qk_buf[qk_offset]) + __half2float(__ldg(relative_pos_bias + relative_pos_bias_offset));
        }
        float max_val = blockReduceMax<float>(tmp);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float qk_tmp = qual ? __expf(tmp - s_max) : 0.0f;
        float sum_val = blockReduceSum<float>(qk_tmp);
        
        if (threadIdx.x == 0) {
            s_mean = sum_val; //+ 1e-12f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();
        if (qual)
        {
            qk_buf[qk_offset] = __float2half(qk_tmp * s_mean);
        }
    }
}


template <typename T>
__global__ void softmax_bwd_cuda_kernel(
    T* grad, 
    const T* softmax_result, 
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len)
{
    bool qual = threadIdx.x < window_len;
    for (int window_id = blockIdx.x; window_id < window_len; window_id += gridDim.x) {
        int grad_offset;
        float da = 0.0f;
        float sfmax = 0.0f;
        float sm1 = 0.0f;
        if (qual) {
            int offset_in_window = threadIdx.x + window_id * window_len;
            grad_offset = (blockIdx.z * gridDim.y + blockIdx.y) * window_len * window_len + offset_in_window;
            da = static_cast<float>(grad[grad_offset]);   
            sfmax = static_cast<float>(softmax_result[grad_offset]);
            sm1 = da * sfmax;
        }

        float tmp = blockReduceSum<float>(sm1);
        __shared__ float sm2;
        if (threadIdx.x == 0)
            sm2 = (float)tmp;
        __syncthreads();

        if (qual) {
            // grad_score
            tmp = sfmax * (da - sm2);
            grad[grad_offset] = (T)tmp;
        }
    }
}



template <>
__global__ void softmax_bwd_cuda_kernel(
    half* grad, 
    const half* softmax_result, 
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len)
{
    bool qual = threadIdx.x < window_len;
    for (int window_id = blockIdx.x; window_id < window_len; window_id += gridDim.x) {
        int grad_offset;
        float da = 0.0f;
        float sfmax = 0.0f;
        float sm1 = 0.0f;
        if (qual) {
            int offset_in_window = threadIdx.x + window_id * window_len;
            grad_offset = (blockIdx.z * gridDim.y + blockIdx.y) * window_len * window_len + offset_in_window;
            da = __half2float(grad[grad_offset]);  
            sfmax = __half2float(softmax_result[grad_offset]);
            sm1 = da * sfmax;
        }

        float tmp = blockReduceSum<float>(sm1);
        __shared__ float sm2;
        if (threadIdx.x == 0)
            sm2 = (float)tmp;
        __syncthreads();

        if (qual) {
            // grad_score
            tmp = sfmax * (da - sm2);
            grad[grad_offset] = __float2half(tmp);
        }
    }
}


//std::vector<at::Tensor> softmax_fwd_cuda(
void softmax_fwd_cuda(
    at::Tensor & input, 
    const at::Tensor & relative_pos_bias,
    const at::Tensor & attn_mask,
    //const at::Tensor & random_tensor,
    //at::Tensor & softmax_result,
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len) {

    dim3 grid(window_len, window_num * num_head, batch_size);
    dim3 block((window_len + 31) / 32 * 32);
    // Every block process N rows, N can be 2, 4, 8
    // gird.x = ceil(grid.x/float(N))
    grid.x = ceil(float(window_len) / float(ROWS_PER_BLOCK));

    //printf("Data type is %s", input.scalar_type());
    switch(input.scalar_type()){
        case at::ScalarType::Half:
            softmax_fwd_cuda_kernel<<<grid, block, 0>>>(
                input.data_ptr<at::Half>(),
                relative_pos_bias.data_ptr<at::Half>(),
                attn_mask.data_ptr<at::Half>(),
                //random_tensor.data_ptr<at::Half>(),
                //softmax_result.data_ptr<at::Half>(),
                batch_size,
                window_num,
                num_head,
                window_len);
            break;
        default:
            AT_DISPATCH_ALL_TYPES(input.type(), "softmax_cuda_kernel", ([&] {
            softmax_fwd_cuda_kernel<scalar_t><<<grid, block, 0>>>(
                input.data<scalar_t>(),
                relative_pos_bias.data<scalar_t>(),
                attn_mask.data<scalar_t>(),
                //random_tensor.data<scalar_t>(),
                //softmax_result.data<scalar_t>(),
                batch_size,
                window_num,
                num_head,
                window_len);
            }));
    };
}





void softmax_nomask_fwd_cuda(
    at::Tensor & input, 
    const at::Tensor & relative_pos_bias,
    //const at::Tensor & random_tensor,
    //at::Tensor & softmax_result,
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len) {

    dim3 grid(window_len, window_num * num_head, batch_size);
    dim3 block((window_len + 31) / 32 * 32);
    // Every block process N rows, N can be 2, 4, 8
    // gird.x = ceil(grid.x/float(N))
    grid.x = ceil(float(window_len) / float(ROWS_PER_BLOCK));

    //printf("Data type is %s", input.scalar_type());
    switch(input.scalar_type()){
        case at::ScalarType::Half:
            softmax_nomask_fwd_cuda_kernel<<<grid, block, 0>>>(
                input.data_ptr<at::Half>(),
                relative_pos_bias.data_ptr<at::Half>(),
                //random_tensor.data_ptr<at::Half>(),
                //softmax_result.data_ptr<at::Half>(),
                batch_size,
                window_num,
                num_head,
                window_len);
            break;
        default:
            AT_DISPATCH_ALL_TYPES(input.type(), "softmax_cuda_kernel", ([&] {
            softmax_nomask_fwd_cuda_kernel<scalar_t><<<grid, block, 0>>>(
                input.data<scalar_t>(),
                relative_pos_bias.data<scalar_t>(),
                //random_tensor.data<scalar_t>(),
                //softmax_result.data<scalar_t>(),
                batch_size,
                window_num,
                num_head,
                window_len);
            }));
    };
}


//std::vector<at::Tensor> softmax_bwd_cuda(
void softmax_bwd_cuda(
    at::Tensor & grad, 
    const at::Tensor & softmax_result,
    const int batch_size, 
    const int window_num,
    const int num_head, 
    const int window_len) {
    //Get tensor size
    dim3 grid(window_len, window_num * num_head, batch_size);
    dim3 block((window_len + 31) / 32 * 32, 1, 1);
    // Every block process N rows, N can be 2, 4, 8
    // gird.x = ceil(grid.x/float(N))
    grid.x = ceil(float(window_len) / float(ROWS_PER_BLOCK));

    switch(grad.scalar_type()){
        case at::ScalarType::Half:
            softmax_bwd_cuda_kernel<<<grid, block, 0>>>(
                    grad.data_ptr<at::Half>(),
                    softmax_result.data_ptr<at::Half>(),
                    batch_size,
                    window_num,
                    num_head,
                    window_len);
                break;
        default:
            AT_DISPATCH_ALL_TYPES(grad.type(), "softmax_cuda_kernel", ([&] {
                softmax_bwd_cuda_kernel<scalar_t><<<grid, block, 0>>>(
                    grad.data<scalar_t>(),
                    softmax_result.data<scalar_t>(),
                    batch_size,
                    window_num,
                    num_head,
                    window_len);
            }));
    };
}