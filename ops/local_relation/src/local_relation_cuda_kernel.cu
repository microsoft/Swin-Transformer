/*!
 * Copyright (c) 2019 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file local_relation_cuda_kernel.cu
 * \brief
 * \author Han Hu
 * \modified by Jiarui Xu, Ze Liu
*/

#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <math.h>
#include <float.h>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N)
{
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

template <typename scalar_t>
__global__ void similarity_compute_forward_kernel(const int n,
                                      const scalar_t* key, 
                                      const scalar_t* query, 
                                      const scalar_t* pos_weight,
                                      const int batch_size, 
                                      const int key_channels,
                                      const int query_channels,
                                      const int height, 
                                      const int width,
                                      const int kernel_height,
                                      const int kernel_width,
                                      const int num_group,
                                      const scalar_t* scale_ptr,
                                      const scalar_t* no_define_value_ptr,
                                      const int dilate,
                                      const int stride,
                                      const int in_height,
                                      const int in_width,
                                      const int sim_method,
                                      scalar_t* output) {
  // n = batch_size * num_group * kernel_height * kernel_width * height * width
  const scalar_t scale = scale_ptr[0];
  const scalar_t no_define_value = no_define_value_ptr[0];
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int kw = h / height;
    h = h % height;
    int kh = kw / kernel_width;
    kw = kw % kernel_width;
    int g = kh / kernel_height;
    kh = kh % kernel_height;
    const int b = g / num_group;
    g = g % num_group;

    scalar_t sum_sim = 0;
    const int half_kh = kernel_height / 2;
    const int half_kw = kernel_width / 2;
    if (sim_method >= 0){
      const int key_per_group = query_channels / num_group;
      
      const int spatial_dim = height * width;
      const int in_spatial_dim = in_height * in_width;
      int query_inds = 0;
      if (sim_method != 1) {
        query_inds = ((b * num_group + g) * key_per_group * height + h) * width + w;
      }
      const int key_saliency_group = key_channels - query_channels;
  
      if (w * stride + dilate * (kw - half_kw) >= 0 && w * stride + dilate * (kw - half_kw) < in_width && h * stride + dilate * (kh - half_kh) >= 0 && h * stride + dilate * (kh - half_kh) < in_height) {
        int key_inds = ((b * key_channels + g * key_per_group) * in_height + h * stride + dilate * (kh - half_kh)) * in_width + w * stride + dilate * (kw - half_kw);
        
        for (int i = 0; i < key_per_group; ++i) {
          if (sim_method == 0) {
            sum_sim += query[query_inds + i * spatial_dim] * key[key_inds + i * in_spatial_dim] * scale;
          }
          else if (sim_method == 1) {
            sum_sim += key[key_inds + i * in_spatial_dim] * scale;
          }
          else if (sim_method == 2) {
            sum_sim += -abs(query[query_inds + i * spatial_dim] - key[key_inds + i * in_spatial_dim]) * scale;
          }
          else if (sim_method == 3) {
            scalar_t query_val = query[query_inds + i * spatial_dim];
            scalar_t key_val = key[key_inds + i * in_spatial_dim];
            sum_sim += -abs(query_val - key_val) / (abs(query_val) + abs(key_val) + scalar_t(1.0)) * scale;
          }
          else if (sim_method == 4) {
            scalar_t query_val = query[query_inds + i * spatial_dim];
            scalar_t key_val = key[key_inds + i * in_spatial_dim];
            sum_sim += -(query_val - key_val) * (query_val - key_val) / (abs(query_val * key_val) + scalar_t(1.0)) * scale;
          }
          else if (sim_method == 5) {
            scalar_t query_val = query[query_inds + i * spatial_dim];
            scalar_t key_val = key[key_inds + i * in_spatial_dim];
            sum_sim += -(query_val - key_val) * (query_val - key_val) * scale;
          }
          if (key_saliency_group > 0) {
              int key_sal_inds = (b * key_channels + query_channels + int(g * key_saliency_group) / num_group) * in_spatial_dim 
                          + (h * stride + dilate * (kh - half_kh)) * in_width + w * stride + dilate * (kw - half_kw);
              sum_sim += key[key_sal_inds];
          }
        }
      }
      else{
        sum_sim = no_define_value;
      }
    }

    if (w * stride + dilate * (kw - half_kw) >= 0 && w * stride + dilate * (kw - half_kw) < in_width && h * stride + dilate * (kh - half_kh) >= 0 && h * stride + dilate * (kh - half_kh) < in_height) {
    }
    else {
      sum_sim = no_define_value;
    }
    int pos_inds = (g * kernel_height + kh) * kernel_width + kw;
    sum_sim += pos_weight[pos_inds];

    output[index] = sum_sim;
  }
}

template <typename scalar_t>
__global__ void similarity_compute_backward_kernel(const int n,
                                      const scalar_t* key, 
                                      const scalar_t* query,
                                      const scalar_t* output_grad,
                                      const int batch_size, 
                                      const int key_channels,
                                      const int query_channels,
                                      const int height, 
                                      const int width,
                                      const int kernel_height,
                                      const int kernel_width,
                                      const int num_group,
                                      const int key_per_group,
                                      const scalar_t* scale_ptr,
                                      const int dilate,
                                      const int stride,
                                      const int in_height,
                                      const int in_width,
                                      const int sim_method,
                                      scalar_t* key_grad,
                                      scalar_t* query_grad) {
  const scalar_t scale = scale_ptr[0];
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int kpg = h / height;
    h = h % height;
    int g = kpg / key_per_group;
    kpg = kpg % key_per_group;
    const int b = g / num_group;
    g = g % num_group;

    const int half_kh = kernel_height / 2;
    const int half_kw = kernel_width / 2;
    
    const int spatial_dim = height * width;
    const int key_saliency_group = key_channels - query_channels;

    int output_inds = ((b * num_group + g) * kernel_height * kernel_width * height + h) * width + w; 
    scalar_t sum_query_grad = 0;

    int key_inds = ((b * key_channels + g * key_per_group + kpg) * in_height + h * stride) * in_width + w * stride; 
      for (int kh = 0; kh < kernel_height; ++kh) {
        for (int kw = 0; kw < kernel_width; ++kw) {
          if (w * stride + dilate * (kw - half_kw) >= 0 && w * stride + dilate * (kw - half_kw) < in_width 
            && h * stride + dilate * (kh - half_kh) >= 0 && h * stride + dilate * (kh - half_kh) < in_height) {
            scalar_t c_out_grad = output_grad[output_inds + (kh * kernel_width + kw) * spatial_dim];
            if (sim_method == 0) {
              sum_query_grad += c_out_grad 
                    * key[key_inds + dilate * (kh - half_kh) * in_width + dilate * (kw - half_kw)];
            }
            else if (sim_method == 2) {
              scalar_t key_val = key[key_inds + dilate * (kh - half_kh) * in_width + dilate * (kw - half_kw)];
              scalar_t query_val = query[index];
              if (key_val > query_val) {
                sum_query_grad += c_out_grad;
              }
              else if (key_val < query_val) {
                sum_query_grad += -c_out_grad;
              }
            }
            else if (sim_method == 3) {
              scalar_t key_val = key[key_inds + dilate * (kh - half_kh) * in_width + dilate * (kw - half_kw)];
              scalar_t query_val = query[index];
              if (key_val > query_val) {
                sum_query_grad += c_out_grad / (abs(key_val) + abs(query_val) + scalar_t(1.0)); 
              }
              else if (key_val < query_val) {
                sum_query_grad += -c_out_grad / (abs(key_val) + abs(query_val) + scalar_t(1.0)); 
              }

              if (query_val > 0) {
                sum_query_grad += c_out_grad * abs(key_val - query_val) / ((abs(key_val) +abs(query_val) + scalar_t(1.0)) * (abs(key_val) +abs(query_val) + scalar_t(1.0))); 
              }
              else if (query_val < 0) {
                sum_query_grad += -c_out_grad * abs(key_val - query_val) / ((abs(key_val) +abs(query_val) + scalar_t(1.0)) * (abs(key_val) +abs(query_val) + scalar_t(1.0))); 
              }
            }
            else if (sim_method == 4) {
              scalar_t key_val = key[key_inds + dilate * (kh - half_kh) * in_width + dilate * (kw - half_kw)];
              scalar_t query_val = query[index];
              sum_query_grad += 2 * c_out_grad * (key_val - query_val) / (abs(key_val * query_val) + scalar_t(1.0));
              
              if (key_val * query_val > 0) {
                sum_query_grad += c_out_grad * key_val * (key_val - query_val) * (key_val - query_val) / ((abs(key_val * query_val) + scalar_t(1.0)) * (abs(key_val * query_val) + scalar_t(1.0))); 
              }
              else if(key_val * query_val < 0) {
                sum_query_grad += -c_out_grad * key_val * (key_val - query_val) * (key_val - query_val) / ((abs(key_val * query_val) + scalar_t(1.0)) * (abs(key_val * query_val) + scalar_t(1.0))); 
              }
            }
            else if (sim_method == 5) {
              scalar_t key_val = key[key_inds + dilate * (kh - half_kh) * in_width + dilate * (kw - half_kw)];
              scalar_t query_val = query[index];
              sum_query_grad += 2 * c_out_grad * (key_val - query_val);
            }
          }
        }
      }
      sum_query_grad *= scale;
      query_grad[index] += sum_query_grad;

    scalar_t sum_key_grad = 0;
    int start_kh = -half_kh / stride;
    int end_kh = half_kh / stride;
    int start_kw = -half_kw / stride;
    int end_kw = half_kw / stride;
    int key_sal_inds = (b * key_channels + query_channels + int(g * key_saliency_group) / num_group) * in_height * in_width
                        + h * stride * in_width + w * stride;

    scalar_t sum_key_sal_grad = 0;
    for (int kh = start_kh; kh <= end_kh; ++kh) {
      for (int kw = start_kw; kw <= end_kw; ++kw) {
        if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
          int spatial_offset = dilate * kh * width + dilate * kw;
          scalar_t c_out_grad = output_grad[output_inds + ((half_kh - kh * stride) * kernel_width + half_kw - kw * stride) * spatial_dim + spatial_offset];
          scalar_t query_val = query[index + spatial_offset];
          if (sim_method == 0) {
            sum_key_grad += c_out_grad 
                     * query_val * scalar_t(scale);
          }
          else if (sim_method == 1) {
            sum_key_grad += c_out_grad * scalar_t(scale);
          }
          else if (sim_method == 2) {
            scalar_t key_val = key[key_inds];
            if (key_val > query_val) {
              sum_key_grad += scalar_t(-scale) * c_out_grad;
            }
            else if (key_val < query_val) {
              sum_key_grad += scalar_t(scale) * c_out_grad;
            }
          }
          else if (sim_method == 3) {
            scalar_t key_val = key[key_inds];
            if (key_val > query_val) {
              sum_key_grad += -scalar_t(scale) * c_out_grad / (abs(key_val) + abs(query_val) + scalar_t(1.0));
            }
            else if (key_val < query_val) {
              sum_key_grad += scalar_t(scale) * c_out_grad / (abs(key_val) + abs(query_val) + scalar_t(1.0));
            }

            if (key_val > 0) {
              sum_key_grad += c_out_grad * scalar_t(scale) * abs(key_val - query_val) 
                            / ((abs(key_val) +abs(query_val) + scalar_t(1.0)) 
                            * (abs(key_val) +abs(query_val) + scalar_t(1.0))); 
            }
            else if (key_val < 0){
              sum_key_grad += -c_out_grad * scalar_t(scale) * abs(key_val - query_val) 
                            / ((abs(key_val) +abs(query_val) + scalar_t(1.0)) 
                            * (abs(key_val) +abs(query_val) + scalar_t(1.0))); 
            }
          }
          else if (sim_method == 4) {
            scalar_t key_val = key[key_inds];
            sum_key_grad += 2 * scalar_t(scale) * c_out_grad * (query_val - key_val) / (abs(key_val * query_val) + scalar_t(1.0));
              
            if (key_val * query_val > 0) {
              sum_key_grad += scalar_t(scale) * c_out_grad * query_val * (key_val - query_val) * (key_val - query_val) / ((abs(key_val * query_val) + scalar_t(1.0)) * (abs(key_val * query_val) + scalar_t(1.0))); 
            }
            else if(key_val * query_val < 0) {
              sum_key_grad += -scalar_t(scale) * c_out_grad * query_val * (key_val - query_val) * (key_val - query_val) / ((abs(key_val * query_val) + scalar_t(1.0)) * (abs(key_val * query_val) + scalar_t(1.0))); 
            }
          }
          else if (sim_method == 5) {
            scalar_t key_val = key[key_inds];
            sum_key_grad += scalar_t(scale) * c_out_grad * (query_val - key_val) * 2; 
          }

          if (key_saliency_group > 0) {
            sum_key_sal_grad += c_out_grad; 
          }
        }
      }
    }
    key_grad[key_inds] += sum_key_grad;
    if (key_saliency_group > 0) {
      atomicAdd(key_grad + key_sal_inds, sum_key_sal_grad);
    }

    if (stride == 2){
      if (h * stride + 1 < in_height) {
        sum_key_grad = 0;
        sum_key_sal_grad = 0;
        start_kh = (1 - half_kh) / stride;
        end_kh = (half_kh + 1) / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              scalar_t c_out_grad = output_grad[output_inds + ((half_kh - kh * stride + 1) * kernel_width + half_kw - kw * stride) * spatial_dim + spatial_offset];
              scalar_t query_val = query[index + spatial_offset];
              if (sim_method == 0) {
                sum_key_grad += c_out_grad 
                         * query_val * scalar_t(scale);
              }
              else if (sim_method == 1) {
                sum_key_grad += c_out_grad * scalar_t(scale);
              }
              else if (sim_method == 2) {
                scalar_t key_val = key[key_inds + in_width];
                if (key_val > query_val) {
                  sum_key_grad += scalar_t(-scale) * c_out_grad;
                }
                else if (key_val < query_val) {
                  sum_key_grad += scalar_t(scale) * c_out_grad;
                }
                else {
                  sum_key_grad += scalar_t(0.0);
                }
              }
              else if (sim_method == 3) {
                scalar_t key_val = key[key_inds + in_width];
                if (key_val > query_val) {
                  sum_key_grad += -scalar_t(scale) * c_out_grad / (abs(key_val) + abs(query_val) + scalar_t(1.0));
                }
                else if (key_val < query_val) {
                  sum_key_grad += scalar_t(scale) * c_out_grad / (abs(key_val) + abs(query_val) + scalar_t(1.0));
                }
    
                if (key_val > 0) {
                  sum_key_grad += scalar_t(scale) * c_out_grad * abs(key_val - query_val) 
                                / ((abs(key_val) +abs(query_val) + scalar_t(1.0)) 
                                * (abs(key_val) +abs(query_val) + scalar_t(1.0))); 
                }
                else if (key_val < 0){
                  sum_key_grad += -scalar_t(scale) * c_out_grad * abs(key_val - query_val) 
                                / ((abs(key_val) +abs(query_val) + scalar_t(1.0)) 
                                * (abs(key_val) +abs(query_val) + scalar_t(1.0))); 
                }
              }
              else if (sim_method == 4) {
                scalar_t key_val = key[key_inds + in_width];
                sum_key_grad += 2 * scalar_t(scale) * c_out_grad * (query_val - key_val) / (abs(key_val * query_val) + scalar_t(1.0));
              
                if (key_val * query_val > 0) {
                  sum_key_grad += scalar_t(scale) * c_out_grad * query_val * (key_val - query_val) * (key_val - query_val) / ((abs(key_val * query_val) + scalar_t(1.0)) * (abs(key_val * query_val) + scalar_t(1.0))); 
                }
                else if(key_val * query_val < 0) {
                  sum_key_grad += -scalar_t(scale) * c_out_grad * query_val * (key_val - query_val) * (key_val - query_val) / ((abs(key_val * query_val) + scalar_t(1.0)) * (abs(key_val * query_val) + scalar_t(1.0))); 
                }
              }
              else if (sim_method == 5) {
                scalar_t key_val = key[key_inds + in_width];
                sum_key_grad += scalar_t(scale) * c_out_grad * (query_val - key_val) * 2; 
              }

              if (key_saliency_group > 0) {
                sum_key_sal_grad += c_out_grad; 
              }
            }
          }
        }
        key_grad[key_inds + in_width] += sum_key_grad;
        if (key_saliency_group > 0) {
          atomicAdd(key_grad + key_sal_inds + in_width, sum_key_sal_grad);
        }
      }
      if (w * stride + 1 < in_width) {
        sum_key_grad = 0;
        sum_key_sal_grad = 0;
        start_kh = -half_kh / stride;
        end_kh = half_kh / stride;
        start_kw = (1 - half_kw) / stride;
        end_kw = (half_kw + 1) / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              scalar_t c_out_grad = output_grad[output_inds + ((half_kh - kh * stride) * kernel_width + half_kw - kw * stride + 1) * spatial_dim + spatial_offset];
              scalar_t query_val = query[index + spatial_offset];
              if (sim_method == 0) {
                sum_key_grad += c_out_grad 
                         * query_val * scalar_t(scale);
              }
              else if (sim_method == 1) {
                sum_key_grad += c_out_grad * scalar_t(scale);
              }
              else if (sim_method == 2) {
                scalar_t key_val = key[key_inds + 1];
                if (key_val > query_val) {
                  sum_key_grad += scalar_t(-scale) * c_out_grad;
                }
                else if (key_val < query_val) {
                  sum_key_grad += scalar_t(scale) * c_out_grad;
                }
                else {
                  sum_key_grad += scalar_t(0.0);
                }
              }
              else if (sim_method == 3) {
                scalar_t key_val = key[key_inds + 1];
                if (key_val > query_val) {
                  sum_key_grad += -scalar_t(scale) * c_out_grad / (abs(key_val) + abs(query_val) + scalar_t(1.0));
                }
                else if (key_val < query_val) {
                  sum_key_grad += scalar_t(scale) * c_out_grad / (abs(key_val) + abs(query_val) + scalar_t(1.0));
                }
    
                if (key_val > 0) {
                  sum_key_grad += scalar_t(scale) * c_out_grad * abs(key_val - query_val) 
                                / ((abs(key_val) +abs(query_val) + scalar_t(1.0)) 
                                * (abs(key_val) +abs(query_val) + scalar_t(1.0))); 
                }
                else if (key_val < 0){
                  sum_key_grad += -scalar_t(scale) * c_out_grad * abs(key_val - query_val) 
                                / ((abs(key_val) +abs(query_val) + scalar_t(1.0)) 
                                * (abs(key_val) +abs(query_val) + scalar_t(1.0))); 
                }
              }
              else if (sim_method == 4) {
                scalar_t key_val = key[key_inds + 1];
                sum_key_grad += 2 * scalar_t(scale) * c_out_grad * (query_val - key_val) / (abs(key_val * query_val) + scalar_t(1.0));
              
                if (key_val * query_val > 0) {
                  sum_key_grad += scalar_t(scale) * c_out_grad * query_val * (key_val - query_val) * (key_val - query_val) / ((abs(key_val * query_val) + scalar_t(1.0)) * (abs(key_val * query_val) + scalar_t(1.0))); 
                }
                else if(key_val * query_val < 0) {
                  sum_key_grad += -scalar_t(scale) * c_out_grad * query_val * (key_val - query_val) * (key_val - query_val) / ((abs(key_val * query_val) + scalar_t(1.0)) * (abs(key_val * query_val) + scalar_t(1.0))); 
                }
              }
              else if (sim_method == 5) {
                scalar_t key_val = key[key_inds + 1];
                sum_key_grad += scalar_t(scale) * c_out_grad * (query_val - key_val) * 2; 
              }

              if (key_saliency_group > 0) {
                sum_key_sal_grad += c_out_grad; 
              }
            }
          }
        }
        key_grad[key_inds + 1] += sum_key_grad;
        if (key_saliency_group > 0) {
          atomicAdd(key_grad + key_sal_inds + 1, sum_key_sal_grad);
        }
      }
      if (h * stride + 1 < in_height && w * stride + 1 < in_width) {
        sum_key_grad = 0;
        sum_key_sal_grad = 0;
        start_kh = (1 - half_kh) / stride;
        end_kh = (half_kh + 1) / stride;
        start_kw = (1 - half_kw) / stride;
        end_kw = (half_kw + 1) / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              scalar_t c_out_grad = output_grad[output_inds + ((half_kh - kh * stride + 1) * kernel_width + half_kw - kw * stride + 1) * spatial_dim + spatial_offset];
              scalar_t query_val = query[index + spatial_offset];
              if (sim_method == 0) {
                sum_key_grad += c_out_grad 
                         * query_val * scalar_t(scale);
              }
              else if (sim_method == 1) {
                sum_key_grad += c_out_grad * scalar_t(scale);
              }
              else if (sim_method == 2) {
                scalar_t key_val = key[key_inds + in_width + 1];
                if (key_val > query_val) {
                  sum_key_grad += scalar_t(-scale) * c_out_grad;
                }
                else if (key_val < query_val) {
                  sum_key_grad += scalar_t(scale) * c_out_grad;
                }
              }
              else if (sim_method == 3) {
                scalar_t key_val = key[key_inds + in_width + 1];
                if (key_val > query_val) {
                  sum_key_grad += -scalar_t(scale) * c_out_grad / (abs(key_val) + abs(query_val) + scalar_t(1.0));
                }
                else if (key_val < query_val) {
                  sum_key_grad += scalar_t(scale) * c_out_grad / (abs(key_val) + abs(query_val) + scalar_t(1.0));
                }
    
                if (key_val > 0) {
                  sum_key_grad += scalar_t(scale) * c_out_grad * abs(key_val - query_val) 
                                / ((abs(key_val) +abs(query_val) + scalar_t(1.0)) 
                                * (abs(key_val) +abs(query_val) + scalar_t(1.0))); 
                }
                else if (key_val < 0){
                  sum_key_grad += -scalar_t(scale) * c_out_grad * abs(key_val - query_val) 
                                / ((abs(key_val) +abs(query_val) + scalar_t(1.0)) 
                                * (abs(key_val) +abs(query_val) + scalar_t(1.0))); 
                }
              }
              else if (sim_method == 4) {
                scalar_t key_val = key[key_inds + in_width + 1];
                sum_key_grad += 2 * scalar_t(scale) * c_out_grad * (query_val - key_val) / (abs(key_val * query_val) + scalar_t(1.0));
              
                if (key_val * query_val > 0) {
                  sum_key_grad += scalar_t(scale) * c_out_grad * query_val * (key_val - query_val) * (key_val - query_val) / ((abs(key_val * query_val) + scalar_t(1.0)) * (abs(key_val * query_val) + scalar_t(1.0))); 
                }
                else if(key_val * query_val < 0) {
                  sum_key_grad += -scalar_t(scale) * c_out_grad * query_val * (key_val - query_val) * (key_val - query_val) / ((abs(key_val * query_val) + scalar_t(1.0)) * (abs(key_val * query_val) + scalar_t(1.0))); 
                }
              }
              else if (sim_method == 5) {
                scalar_t key_val = key[key_inds + in_width + 1];
                sum_key_grad += scalar_t(scale) * c_out_grad * (query_val - key_val) * 2; 
              }


              if (key_saliency_group > 0) {
                sum_key_sal_grad += c_out_grad; 
              }
            }
          }
        }
        key_grad[key_inds + in_width + 1] += sum_key_grad;
        if (key_saliency_group > 0) {
          atomicAdd(key_grad + key_sal_inds + in_width + 1, sum_key_sal_grad);
        }
      }
    }
  }
}

/*
# [batch_size, num_group, 49, height, width]
app_geo_sim = mx.sym.softmax(app_geo_sim, axis=2)
# [batch_size, num_group, 1, 49, height, width]
app_geo_sim = mx.sym.expand_dims(app_geo_sim, axis=2)
output_value = mx.sym.reshape(mx.sym.sum(mx.sym.broadcast_mul(app_geo_sim, warp_value_data_reshape), axis=3), shape=(0, -3, -2))
*/
// value: [batch_size, value_channels, height, width]
// softmax_data: [batch_size, num_group * kernel_height * kernel_width, height, width]
// num_group:
// output: [batch_size, value_channels, height, width]

template <typename scalar_t>
__global__ void aggregation_forward_kernel(const int n,
                                      const scalar_t* value, 
                                      const scalar_t* softmax_data,
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
                                      scalar_t* output) {
  // n = batch_size * value_channels * height * width
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int c = h / height;
    h = h % height;
    const int b = c / value_channels;
    c = c % value_channels;

    const int value_per_group = value_channels / num_group;

    const int g = c / value_per_group;
    const int g_in_group = c % value_per_group;

    const int half_kh = kernel_height / 2;
    const int half_kw = kernel_width / 2;
    
    const int spatial_dim = height * width;
    scalar_t sum_val = 0;

    int value_inds = (((b * num_group + g) * value_per_group + g_in_group) * in_height + h * stride) * in_width + w * stride;
    int softmax_inds = ((b * num_group + g) * kernel_height * kernel_width * height + h) * width + w;
    for (int kh = 0; kh < kernel_height; ++kh) {
      for (int kw = 0; kw < kernel_width; ++kw) {
        if (w * stride + dilate * (kw - half_kw) >= 0 && w * stride + dilate * (kw - half_kw) < in_width 
            && h * stride + dilate * (kh - half_kh) >= 0 && h * stride + dilate * (kh - half_kh) < in_height) {
          sum_val += value[value_inds + dilate * (kh - half_kh) * in_width + dilate * (kw - half_kw)] * softmax_data[softmax_inds + kh * kernel_width * spatial_dim + kw * spatial_dim];
           //if ((value_inds) == 10001) {
           //     printf("b: %d, g: %d, h: %d, w: %d, softmax_inds: %d, value_inds: %d, sum_val: %.4f, k:%d, w:%d, softmax: %.4f, val: %.4f\n", 
           //         b, g, h, w, softmax_inds, value_inds, sum_val, kh, kw,
           //         softmax_data[softmax_inds + kh * kernel_width * spatial_dim + kw * spatial_dim],
           //         value[value_inds + dilate * (kh - half_kh) * width + dilate * (kw - half_kw)]);
           //   }
        }
      }
    }
    //if (value_inds % 10000 == 1) {
    //  printf("b: %d, g: %d, h: %d, w: %d, softmax_inds: %d, value_inds: %d, sum_val: %.4f\n", 
    //                b, g, h, w, softmax_inds, value_inds, sum_val);
    //}
    output[index] = sum_val;
  }
}

template <typename scalar_t>
__global__ void aggregation_value_backward_kernel(const int n,
                                      const scalar_t* softmax_data,
                                      const scalar_t* output_grad,
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
                                      scalar_t* value_grad) {
  // n = batch_size * value_channels * height * width
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int c = h / height;
    h = h % height;
    const int b = c / value_channels;
    c = c % value_channels;

    const int value_per_group = value_channels / num_group;

    const int g = c / value_per_group;
    const int g_in_group = c % value_per_group;

    const int half_kh = kernel_height / 2;
    const int half_kw = kernel_width / 2;
    
    const int spatial_dim = height * width;
    scalar_t sum_val = 0;

    int value_inds = (((b * num_group + g) * value_per_group + g_in_group) * in_height + h * stride) * in_width + w * stride;
    int softmax_inds = ((b * num_group + g) * kernel_height * kernel_width * height + h) * width + w;

    int start_kh = -half_kh / stride;
    int end_kh = half_kh / stride;
    int start_kw = -half_kw / stride;
    int end_kw = half_kw / stride;
    for (int kh = start_kh; kh <= end_kh; ++kh) {
      for (int kw = start_kw; kw <= end_kw; ++kw) {
        if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
          int spatial_offset = dilate * kh * width + dilate * kw;
          sum_val += output_grad[index + spatial_offset] 
                  * softmax_data[softmax_inds + spatial_offset + ((half_kh - kh * stride) * kernel_width + half_kw - kw * stride) * spatial_dim];
        }
      }
    }
    value_grad[value_inds] += sum_val;

    if (stride == 2){
      if (h * stride + 1 < in_height) {
        sum_val = 0;
        start_kh = (1 - half_kh) / stride;
        end_kh = (half_kh + 1) / stride;
        start_kw = -half_kw / stride;
        end_kw = half_kw / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              sum_val += output_grad[index + spatial_offset] 
                  * softmax_data[softmax_inds + spatial_offset + ((half_kh - kh * stride + 1) * kernel_width + half_kw - kw * stride) * spatial_dim];
            }
          }
        }
        value_grad[value_inds + in_width] += sum_val;
      }
      if (w * stride + 1 < in_width) {
        sum_val = 0;
        start_kh = -half_kh / stride;
        end_kh = half_kh / stride;
        start_kw = (1 - half_kw) / stride;
        end_kw = (half_kw + 1) / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              sum_val += output_grad[index + spatial_offset] 
                  * softmax_data[softmax_inds + spatial_offset + ((half_kh - kh * stride) * kernel_width + half_kw - kw * stride + 1) * spatial_dim];
            }
          }
        }
        value_grad[value_inds + 1] += sum_val;
      }
      if (h * stride + 1 < in_height && w * stride + 1 < in_width) {
        sum_val = 0;
        start_kh = (1 - half_kh) / stride;
        end_kh = (half_kh + 1) / stride;
        start_kw = (1 - half_kw) / stride;
        end_kw = (half_kw + 1) / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              sum_val += output_grad[index + spatial_offset] 
                  * softmax_data[softmax_inds + spatial_offset + ((half_kh - kh * stride + 1) * kernel_width + half_kw - kw * stride + 1) * spatial_dim];
            }
          }
        }
        value_grad[value_inds + in_width + 1] += sum_val;
      }
    }
  }
}

template <typename scalar_t>
__global__ void aggregation_softmax_backward_kernel(const int n,
                                      const scalar_t* value,
                                      const scalar_t* output_grad,
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
                                      scalar_t* softmax_grad) {
  // n = batch_size * num_group * kernel_height * kernel_width * height * width
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int kw = h / height;
    h = h % height;
    int kh = kw / kernel_width;
    kw = kw % kernel_width;
    int g = kh / kernel_height;
    kh = kh % kernel_height;
    const int b = g / num_group;
    g = g % num_group;
    
    const int half_kh = kernel_height / 2;
    const int half_kw = kernel_width / 2;

    const int value_per_group = value_channels / num_group;
    
    const int spatial_dim = height * width;
    const int in_spatial_dim = in_height * in_width;
    scalar_t sum_val = 0;

    int value_inds = ((b * num_group + g) * value_per_group * in_height + h * stride) * in_width + w * stride;
    int output_inds = ((b * num_group + g) * value_per_group * height + h) * width + w;
    
    if (w * stride + dilate * (kw - half_kw) >= 0 && w * stride + dilate * (kw - half_kw) < in_width && h * stride + dilate * (kh - half_kh) >= 0 && h * stride + dilate * (kh - half_kh) < in_height) {
      for (int iv = 0; iv < value_per_group; ++iv) {
        sum_val += output_grad[output_inds + iv * spatial_dim] * value[value_inds + iv * in_spatial_dim + dilate * (kh - half_kh) * in_width + dilate * (kw - half_kw)];
      }
    }
    softmax_grad[index] = sum_val;
  }
}

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
	const int query_offset) 
{

	const int num_kernels = batch_size * num_group * kernel_width * kernel_height * height * width;

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		key.type(), "similarity_compute_forward_gpu", ([&] {
			const scalar_t *key_ptr = key.data_ptr<scalar_t>() + key_offset;
			const scalar_t *query_ptr = query.data_ptr<scalar_t>() + query_offset;
			const scalar_t *pos_weight_ptr = pos_weight.data_ptr<scalar_t>();
			scalar_t *output_ptr = output.data_ptr<scalar_t>();
      const scalar_t *scale_ptr = scale.data_ptr<scalar_t>();
			const scalar_t *no_define_value_ptr = no_define_value.data_ptr<scalar_t>();

			similarity_compute_forward_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
				num_kernels, key_ptr, query_ptr, pos_weight_ptr, 
				batch_size, key_channels, query_channels, height, width,
				kernel_height, kernel_width, num_group,
				scale_ptr, no_define_value_ptr, dilate, stride, in_height, in_width, sim_method, output_ptr);
		}));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("error in similarity_compute_forward: %s\n", cudaGetErrorString(err));
	}
}

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
  const int query_grad_offset) 
{
	const int num_kernels = batch_size * query_channels * height * width;

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		key.type(), "similarity_compute_backward_gpu", ([&] {
      // fixbug: add offset to key and query
			const scalar_t *key_ptr = key.data_ptr<scalar_t>() + key_grad_offset;
			const scalar_t *query_ptr = query.data_ptr<scalar_t>() + query_grad_offset;
			const scalar_t *output_grad_ptr = output_grad.data_ptr<scalar_t>();
			scalar_t *key_grad_ptr = key_grad.data_ptr<scalar_t>() + key_grad_offset;
			scalar_t *query_grad_ptr = query_grad.data_ptr<scalar_t>() + query_grad_offset;
			const scalar_t *scale_ptr = scale.data_ptr<scalar_t>();

			similarity_compute_backward_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
				num_kernels, key_ptr, query_ptr, output_grad_ptr, batch_size,
				key_channels, query_channels, height, width,
				kernel_height, kernel_width, num_group,
				key_per_group, scale_ptr, dilate, stride, in_height, in_width,
				sim_method, key_grad_ptr, query_grad_ptr);
		}));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("error in similarity_compute_backward: %s\n", cudaGetErrorString(err));
	}

}

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
	const int output_offset) 
{
	const int num_kernels = batch_size * value_channels * height * width;
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		value.type(), "aggregation_forward_gpu", ([&] {
			const scalar_t *value_ptr = value.data_ptr<scalar_t>() + value_offset;
			const scalar_t *softmax_data_ptr = softmax_data.data_ptr<scalar_t>();
			scalar_t *output_ptr = output.data_ptr<scalar_t>() + output_offset;

			aggregation_forward_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
				num_kernels, value_ptr, softmax_data_ptr,  
				batch_size, value_channels, height, width,
				kernel_height, kernel_width, num_group,
				dilate, stride, in_height, in_width,
				output_ptr);
		}));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("error in aggregation_forward: %s\n", cudaGetErrorString(err));
	}

}

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
	const int value_grad_offset) 
{
	const int num_kernels = batch_size * value_channels * height * width;
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		output_grad.type(), "aggregation_value_backward_gpu", ([&] {
			const scalar_t *softmax_data_ptr = softmax_data.data_ptr<scalar_t>();
			const scalar_t *output_grad_ptr = output_grad.data_ptr<scalar_t>() + output_grad_offset;
			scalar_t *value_grad_ptr = value_grad.data_ptr<scalar_t>() + value_grad_offset;

			aggregation_value_backward_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
				num_kernels, softmax_data_ptr, output_grad_ptr,  
				batch_size, value_channels, height, width,
				kernel_height, kernel_width, num_group,
				dilate, stride, in_height, in_width,
				value_grad_ptr);
		}));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("error in aggregation_value_backward: %s\n", cudaGetErrorString(err));
	}

}

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
	const int output_grad_offset) 
{
	const int num_kernels = batch_size * num_group * kernel_height * kernel_width * height * width;
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		value.type(), "aggregation_softmax_backward_gpu", ([&] {

			const scalar_t *value_ptr = value.data_ptr<scalar_t>() + value_offset;
			const scalar_t *output_grad_ptr = output_grad.data_ptr<scalar_t>() + output_grad_offset;
			scalar_t *softmax_grad_ptr = softmax_grad.data_ptr<scalar_t>();

			aggregation_softmax_backward_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
				num_kernels, value_ptr, output_grad_ptr,  
				batch_size, value_channels, height, width,
				kernel_height, kernel_width, num_group,
				dilate, stride, in_height, in_width,
				softmax_grad_ptr);
		}));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("error in aggregation_softmax_backward: %s\n", cudaGetErrorString(err));
	}
}
