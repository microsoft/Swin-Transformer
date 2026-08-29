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

#include <cstdlib>
#include <limits>


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
    const int window_w);


at::Tensor roll_and_window_partition_backward_cuda(
    at::Tensor & grad_in, 
    //at::Tensor & grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w);


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
    const int window_w);

at::Tensor window_merge_and_roll_backward_cuda(
    at::Tensor & grad_in, 
    //at::Tensor & grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w);


#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// The kernels derive nH = H / window_h and nW = W / window_w with integer
// division and index with int arithmetic. All four entry points share the same
// constraints -- the window layout and the spatial layout hold the same number
// of elements -- so one check covers them.
//
// The checks below are not all guarding the same failure. Divisibility, numel
// and the int32 bound catch what used to be a silent wrong read or an
// out-of-bounds one, with no error raised at any point. The grid bounds catch a
// launch that used to fail asynchronously somewhere else, leaving the output
// uninitialised. The shift bound is neither: the formula stays correct for
// window <= |shift| <= H (and <= W on the other axis), which is simply a larger
// roll. Past that the `+ H` no longer keeps the operand of `%` non-negative,
// and since the block index is unsigned the expression wraps modulo 2^32 rather
// than going negative -- `% H` still yields an in-range index, so the read is
// silently wrong rather than out of bounds. The bound is here because it is the
// contract the model already documents (`0 <= shift_size < window_size` in
// models/swin_transformer.py), made explicit at the boundary.
static void check_window_args(
    const at::Tensor & tensor,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){

    TORCH_CHECK(B > 0 && H > 0 && W > 0 && C > 0,
        "B, H, W and C must be positive, got (", B, ", ", H, ", ", W, ", ", C, ")");
    TORCH_CHECK(window_h > 0 && window_w > 0,
        "window size must be positive, got (", window_h, ", ", window_w, ")");
    TORCH_CHECK(H % window_h == 0,
        "H (", H, ") must be divisible by window_h (", window_h, ")");
    TORCH_CHECK(W % window_w == 0,
        "W (", W, ") must be divisible by window_w (", window_w, ")");
    TORCH_CHECK(std::abs(shift_h) < window_h && std::abs(shift_w) < window_w,
        "|shift| must be smaller than the window size on each axis, got shift (",
        shift_h, ", ", shift_w, ") for window (", window_h, ", ", window_w, ")");

    const int64_t numel = static_cast<int64_t>(B) * H * W * C;
    TORCH_CHECK(tensor.numel() == numel,
        "expected ", numel, " elements for B=", B, ", H=", H, ", W=", W, ", C=", C,
        ", got ", tensor.numel());
    // Offsets are computed in int inside the kernels, as they are upstream.
    TORCH_CHECK(numel <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "tensor holds ", numel, " elements; kernel offsets are computed in int32 "
        "and would overflow");

    // grid.z is B * nH * nW for the partition kernels and B for the merge ones;
    // grid.y is window_h or H. Both are capped at 65535 by the CUDA driver.
    const int64_t num_windows = static_cast<int64_t>(B) * (H / window_h) * (W / window_w);
    TORCH_CHECK(num_windows <= 65535,
        "B * nH * nW = ", num_windows, " exceeds the CUDA grid.z limit of 65535");
    TORCH_CHECK(H <= 65535,
        "H = ", H, " exceeds the CUDA grid.y limit of 65535");
}



at::Tensor roll_and_window_partition_forward(
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
    CHECK_INPUT(input);
    check_window_args(input, B, H, W, C, shift_h, shift_w, window_h, window_w);
    return roll_and_window_partition_forward_cuda(input, B, H, W, C, shift_h, shift_w, window_h, window_w);
}


at::Tensor roll_and_window_partition_backward(
    at::Tensor & grad_in, 
    //at::Tensor & grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){
    CHECK_INPUT(grad_in);
    check_window_args(grad_in, B, H, W, C, shift_h, shift_w, window_h, window_w);
    return roll_and_window_partition_backward_cuda(grad_in, B, H, W, C, shift_h, shift_w, window_h, window_w);
}


at::Tensor window_merge_and_roll_forward(
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
    CHECK_INPUT(input);
    check_window_args(input, B, H, W, C, shift_h, shift_w, window_h, window_w);
    return window_merge_and_roll_forward_cuda(input, B, H, W, C, shift_h, shift_w, window_h, window_w);
}


at::Tensor window_merge_and_roll_backward(
    at::Tensor & grad_in, 
    //at::Tensor & grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){
    CHECK_INPUT(grad_in);
    check_window_args(grad_in, B, H, W, C, shift_h, shift_w, window_h, window_w);
    return window_merge_and_roll_backward_cuda(grad_in, B, H, W, C, shift_h, shift_w, window_h, window_w);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("roll_and_window_partition_forward", &roll_and_window_partition_forward, "torch.roll and window_partition.");
    m.def("roll_and_window_partition_backward", &roll_and_window_partition_backward, "torch.roll and window_partition.");
    m.def("window_merge_and_roll_forward", &window_merge_and_roll_forward, "window merge and torch.roll.");
    m.def("window_merge_and_roll_backward", &window_merge_and_roll_backward, "window merge and torch.roll.");
}