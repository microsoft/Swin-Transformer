# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Device-independent transcription of the index arithmetic used by the four
# CUDA kernels in swin_window_process_kernel.cu.
#
# The kernels perform no arithmetic on the tensor values: every one is a pure
# gather, and all of their logic lives in computing `input_offset` from
# `blockIdx`. Transcribing that computation to vectorised PyTorch therefore
# reproduces the kernels exactly -- which is what makes them testable with no
# compiled extension and no GPU, on a CI runner (see test_index_math.py).
#
# Exactly, but not unconditionally. Python's % and // floor, C++'s truncate, and
# in the kernels the expression is unsigned as well, so it wraps modulo 2^32
# where this file would carry a negative value through. The two agree for as
# long as the operand of every % stays non-negative, which the `+ H` and `+ W`
# terms guarantee while |shift| <= H and |shift| <= W -- a wider range than the
# launcher's TORCH_CHECKs admit, so inside the supported domain the two are the
# same arithmetic. Outside it they are not, and this file is the more forgiving
# of the two.
# --------------------------------------------------------

import torch


def to_pair(value):
    """Accept an int (isotropic) or a (h, w) iterable and return a (h, w) tuple."""
    if isinstance(value, int):
        return value, value
    h, w = value
    return int(h), int(w)


def window_partition(x, window_size):
    """(B, H, W, C) -> (B * nH * nW, window_h, window_w, C)."""
    window_h, window_w = to_pair(window_size)
    B, H, W, C = x.shape
    x = x.view(B, H // window_h, window_h, W // window_w, window_w, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_h, window_w, C)


def window_reverse(windows, window_size, H, W):
    """(B * nH * nW, window_h, window_w, C) -> (B, H, W, C)."""
    window_h, window_w = to_pair(window_size)
    C = windows.shape[-1]
    B = windows.shape[0] // ((H // window_h) * (W // window_w))
    x = windows.view(B, H // window_h, W // window_w, window_h, window_w, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)


# --------------------------------------------------------------------------
# Index maps
#
# Each function returns the flat index, into the *spatial* grid (B * H * W) or
# into the *window* grid (B * nH * nW * window_h * window_w), that every output
# element reads from. Channels are omitted: the kernels loop over C with a fixed
# stride of 1, so the channel dimension contributes an identical offset to every
# element and can be factored out by gathering over rows of a (-1, C) view.
#
# `shift_h` / `shift_w` follow the kernel convention, which is the *negated*
# torch.roll shift on the partition path — callers pass -shift_size to the
# forward kernels and +shift_size to the reverse ones. See the identities
# asserted in test_index_math.py.
# --------------------------------------------------------------------------


def roll_and_window_partition_forward_index(B, H, W, shift_h, shift_w, window_h, window_w):
    """K1: roll + window partition. Reads from the spatial grid.

    Returns:
        (B * nH * nW, window_h, window_w) int64 tensor of flat indices into
        a (B * H * W, C) view of the input.
    """
    nH, nW = H // window_h, W // window_w
    flat_win = torch.arange(B * nH * nW).view(-1, 1, 1)      # blockIdx.z
    ly = torch.arange(window_h).view(1, -1, 1)               # blockIdx.y
    lx = torch.arange(window_w).view(1, 1, -1)               # blockIdx.x

    b = flat_win // (nH * nW)
    row = (flat_win % (nH * nW) // nW * window_h + ly - shift_h + H) % H
    col = (flat_win % nW * window_w + lx - shift_w + W) % W
    return b * (H * W) + row * W + col


def roll_and_window_partition_backward_index(B, H, W, shift_h, shift_w, window_h, window_w):
    """K2: backward of K1. Reads from the window grid.

    Returns:
        (B, H, W) int64 tensor of flat indices into a
        (B * nH * nW * window_h * window_w, C) view of the incoming gradient.
    """
    nH, nW = H // window_h, W // window_w
    b = torch.arange(B).view(-1, 1, 1)                       # blockIdx.z
    y = torch.arange(H).view(1, -1, 1)                       # blockIdx.y
    x = torch.arange(W).view(1, 1, -1)                       # blockIdx.x

    src_y = (y + shift_h + H) % H
    src_x = (x + shift_w + W) % W
    win = b * (nH * nW) + src_y // window_h * nW + src_x // window_w
    return win * (window_h * window_w) + (src_y % window_h) * window_w + (src_x % window_w)


def window_merge_and_roll_forward_index(
    B, H, W, shift_h, shift_w, window_h, window_w,
    *,
    legacy_row_stride=False,
    legacy_intra_modulo=False,
):
    """K3: window merge + reverse roll. Reads from the window grid.

    Args:
        legacy_row_stride: reproduce the upstream ``* nH`` term. The stride
            between consecutive window *rows* is nW (there are nW windows per
            row in the row-major layout ``b * nH * nW + wrow * nW + wcol``), so
            ``* nH`` is only correct when nH == nW, i.e. for square feature
            maps. This is the bug fixed by this branch.
        legacy_intra_modulo: reproduce the upstream intra-window modulo, which
            omits the ``% H`` / ``% W`` reduction before ``% window_h`` /
            ``% window_w``. This is equivalent to the explicit form whenever
            H % window_h == 0 and W % window_w == 0, which the launcher already
            requires; the flag exists to make that equivalence testable rather
            than assumed.

    Returns:
        (B, H, W) int64 tensor of flat indices into a
        (B * nH * nW * window_h * window_w, C) view of the input.
    """
    nH, nW = H // window_h, W // window_w
    b = torch.arange(B).view(-1, 1, 1)                       # blockIdx.z
    y = torch.arange(H).view(1, -1, 1)                       # blockIdx.y
    x = torch.arange(W).view(1, 1, -1)                       # blockIdx.x

    src_y = (y - shift_h + H) % H
    src_x = (x - shift_w + W) % W

    row_stride = nH if legacy_row_stride else nW
    win = b * (nH * nW) + src_y // window_h * row_stride + src_x // window_w

    if legacy_intra_modulo:
        intra_y = (y - shift_h + H) % window_h
        intra_x = (x - shift_w + W) % window_w
    else:
        intra_y = src_y % window_h
        intra_x = src_x % window_w

    return win * (window_h * window_w) + intra_y * window_w + intra_x


def window_merge_and_roll_backward_index(B, H, W, shift_h, shift_w, window_h, window_w):
    """K4: backward of K3. Reads from the spatial grid.

    Returns:
        (B * nH * nW, window_h, window_w) int64 tensor of flat indices into
        a (B * H * W, C) view of the incoming gradient.
    """
    nH, nW = H // window_h, W // window_w
    flat_win = torch.arange(B * nH * nW).view(-1, 1, 1)      # blockIdx.z
    ly = torch.arange(window_h).view(1, -1, 1)               # blockIdx.y
    lx = torch.arange(window_w).view(1, 1, -1)               # blockIdx.x

    b = flat_win // (nH * nW)
    row = (flat_win % (nH * nW) // nW * window_h + ly + shift_h + H) % H
    col = (flat_win % nW * window_w + lx + shift_w + W) % W
    return b * (H * W) + row * W + col


# --------------------------------------------------------------------------
# Gathers
# --------------------------------------------------------------------------


def _gather(src, index, out_shape):
    """Apply a flat index map to the leading dimensions of ``src``.

    Args:
        src: tensor whose last dimension is C.
        index: int64 tensor of flat indices into ``src.view(-1, C)``.
        out_shape: spatial shape of the result, C appended by this function.
    """
    C = src.shape[-1]
    flat = src.reshape(-1, C)
    return flat[index.reshape(-1)].view(*out_shape, C)


def roll_and_window_partition_forward(x, shift_size, window_size):
    """Reference implementation of K1. ``x``: (B, H, W, C)."""
    shift_h, shift_w = to_pair(shift_size)
    window_h, window_w = to_pair(window_size)
    B, H, W, _ = x.shape
    index = roll_and_window_partition_forward_index(
        B, H, W, shift_h, shift_w, window_h, window_w)
    nH, nW = H // window_h, W // window_w
    return _gather(x, index, (B * nH * nW, window_h, window_w))


def roll_and_window_partition_backward(grad_in, shift_size, window_size, H, W):
    """Reference implementation of K2. ``grad_in``: (B*nH*nW, window_h, window_w, C)."""
    shift_h, shift_w = to_pair(shift_size)
    window_h, window_w = to_pair(window_size)
    B = grad_in.shape[0] // ((H // window_h) * (W // window_w))
    index = roll_and_window_partition_backward_index(
        B, H, W, shift_h, shift_w, window_h, window_w)
    return _gather(grad_in, index, (B, H, W))


def window_merge_and_roll_forward(x, shift_size, window_size, H, W, **legacy):
    """Reference implementation of K3. ``x``: (B*nH*nW, window_h, window_w, C)."""
    shift_h, shift_w = to_pair(shift_size)
    window_h, window_w = to_pair(window_size)
    B = x.shape[0] // ((H // window_h) * (W // window_w))
    index = window_merge_and_roll_forward_index(
        B, H, W, shift_h, shift_w, window_h, window_w, **legacy)
    return _gather(x, index, (B, H, W))


def window_merge_and_roll_backward(grad_in, shift_size, window_size):
    """Reference implementation of K4. ``grad_in``: (B, H, W, C)."""
    shift_h, shift_w = to_pair(shift_size)
    window_h, window_w = to_pair(window_size)
    B, H, W, _ = grad_in.shape
    index = window_merge_and_roll_backward_index(
        B, H, W, shift_h, shift_w, window_h, window_w)
    nH, nW = H // window_h, W // window_w
    return _gather(grad_in, index, (B * nH * nW, window_h, window_w))
