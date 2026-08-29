# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Fused kernels against the PyTorch ops they replace, in time and peak memory.

The eager path materialises the tensor twice -- torch.roll writes a copy, then
window_partition permutes and calls .contiguous() for another -- where the fused
kernel writes once. The gain is what that saved traffic is worth on the device,
which varies enough between vendors to be worth measuring rather than
predicting: README.md records what these configurations gave on an RTX 3080 and
on an MI300X.

    python benchmark.py                  # forward + backward, all dtypes
    python benchmark.py --iters 200 --batch 64
"""

import argparse

import torch

import reference as ref

try:
    from window_process import WindowProcess, WindowProcessReverse
except ImportError as exc:                           # extension not built here
    WindowProcess = WindowProcessReverse = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


# (name, B, H, W, C, window_h, window_w)
CONFIGS = [
    ('swin-tiny stage 1  56x56 w7', 192, 56, 56, 96, 7, 7),
    ('swin-tiny stage 2  28x28 w7', 192, 28, 28, 192, 7, 7),
    ('non-square         32x16 w8', 192, 32, 16, 96, 8, 8),
    ('non-square window  16x64 w4x16', 192, 16, 64, 96, 4, 16),
]


def timed(fn, iters, warmup=10):
    """Wall time per iteration in ms, and peak allocated memory in MiB."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2 ** 20


def make_runners(B, H, W, C, window_h, window_w, shift_h, shift_w, dtype, backward):
    window = (window_h, window_w)
    n = B * (H // window_h) * (W // window_w)

    x = torch.randn((B, H, W, C), dtype=dtype, device='cuda', requires_grad=backward)
    grad = torch.randn((n, window_h, window_w, C), dtype=dtype, device='cuda')

    def eager():
        shifted = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
        out = ref.window_partition(shifted, window)
        if backward:
            x.grad = None
            out.backward(grad)

    def fused():
        out = WindowProcess.apply(x, B, H, W, C, (-shift_h, -shift_w), window)
        if backward:
            x.grad = None
            out.backward(grad)

    return eager, fused


def make_reverse_runners(B, H, W, C, window_h, window_w, shift_h, shift_w, dtype, backward):
    window = (window_h, window_w)
    n = B * (H // window_h) * (W // window_w)

    x = torch.randn((n, window_h, window_w, C), dtype=dtype, device='cuda',
                    requires_grad=backward)
    grad = torch.randn((B, H, W, C), dtype=dtype, device='cuda')

    def eager():
        merged = ref.window_reverse(x, window, H, W)
        out = torch.roll(merged, shifts=(shift_h, shift_w), dims=(1, 2))
        if backward:
            x.grad = None
            out.backward(grad)

    def fused():
        out = WindowProcessReverse.apply(x, B, H, W, C, (shift_h, shift_w), window)
        if backward:
            x.grad = None
            out.backward(grad)

    return eager, fused


def run(args):
    dtypes = [torch.float32, torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    print(f'device: {torch.cuda.get_device_name()}')
    print(f'torch:  {torch.__version__} (cuda {torch.version.cuda}, hip {torch.version.hip})')
    print(f'mode:   {"forward + backward" if not args.forward_only else "forward only"}, '
          f'{args.iters} iterations after 10 warmup\n')

    header = (f'| {"config":<32} | {"op":<7} | {"dtype":<9} | {"eager ms":>9} | '
              f'{"fused ms":>9} | {"speedup":>8} | {"eager MiB":>10} | {"fused MiB":>10} |')
    print(header)
    print('|' + '|'.join('-' * (len(c) + 2) for c in header.split('|')[1:-1]) + '|')

    for name, B, H, W, C, window_h, window_w in CONFIGS:
        B = args.batch or B
        shift_h, shift_w = window_h // 2, window_w // 2
        for op, factory in (('partition', make_runners), ('merge', make_reverse_runners)):
            for dtype in dtypes:
                eager, fused = factory(B, H, W, C, window_h, window_w,
                                       shift_h, shift_w, dtype,
                                       backward=not args.forward_only)
                e_ms, e_mem = timed(eager, args.iters)
                f_ms, f_mem = timed(fused, args.iters)
                print(f'| {name:<32} | {op:<7} | {str(dtype).replace("torch.", ""):<9} | '
                      f'{e_ms:>9.3f} | {f_ms:>9.3f} | {e_ms / f_ms:>7.2f}x | '
                      f'{e_mem:>10.1f} | {f_mem:>10.1f} |')
                del eager, fused
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--iters', type=int, default=100,
                        help='timed iterations per configuration')
    parser.add_argument('--batch', type=int, default=None,
                        help='override the batch size of every config')
    parser.add_argument('--forward-only', action='store_true',
                        help='skip the backward pass')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('benchmark.py requires a CUDA device')
    if IMPORT_ERROR is not None:
        raise SystemExit(
            f'the swin_window_process extension is not importable ({IMPORT_ERROR}); '
            'build it with: python setup.py install')

    torch.manual_seed(0)
    run(args)


if __name__ == '__main__':
    main()
