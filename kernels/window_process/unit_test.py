# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Parity tests for the compiled kernels against the PyTorch ops they replace.
# Requires a CUDA device and the built extension; the index math itself is
# covered without either by test_index_math.py.
#
#   python unit_test.py
# --------------------------------------------------------

import unittest

import torch

import reference as ref

try:
    from window_process import WindowProcess, WindowProcessReverse
    EXTENSION_AVAILABLE = True
except ImportError:                                  # extension not built here
    WindowProcess = WindowProcessReverse = None
    EXTENSION_AVAILABLE = False


CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = unittest.skipIf(
    not (CUDA_AVAILABLE and EXTENSION_AVAILABLE),
    'requires a CUDA device and the compiled swin_window_process extension')


def available_dtypes():
    """float64 and float32 come from AT_DISPATCH_FLOATING_TYPES; the rest are added."""
    dtypes = [torch.float64, torch.float32, torch.float16]
    if CUDA_AVAILABLE and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


# (B, H, W, C, window_h, window_w)
SHAPES = [
    (2, 56, 56, 96, 7, 7),     # nH == nW == 8   the ImageNet configuration
    (2, 32, 16, 64, 8, 8),     # nH=4,  nW=2     out-of-bounds read before the fix
    (2, 16, 32, 64, 8, 8),     # nH=2,  nW=4     silent corruption before the fix
    (2, 8, 32, 64, 8, 8),      # nH=1,  nW=4     one row of windows: hides the bug
    (2, 16, 64, 64, 4, 16),    # nH=4,  nW=4     non-square window
    (2, 64, 16, 64, 16, 4),    # nH=4,  nW=4     non-square window, transposed
    (2, 48, 48, 32, 6, 16),    # nH=8,  nW=3     square image, non-square window
    (2, 24, 16, 64, 8, 8),     # nH=3,  nW=2     coprime counts, both > 1
    (2, 24, 40, 32, 8, 8),     # nH=3,  nW=5     coprime counts, nH < nW
    (2, 24, 24, 32, 24, 24),   # nH == nW == 1   a single window
    # C selects the block width in best_block_dim(): < 384 -> 64 threads,
    # < 1024 -> 128, otherwise 256. All three branches must be exercised, and C
    # must be smaller than, equal to and larger than the block width so that the
    # strided `for (i = threadIdx.x; i < C; i += blockDim.x)` loop is covered in
    # all four of its forms: no full pass, exactly one, an exact number, and a
    # ragged one where the last pass is partial.
    (2, 16, 16, 1, 8, 8),      # C < blockDim      64 threads, 63 idle
    (2, 16, 16, 64, 8, 8),     # C == blockDim     64 threads, one pass
    (2, 16, 64, 100, 4, 16),   # C % blockDim != 0 64 threads, 2 passes, 36 wide,
                               #                   and a non-square window with it
    (2, 16, 16, 512, 8, 8),    # 384 <= C < 1024   128 threads, 4 exact passes
    (2, 16, 16, 1024, 8, 8),   # C >= 1024         256 threads, 4 exact passes
]


def shifts_for(window_h, window_w):
    """W-MSA (no shift), SW-MSA (half window, the isotropic shift Swin uses), and
    an asymmetric shift_h != shift_w -- a regime only the per-axis signature on
    this branch can express, so it needs its own coverage."""
    return [(0, 0), (window_h // 2, window_w // 2), (1, 2)]


def pyt_forward(x, shift, window):
    """torch.roll followed by window_partition -- what WindowProcess fuses."""
    shift_h, shift_w = ref.to_pair(shift)
    if shift_h or shift_w:
        x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
    return ref.window_partition(x, window)


def reverse_pyt_forward(windows, shift, window, H, W):
    """window_reverse followed by torch.roll -- what WindowProcessReverse fuses."""
    shift_h, shift_w = ref.to_pair(shift)
    x = ref.window_reverse(windows, window, H, W)
    if shift_h or shift_w:
        x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))
    return x


def leaf(tensor, requires_grad=True):
    # .cuda() must come before requires_grad_: moving a tensor that already
    # requires grad makes the result a non-leaf, and .grad never populates on it.
    return tensor.clone().detach().cuda().requires_grad_(requires_grad)


def spatial(B, H, W, C, dtype):
    return torch.randn((B, H, W, C), dtype=dtype)


def windowed(B, H, W, C, window_h, window_w, dtype):
    n = B * (H // window_h) * (W // window_w)
    return torch.randn((n, window_h, window_w, C), dtype=dtype)


@requires_cuda
class TestWindowProcess(unittest.TestCase):
    """The kernels are exact permutations, so parity must be bit-for-bit.

    torch.equal is a stronger check than gradcheck here: no arithmetic is
    performed on the values, so any deviation is an indexing error, not a
    numerical one. Every dtype is therefore held to exact equality.
    """

    def _cases(self):
        for dtype in available_dtypes():
            for B, H, W, C, window_h, window_w in SHAPES:
                for shift in shifts_for(window_h, window_w):
                    yield dtype, B, H, W, C, (window_h, window_w), shift

    def test_partition_forward(self):
        for dtype, B, H, W, C, window, shift in self._cases():
            with self.subTest(dtype=dtype, shape=(B, H, W, C), window=window, shift=shift):
                x = spatial(B, H, W, C, dtype)
                with torch.no_grad():
                    expected = pyt_forward(leaf(x), shift, window)
                    fused = WindowProcess.apply(
                        leaf(x), B, H, W, C, (-shift[0], -shift[1]), window)
                self.assertTrue(torch.equal(expected, fused))

    def test_partition_backward(self):
        for dtype, B, H, W, C, window, shift in self._cases():
            with self.subTest(dtype=dtype, shape=(B, H, W, C), window=window, shift=shift):
                x = spatial(B, H, W, C, dtype)
                grad = windowed(B, H, W, C, window[0], window[1], dtype).cuda()

                a, b = leaf(x), leaf(x)
                pyt_forward(a, shift, window).backward(grad)
                WindowProcess.apply(
                    b, B, H, W, C, (-shift[0], -shift[1]), window).backward(grad)

                self.assertIsNotNone(a.grad)
                self.assertTrue(torch.equal(a.grad, b.grad))

    def test_merge_forward(self):
        for dtype, B, H, W, C, window, shift in self._cases():
            with self.subTest(dtype=dtype, shape=(B, H, W, C), window=window, shift=shift):
                x = windowed(B, H, W, C, window[0], window[1], dtype)
                with torch.no_grad():
                    expected = reverse_pyt_forward(leaf(x), shift, window, H, W)
                    fused = WindowProcessReverse.apply(leaf(x), B, H, W, C, shift, window)
                self.assertTrue(torch.equal(expected, fused))

    def test_merge_backward(self):
        for dtype, B, H, W, C, window, shift in self._cases():
            with self.subTest(dtype=dtype, shape=(B, H, W, C), window=window, shift=shift):
                x = windowed(B, H, W, C, window[0], window[1], dtype)
                grad = spatial(B, H, W, C, dtype).cuda()

                a, b = leaf(x), leaf(x)
                reverse_pyt_forward(a, shift, window, H, W).backward(grad)
                WindowProcessReverse.apply(b, B, H, W, C, shift, window).backward(grad)

                self.assertIsNotNone(a.grad)
                self.assertTrue(torch.equal(a.grad, b.grad))

    def test_round_trip(self):
        """WindowProcessReverse must undo WindowProcess for the same shift."""
        for dtype, B, H, W, C, window, shift in self._cases():
            with self.subTest(dtype=dtype, shape=(B, H, W, C), window=window, shift=shift):
                x = leaf(spatial(B, H, W, C, dtype), requires_grad=False)
                windows = WindowProcess.apply(
                    x, B, H, W, C, (-shift[0], -shift[1]), window)
                back = WindowProcessReverse.apply(
                    windows.contiguous(), B, H, W, C, shift, window)
                self.assertTrue(torch.equal(x, back))


    def test_non_contiguous_gradient(self):
        """An upstream op can hand the backward a non-contiguous gradient.

        The C++ side asserts contiguity, so window_process.py normalises it.

        The window is non-square on purpose: with 8x8 the transpose below is
        shape-invariant, so this would pass even if the two window axes were
        swapped somewhere along the path.
        """
        B, H, W, C, window, shift = 2, 16, 16, 32, (4, 16), (2, 8)
        n = B * (H // window[0]) * (W // window[1])
        x = spatial(B, H, W, C, torch.float32)
        grad = torch.randn((n, window[1], window[0], C)).cuda().transpose(1, 2)
        self.assertFalse(grad.is_contiguous())

        a, b = leaf(x), leaf(x)
        pyt_forward(a, shift, window).backward(grad)
        WindowProcess.apply(b, B, H, W, C, (-shift[0], -shift[1]), window).backward(grad)
        self.assertTrue(torch.equal(a.grad, b.grad))


@requires_cuda
class TestStreamSemantics(unittest.TestCase):
    """The kernels must run on the current stream, not on the default one.

    A race against the default stream is timing dependent and makes a poor
    test. CUDA graph capture is the deterministic form of the same property:
    capture runs on a non-default stream and rejects any kernel launched on the
    legacy default stream outright, so this fails to capture unless the launch
    uses at::cuda::getCurrentCUDAStream().
    """

    def test_capturable_in_a_cuda_graph(self):
        B, H, W, C, window, shift = 2, 16, 16, 32, (8, 8), (4, 4)
        static_in = torch.randn((B, H, W, C), device='cuda')

        def call():
            return WindowProcess.apply(
                static_in, B, H, W, C, (-shift[0], -shift[1]), window)

        # Warm up on a side stream, as the CUDA graph capture protocol requires.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = call()
        graph.replay()
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(static_out, pyt_forward(static_in, shift, window)))

    def test_matches_when_run_on_a_side_stream(self):
        B, H, W, C, window, shift = 2, 16, 16, 32, (8, 8), (4, 4)
        x = torch.randn((B, H, W, C), device='cuda')
        expected = pyt_forward(x, shift, window)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fused = WindowProcess.apply(x, B, H, W, C, (-shift[0], -shift[1]), window)
        torch.cuda.current_stream().wait_stream(side)

        self.assertTrue(torch.equal(expected, fused))


@requires_cuda
class TestScalarWindowStillWorks(unittest.TestCase):
    """An int shift/window must behave exactly as the (h, w) pair it expands to.

    This is the compatibility path used by models/swin_transformer.py, which
    passes ints and must keep working unchanged.
    """

    def test_int_and_pair_agree(self):
        B, H, W, C, window, shift = 2, 56, 56, 96, 7, 3
        x = spatial(B, H, W, C, torch.float32)
        with torch.no_grad():
            as_int = WindowProcess.apply(leaf(x), B, H, W, C, -shift, window)
            as_pair = WindowProcess.apply(
                leaf(x), B, H, W, C, (-shift, -shift), (window, window))
        self.assertTrue(torch.equal(as_int, as_pair))


@requires_cuda
class TestPreconditions(unittest.TestCase):
    """Invalid arguments must raise instead of reading the wrong memory."""

    def setUp(self):
        self.B, self.H, self.W, self.C = 2, 16, 16, 32
        self.x = leaf(spatial(self.B, self.H, self.W, self.C, torch.float32), False)

    def _apply(self, **overrides):
        kwargs = dict(B=self.B, H=self.H, W=self.W, C=self.C, shift=(0, 0), window=(8, 8))
        kwargs.update(overrides)
        return WindowProcess.apply(
            self.x, kwargs['B'], kwargs['H'], kwargs['W'], kwargs['C'],
            kwargs['shift'], kwargs['window'])

    def test_window_must_divide_height(self):
        with self.assertRaisesRegex(RuntimeError, 'divisible by window_h'):
            self._apply(window=(5, 8))

    def test_window_must_divide_width(self):
        with self.assertRaisesRegex(RuntimeError, 'divisible by window_w'):
            self._apply(window=(8, 5))

    def test_shift_must_be_smaller_than_window(self):
        with self.assertRaisesRegex(RuntimeError, 'smaller than the window size'):
            self._apply(shift=(8, 0))

    def test_shape_must_match_tensor(self):
        with self.assertRaisesRegex(RuntimeError, 'expected'):
            self._apply(C=self.C * 2)

    def test_window_count_above_the_grid_limit_is_rejected(self):
        """B * nH * nW is grid.z, which the CUDA driver caps at 65535."""
        B, H, W, C = 70000, 8, 8, 1
        big = leaf(spatial(B, H, W, C, torch.float32), False)
        with self.assertRaisesRegex(RuntimeError, 'grid.z limit'):
            WindowProcess.apply(big, B, H, W, C, (0, 0), (8, 8))

    def test_non_contiguous_input_is_rejected(self):
        transposed = self.x.transpose(1, 2)
        with self.assertRaises(RuntimeError):
            WindowProcess.apply(
                transposed, self.B, self.W, self.H, self.C, (0, 0), (8, 8))


if __name__ == '__main__':
    if not (CUDA_AVAILABLE and EXTENSION_AVAILABLE):
        print('No CUDA device or extension not built: every test here will be skipped.')
        print('Run test_index_math.py to check the index math without a GPU.\n')
    torch.manual_seed(0)
    unittest.main(verbosity=2)
