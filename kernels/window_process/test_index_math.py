# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Correctness tests for the index arithmetic of the fused window kernels.
#
# These run on CPU against reference.py and need neither a GPU nor the compiled
# extension, so they can gate the index math in CI. The GPU parity tests for the
# compiled kernels live in unit_test.py.
#
#   python test_index_math.py
# --------------------------------------------------------

import unittest

import torch

import reference as ref


# (H, W, window_h, window_w) -- covers square and non-square feature maps, and
# square and non-square windows.
SHAPES = [
    (8, 8, 4, 4),      # nH == nW == 2   square grid, square window
    (16, 8, 4, 4),     # nH=4, nW=2      taller than wide
    (8, 16, 4, 4),     # nH=2, nW=4      wider than tall
    (8, 32, 4, 8),     # nH=2, nW=4      non-square window
    (32, 8, 8, 4),     # nH=4, nW=2      non-square window, taller than wide
    (24, 24, 4, 8),    # nH=6, nW=3      square grid, non-square window
    (12, 8, 4, 4),     # nH=3, nW=2      coprime counts, both > 1
    (12, 20, 4, 4),    # nH=3, nW=5      coprime counts, nH < nW
    (8, 8, 8, 8),      # nH == nW == 1   single window
    (12, 12, 4, 4),    # nH == nW == 3   odd window count
    (16, 32, 4, 8),    # nH == nW == 4   square grid, non-square window
    (4, 16, 4, 4),     # nH=1, nW=4      one row of windows: the bug hides here
    (16, 4, 4, 4),     # nH=4, nW=1      one column of windows: it does not
]

BATCH = 2
CHANNELS = 3


def _shifts(window_h, window_w):
    """W-MSA (no shift), SW-MSA (half window), an asymmetric shift_h != shift_w
    that only the per-axis signature on this branch can express, and the
    negative form: models/swin_transformer.py calls the partition path with
    -shift_size, so a suite that only ever passes positive shifts leaves the
    sign the model actually uses untested."""
    return [(0, 0), (window_h // 2, window_w // 2), (1, 2),
            (-(window_h // 2), -(window_w // 2))]


def _make_spatial(B, H, W, C):
    return torch.arange(B * H * W * C, dtype=torch.float32).view(B, H, W, C)


def _make_windows(B, H, W, C, window_h, window_w):
    n = B * (H // window_h) * (W // window_w)
    return torch.arange(n * window_h * window_w * C, dtype=torch.float32).view(
        n, window_h, window_w, C)


class TestKernelIndexMath(unittest.TestCase):
    """Each kernel must equal the composition of PyTorch ops it replaces.

    The kernel `shift_h`/`shift_w` parameters are the negated torch.roll shift on
    the partition path: callers pass -shift_size to the forward kernels and
    +shift_size to the reverse ones. These identities pin that convention down.
    """

    def test_k1_equals_roll_then_partition(self):
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    x = _make_spatial(BATCH, H, W, CHANNELS)
                    got = ref.roll_and_window_partition_forward(x, (sh, sw), (wh, ww))
                    want = ref.window_partition(
                        torch.roll(x, shifts=(sh, sw), dims=(1, 2)), (wh, ww))
                    self.assertTrue(torch.equal(got, want))

    def test_k2_equals_reverse_then_unroll(self):
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    g = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    got = ref.roll_and_window_partition_backward(g, (sh, sw), (wh, ww), H, W)
                    want = torch.roll(
                        ref.window_reverse(g, (wh, ww), H, W), shifts=(-sh, -sw), dims=(1, 2))
                    self.assertTrue(torch.equal(got, want))

    def test_k3_equals_reverse_then_roll(self):
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    x = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    got = ref.window_merge_and_roll_forward(x, (sh, sw), (wh, ww), H, W)
                    want = torch.roll(
                        ref.window_reverse(x, (wh, ww), H, W), shifts=(sh, sw), dims=(1, 2))
                    self.assertTrue(torch.equal(got, want))

    def test_k4_equals_unroll_then_partition(self):
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    g = _make_spatial(BATCH, H, W, CHANNELS)
                    got = ref.window_merge_and_roll_backward(g, (sh, sw), (wh, ww))
                    want = ref.window_partition(
                        torch.roll(g, shifts=(-sh, -sw), dims=(1, 2)), (wh, ww))
                    self.assertTrue(torch.equal(got, want))

    def test_backward_kernels_invert_their_forward(self):
        """K1/K2 and K3/K4 are permutations, so each pair must round-trip."""
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    x = _make_spatial(BATCH, H, W, CHANNELS)
                    k1 = ref.roll_and_window_partition_forward(x, (sh, sw), (wh, ww))
                    self.assertTrue(torch.equal(
                        ref.roll_and_window_partition_backward(k1, (sh, sw), (wh, ww), H, W), x))

                    w = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    k3 = ref.window_merge_and_roll_forward(w, (sh, sw), (wh, ww), H, W)
                    self.assertTrue(torch.equal(
                        ref.window_merge_and_roll_backward(k3, (sh, sw), (wh, ww)), w))


class TestUpstreamRowStrideBug(unittest.TestCase):
    """window_merge_and_roll_forward used `* nH` where the row stride is `* nW`.

    Windows are laid out row-major as `b * nH * nW + wrow * nW + wcol`, so the
    stride between consecutive window rows is nW. The two agree exactly when
    nH == nW, which is why the bug never surfaced: every model in this repository
    is trained on square images.
    """

    def _legacy_index(self, B, H, W, sh, sw, wh, ww):
        return ref.window_merge_and_roll_forward_index(
            B, H, W, sh, sw, wh, ww, legacy_row_stride=True)

    def test_legacy_is_correct_only_on_square_grids(self):
        for H, W, wh, ww in SHAPES:
            nH, nW = H // wh, W // ww
            if nH != nW:
                continue
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw), nH=nH, nW=nW):
                    x = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    want = ref.window_merge_and_roll_forward(x, (sh, sw), (wh, ww), H, W)
                    legacy = ref.window_merge_and_roll_forward(
                        x, (sh, sw), (wh, ww), H, W, legacy_row_stride=True)
                    self.assertTrue(torch.equal(legacy, want))

    def test_legacy_reads_out_of_bounds_when_nH_greater_than_nW(self):
        """nH > nW makes the miscomputed offset exceed the input, an illegal read."""
        checked = 0
        for H, W, wh, ww in SHAPES:
            nH, nW = H // wh, W // ww
            if nH <= nW:
                continue
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw), nH=nH, nW=nW):
                    numel = BATCH * nH * nW * wh * ww
                    legacy_max = int(self._legacy_index(BATCH, H, W, sh, sw, wh, ww).max())
                    self.assertGreaterEqual(legacy_max, numel)
                    checked += 1
        self.assertGreater(checked, 0, "no nH > nW shape exercised")

    def test_legacy_returns_wrong_values_when_nH_less_than_nW(self):
        """nH < nW keeps the offset in bounds, so the corruption is silent."""
        checked = 0
        for H, W, wh, ww in SHAPES:
            nH, nW = H // wh, W // ww
            if nH >= nW or nH == 1:
                continue           # nH == 1 hides the bug -- covered separately
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw), nH=nH, nW=nW):
                    numel = BATCH * nH * nW * wh * ww
                    legacy_max = int(self._legacy_index(BATCH, H, W, sh, sw, wh, ww).max())
                    self.assertLess(legacy_max, numel)

                    x = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    want = ref.window_merge_and_roll_forward(x, (sh, sw), (wh, ww), H, W)
                    legacy = ref.window_merge_and_roll_forward(
                        x, (sh, sw), (wh, ww), H, W, legacy_row_stride=True)
                    self.assertFalse(torch.equal(legacy, want))
                    checked += 1
        self.assertGreater(checked, 0, "no nH < nW shape exercised")

    def test_legacy_is_invisible_when_nH_is_one(self):
        """A single row of windows hides the bug entirely; a single column does not.

        The legacy term is `src_y / window_h * nH`, and `src_y / window_h` runs
        over [0, nH). At nH == 1 it is always 0, so the wrong stride is never
        multiplied by anything and the result is bit-identical to the correct
        one -- even though nH != nW and the grid is as non-square as it gets.
        The transpose has no such reprieve: nW == 1 with nH > 1 is the
        out-of-bounds case, asserted above.

        This is why H != W is not on its own enough to reproduce the bug, and
        why the shape list has to contain both orientations.
        """
        checked = 0
        for H, W, wh, ww in SHAPES:
            nH, nW = H // wh, W // ww
            if nH != 1 or nW == 1:
                continue
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw), nH=nH, nW=nW):
                    x = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    want = ref.window_merge_and_roll_forward(x, (sh, sw), (wh, ww), H, W)
                    legacy = ref.window_merge_and_roll_forward(
                        x, (sh, sw), (wh, ww), H, W, legacy_row_stride=True)
                    self.assertTrue(torch.equal(legacy, want))
                    checked += 1
        self.assertGreater(checked, 0, "no nH == 1 < nW shape exercised")


class TestIntraWindowModuloIsEquivalent(unittest.TestCase):
    """The upstream intra-window modulo omits `% H` / `% W`, which is a no-op here.

    `(y - s + H) % window_h` and `((y - s + H) % H) % window_h` differ by a
    multiple of H, and H is a multiple of window_h whenever the launcher's
    `nH = H / window_h` is exact -- which it must be for the kernel to be valid
    at all. The explicit form is kept for readability, not for correctness.
    """

    def test_forms_agree_under_exact_divisibility(self):
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    explicit = ref.window_merge_and_roll_forward_index(
                        BATCH, H, W, sh, sw, wh, ww, legacy_intra_modulo=False)
                    legacy = ref.window_merge_and_roll_forward_index(
                        BATCH, H, W, sh, sw, wh, ww, legacy_intra_modulo=True)
                    self.assertTrue(torch.equal(explicit, legacy))


if __name__ == '__main__':
    unittest.main(verbosity=2)
