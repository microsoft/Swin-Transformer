# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# End-to-end check: a SwinTransformer must produce identical output with and
# without the fused window kernels. The kernels replace torch.roll plus
# window_partition, which are exact data movements, so the two paths are not
# merely close -- they are bit-for-bit equal.
#
#   python test_model_parity.py
# --------------------------------------------------------

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    import swin_window_process  # noqa: F401
    EXTENSION_AVAILABLE = True
except ImportError:
    EXTENSION_AVAILABLE = False

try:
    from models.swin_transformer import SwinTransformer
    MODEL_AVAILABLE = True
except ImportError:                                  # timm missing, for instance
    SwinTransformer = None
    MODEL_AVAILABLE = False


# How far a gradient may move under the fused path before the test calls it a
# regression, as a multiple of the run-to-run noise measured on the same machine.
# Only gradients that already fail to reproduce against a second eager run are
# held to this; every other one must still be bit-exact. The noise floor is a
# single sample, hence the headroom: on an MI300X it measured 1.9e-9 against a
# gradient scale of 0.254, so even 8x of it stays seven orders of magnitude below
# the signal.
NOISE_HEADROOM = 8

requires_everything = unittest.skipIf(
    not (torch.cuda.is_available() and EXTENSION_AVAILABLE and MODEL_AVAILABLE),
    'requires a CUDA device, the compiled extension and an importable model')


def set_fused(model, enabled):
    """Flip the flag on every block, so both runs share exactly the same weights."""
    switched = 0
    for module in model.modules():
        if hasattr(module, 'fused_window_process'):
            module.fused_window_process = enabled
            switched += 1
    return switched


@requires_everything
class TestModelParity(unittest.TestCase):

    def _model(self, img_size=56, window_size=7, depths=(2, 2), num_heads=(3, 6)):
        torch.manual_seed(0)
        model = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=10,
            embed_dim=48,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            drop_path_rate=0.0,
            fused_window_process=False,
        )
        return model.cuda().eval()

    def _assert_non_square_parity(self, img_size):
        """img_size is documented as `int | tuple(int)`, and a non-square one
        gives nH != nW inside the shifted blocks -- the configuration the row
        stride fix is about.

        For img_size=(256, 128) with window_size=8 the shifted blocks run at
        64x32 and 32x16, so nH/nW are 8/4 and 4/2; the transpose (128, 256)
        gives nW > nH instead. Before the fix the merge kernel mis-indexes on
        both -- out of bounds when nH > nW, silently wrong when nH < nW.
        """
        H, W = img_size
        model = self._model(img_size=img_size, window_size=8,
                            depths=(2, 2, 2), num_heads=(3, 6, 12))

        shifted_non_square = [
            m for m in model.modules()
            if type(m).__name__ == 'SwinTransformerBlock'
            and m.shift_size > 0
            and (m.input_resolution[0] // m.window_size)
            != (m.input_resolution[1] // m.window_size)
        ]
        self.assertGreater(len(shifted_non_square), 0,
                           'this configuration must exercise nH != nW')

        x = torch.randn(1, 3, H, W, device='cuda')
        with torch.no_grad():
            eager = model(x)
            set_fused(model, True)
            fused = model(x)
            set_fused(model, False)

        self.assertTrue(torch.equal(eager, fused))

    def test_non_square_image_tall(self):
        """H > W: the shifted blocks hit nH > nW, the out-of-bounds case."""
        self._assert_non_square_parity((256, 128))

    def test_non_square_image_wide(self):
        """W > H: the shifted blocks hit nH < nW, the silent-corruption case."""
        self._assert_non_square_parity((128, 256))

    def test_forward_is_identical(self):
        model = self._model()
        x = torch.randn(2, 3, 56, 56, device='cuda')

        with torch.no_grad():
            eager = model(x)
            blocks = set_fused(model, True)
            fused = model(x)
            set_fused(model, False)

        # set_fused counts every block carrying the flag, not just the shifted
        # ones that route through the kernels. Zero would mean the model exposes
        # no flag at all, and that the comparison below proved nothing.
        self.assertGreater(blocks, 0, 'no block exposes fused_window_process')
        self.assertTrue(torch.equal(eager, fused))

    def test_gradients_are_identical(self):
        """The fused path must not perturb a whole-model backward.

        Bit-exactness of the four kernels themselves is asserted in
        unit_test.py, where it holds for every dtype and shape. At model level
        the bar has to account for the rest of the network: the backward of a
        GEMM is not always reproducible run to run, because the library is free
        to pick a different reduction split each time. On an MI300X one of the
        63 gradients moves by ~2e-9 between two *identical* eager runs, so
        asserting torch.equal against eager would fail without any fused kernel
        being involved.

        So the eager path is run twice first, and each gradient is held to what
        that measurement licenses: bit-exactness wherever eager reproduces
        itself, and no further from eager than eager is from itself elsewhere.
        On a platform where the backward is fully deterministic -- CUDA, in
        every run of this test so far -- every gradient takes the first branch
        and this is exactly the strict comparison it replaces.
        """
        model = self._model()
        x = torch.randn(2, 3, 56, 56, device='cuda')
        target = torch.randn(2, 10, device='cuda')

        def grads():
            model.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model(x), target).backward()
            return [p.grad.clone() for p in model.parameters() if p.grad is not None]

        eager = grads()
        eager_again = grads()          # the platform's own run-to-run noise
        set_fused(model, True)
        fused = grads()
        set_fused(model, False)

        self.assertEqual(len(eager), len(fused))
        reproducible = 0
        for i, (a, a2, b) in enumerate(zip(eager, eager_again, fused)):
            with self.subTest(parameter=i):
                if torch.equal(a, a2):
                    reproducible += 1
                    self.assertTrue(torch.equal(a, b))
                else:
                    noise = (a - a2).abs().max().item()
                    delta = (a - b).abs().max().item()
                    self.assertLessEqual(
                        delta, NOISE_HEADROOM * noise,
                        f'gradient {i} moves {delta:.3e} with the fused path, '
                        f'against {noise:.3e} between two eager runs')

        # A platform that reproduces nothing would make this test vacuous.
        self.assertGreater(reproducible, len(eager) // 2)


if __name__ == '__main__':
    if not (torch.cuda.is_available() and EXTENSION_AVAILABLE and MODEL_AVAILABLE):
        print('Skipping: needs a CUDA device, the built extension and the model.\n')
    unittest.main(verbosity=2)
