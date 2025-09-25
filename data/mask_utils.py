"""Utility components for data masking transformations."""

from __future__ import annotations

import numpy as np


class MaskGenerator:
    """Generate random block masks for masked image modeling.

    Parameters
    ----------
    input_size: int, default 192
        Size of the square input image to mask.
    mask_patch_size: int, default 32
        Patch size for selecting random regions to be masked.
    model_patch_size: int, default 4
        Patch size expected by the model. Determines mask upscaling.
    mask_ratio: float, default 0.6
        Ratio of patches to mask.
    """

    def __init__(
        self,
        input_size: int = 192,
        mask_patch_size: int = 32,
        model_patch_size: int = 4,
        mask_ratio: float = 0.6,
    ) -> None:
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise AssertionError("input_size must be divisible by mask_patch_size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise AssertionError("mask_patch_size must be divisible by model_patch_size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self) -> np.ndarray:
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


__all__ = ["MaskGenerator"]
