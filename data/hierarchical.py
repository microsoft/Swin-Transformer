import os

import torch
from timm.data import Mixup, mixup
from torchvision.datasets import ImageFolder


class HierarchicalImageFolder(ImageFolder):
    """
    Parses an image folder where the hierarchy is represented as follows:

    00000_top_middle_..._bottom
    00001_top_middle_..._other
    ...
    """

    num_classes = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_classes(self, directory):
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )

        tier_lookup = {}
        class_to_idxs = {}

        for cls in classes:
            tiers = make_hierarchical(cls)

            for tier, value in enumerate(tiers):
                if tier not in tier_lookup:
                    tier_lookup[tier] = {}

                if value not in tier_lookup[tier]:
                    tier_lookup[tier][value] = len(tier_lookup[tier])

            class_to_idxs[cls] = torch.tensor(
                [tier_lookup[tier][value] for tier, value in enumerate(tiers)]
            )

        # Set self.num_classes
        self.num_classes = tuple(len(tier) for tier in tier_lookup.values())

        return classes, class_to_idxs


def make_hierarchical(name):
    """
    Sometimes the tree is not really a tree; that is, sometimes there are
    repeated orders, for example.

    Arguments:
        name (str): the complete taxonomic name, separated by '_'
    """
    # index is a number
    # top is kingdom
    index, top, *tiers = name.split("_")

    cleaned = [top]

    complete = top
    for tier in tiers:
        complete += f"-{tier}"
        cleaned.append(complete)

    return cleaned


class HierarchicalMixup(Mixup):
    def __call__(self, inputs, targets):
        assert len(inputs) % 2 == 0, "Batch size should be even when using this"
        if self.mode == "elem":
            lam = self._mix_elem(inputs)
        elif self.mode == "pair":
            lam = self._mix_pair(inputs)
        else:
            lam = self._mix_batch(inputs)

        batch_size, *_ = inputs.shape
        assert targets.shape == (
            batch_size,
            len(self.num_classes),
        ), f"{targets.shape} != {batch_size, len(self.num_classes)}"

        targets = [
            mixup.mixup_target(
                target, num_classes, lam, self.label_smoothing, inputs.device
            )
            for target, num_classes in zip(targets.T, self.num_classes)
        ]
        return inputs, targets
