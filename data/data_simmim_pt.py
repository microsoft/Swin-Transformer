# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .memmap_dataset import MaskGenerator, MemmapImageDataset


class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])

        if config.MODEL.TYPE in ['swin', 'swinv2']:
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()

        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def _should_use_memmap(config) -> bool:
    dataset_name = getattr(getattr(config, "DATA", object()), "DATASET", "")
    if isinstance(dataset_name, str) and dataset_name.lower() == "memmap":
        return True

    data_path = getattr(getattr(config, "DATA", object()), "DATA_PATH", "")
    if isinstance(data_path, str) and data_path.lower().endswith(".memmap"):
        return True

    return False


def build_loader_simmim(config):
    transform = SimMIMTransform(config)

    if _should_use_memmap(config):
        metadata_path = getattr(config.DATA, "MEMMAP_META_PATH", None)
        dataset = MemmapImageDataset(config.DATA.DATA_PATH, transform=transform, metadata_path=metadata_path)
    else:
        dataset = ImageFolder(config.DATA.DATA_PATH, transform)

    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    return dataloader
