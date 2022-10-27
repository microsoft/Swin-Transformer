# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import functools
import logging
import os
import sys

import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=""):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"log_rank{dist_rank}.txt"), mode="a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger


class TensorboardWriter:
    writer = None

    def __init__(self, output_dir, dist_rank):
        self.output_dir = output_dir

        if dist_rank == 0:
            self.writer = SummaryWriter(log_dir=self.output_dir)

    def add_hparams(self, hparams, metrics):
        pass

    def log(self, items, step):
        if self.writer is None:
            return

        # Copied from huggingface/accelerate/src/accelerate/tracking.py
        for k, v in items.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step)
            elif isinstance(v, torch.Tensor):
                assert v.numel() == 1
                self.writer.add_scalar(k, v.item(), global_step=step)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step)
            elif isinstance(v, dict):
                self.writer.add_scalars(k, v, global_step=step)
            else:
                print(f"Can't log {v} because it is {type(v)}!")


class CorrectedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)
