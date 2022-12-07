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

import wandb


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


class WandbWriter:
    def __init__(self, rank):
        self.rank = rank

    def init(self, config):
        if self.rank != 0:
            return

        kwargs = dict(
            config=config,
            project="hierarchical-vision",
            name=config.EXPERIMENT.NAME,
        )

        if not config.EXPERIMENT.WANDB_ID:
            print("Cannot resume wandb run because no id was provided!")
        else:
            kwargs["id"] = config.EXPERIMENT.WANDB_ID
            kwargs["resume"] = "allow"

        wandb.init(**kwargs, mode="disabled")

        # Validation metrics
        wandb.define_metric("val/loss", step_metric="epoch", summary="max")
        wandb.define_metric("val/acc1", step_metric="epoch", summary="max")
        wandb.define_metric("val/acc5", step_metric="epoch", summary="max")

        # Training metrics
        wandb.define_metric("train/batch_time", step_metric="step", summary="last")
        wandb.define_metric("train/grad_norm", step_metric="step", summary="last")
        wandb.define_metric("train/batch_loss", step_metric="step", summary="last")
        wandb.define_metric("train/loss_scale", step_metric="step", summary="last")
        wandb.define_metric("train/learning_rate", step_metric="step", summary="last")

        wandb.define_metric("train/epoch_time", step_metric="epoch", summary="last")
        wandb.define_metric("train/loss", step_metric="epoch", summary="last")

        # Other metrics
        wandb.define_metric("memory_mb", summary="max")

    def log(self, dct):
        if self.rank != 0:
            return

        wandb.log(dct)

    @property
    def name(self):
        if self.rank != 0:
            raise RuntimeError(f"Should not get .name with rank {self.rank}.")

        return wandb.run.name


#############################  Added Script to generate yaml files  #################################

def init(name: str, verbose: bool = False, date=True) -> logging.Logger:
    if date:
        log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    else:
        log_format = "[%(levelname)s] [%(name)s] %(message)s"

    if not verbose:
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logging.basicConfig(level=logging.DEBUG, format=log_format)

    return logging.getLogger(name)
