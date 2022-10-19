# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os

import numpy as np
import torch
import torch.distributed as dist
from scipy import interpolate


def load_checkpoint(config, model, optimizer, lr_scheduler, scaler, logger):
    logger.info(f">>>>>>>>>> Resuming from {config.MODEL.RESUME} ..........")
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")

    # re-map keys due to name change (only for loading provided models)
    rpe_mlp_keys = [k for k in checkpoint["model"].keys() if "rpe_mlp" in k]
    for k in rpe_mlp_keys:
        checkpoint["model"][k.replace("rpe_mlp", "cpb_mlp")] = checkpoint["model"].pop(
            k
        )

    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)

    max_accuracy = 0.0
    if (
        not config.EVAL_MODE
        and "optimizer" in checkpoint
        and "lr_scheduler" in checkpoint
        and "scaler" in checkpoint
        and "epoch" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])

        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()

        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})"
        )
        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]
        else:
            max_accuracy = 0.0

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(
    config, epoch, model, max_accuracy, optimizer, lr_scheduler, scaler, logger
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "max_accuracy": max_accuracy,
        "epoch": epoch,
        "config": config,
    }

    save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def auto_resume_helper(output_dir, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_pretrained(config, model, logger):
    logger.info(f">>>>>>>>>> Fine-tuned from {config.MODEL.PRETRAINED} ..........")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location="cpu")
    checkpoint_model = checkpoint["model"]

    if any([True if "encoder." in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {
            k.replace("encoder.", ""): v
            for k, v in checkpoint_model.items()
            if k.startswith("encoder.")
        }
        logger.info("Detect pre-trained model, remove [encoder.] prefix.")
    else:
        logger.info("Detect non-pre-trained model, pass without doing anything.")

    if config.MODEL.TYPE in ["swin", "swinv2"]:
        logger.info(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        checkpoint = remap_pretrained_keys_swin(model, checkpoint_model, logger)
    else:
        raise NotImplementedError

    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)

    del checkpoint
    torch.cuda.empty_cache()
    logger.info(f">>>>>>>>>> loaded successfully '{config.MODEL.PRETRAINED}'")


def remap_pretrained_keys_swin(model, checkpoint_model, logger):
    state_dict = model.state_dict()

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.info(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    logger.info(
                        f"{key}: Interpolate relative_position_bias_table using geo."
                    )
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r**n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = (
                            relative_position_bias_table_pretrained[:, i]
                            .view(src_size, src_size)
                            .float()
                            .numpy()
                        )
                        f_cubic = interpolate.interp2d(x, y, z, kind="cubic")
                        all_rel_pos_bias.append(
                            torch.Tensor(f_cubic(dx, dy))
                            .contiguous()
                            .view(-1, 1)
                            .to(relative_position_bias_table_pretrained.device)
                        )

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in checkpoint_model.keys() if "relative_position_index" in k
    ]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [
        k for k in checkpoint_model.keys() if "relative_coords_table" in k
    ]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # re-map keys due to name change
    rpe_mlp_keys = [k for k in checkpoint_model.keys() if "rpe_mlp" in k]
    for k in rpe_mlp_keys:
        checkpoint_model[k.replace("rpe_mlp", "cpb_mlp")] = checkpoint_model.pop(k)

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model
