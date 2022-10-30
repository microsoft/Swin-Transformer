# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os

import torch
import torch.distributed as dist
from torch._six import inf


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(
        f"==============> Resuming form {config.MODEL.RESUME}...................."
    )
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if (
        not config.EVAL_MODE
        and "optimizer" in checkpoint
        and "lr_scheduler" in checkpoint
        and "epoch" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})"
        )
        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def _hierarchical_weight_k(i):
    return f"head.heads.{i}.weight"


def _hierarchical_bias_k(i):
    return f"head.heads.{i}.bias"


def handle_linear_head(config, model, state_dict, logger):
    """
    Check classifier, if not match, then re-init classifier to zero
    Ways it could not match:

    1. Both have a single linear head, with different number of classes

    2. Pretrained with hierarchical head and are now finetuning with single linear head.
       If the fine-grained head was 10K classes and the linear head has 10K classes,
       and the current dataset is iNat, we use the pre-trained fine-grained head.

    3. Pretrained with single linear head and are finetuning with hierarchical head
       (for instance, if starting with imagenet pre-training, then doing domain-specific
       pre-training on iNat21)

    4. Both have a hierarchical head with different number of tiers/classes.
       We always reinitialize the hierarchical head.
    """

    pretrained_hierarchical = _hierarchical_bias_k(0) in state_dict
    current_hierarchical = config.HIERARCHICAL

    if not pretrained_hierarchical and not current_hierarchical:
        # TESTED because Microsoft wrote this code.
        # Both have a single linear head
        assert "head.bias" in state_dict, "Should have a single pre-trained linear head"
        assert hasattr(model.head, "bias"), "Should have a single random linear head"

        head_bias_pretrained = state_dict["head.bias"]
        num_classes_pretrained = head_bias_pretrained.shape[0]
        num_classes = model.head.bias.shape[0]
        if num_classes_pretrained != num_classes:
            if num_classes_pretrained == 21841 and num_classes == 1000:
                logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
                map22kto1k_path = "data/map22kto1k.txt"
                with open(map22kto1k_path) as f:
                    map22kto1k = f.readlines()
                map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                state_dict["head.weight"] = state_dict["head.weight"][map22kto1k, :]
                state_dict["head.bias"] = state_dict["head.bias"][map22kto1k]
            else:
                torch.nn.init.constant_(model.head.bias, 0.0)
                torch.nn.init.constant_(model.head.weight, 0.0)
                del state_dict["head.weight"]
                del state_dict["head.bias"]
                logger.warning(
                    "Error in loading classifier head, re-init classifier head to 0"
                )
    elif pretrained_hierarchical and not current_hierarchical:
        # UNTESTED
        assert (
            "head.bias" not in state_dict
        ), "Should not have a single pre-trained linear head"
        assert hasattr(model.head, "bias"), "Should have a single random linear head"
        # Increment finegrained level until the key doesn't exist.
        # Then it is the last level in the hierarchical model
        max_level = 0
        while _hierarchical_bias_k(max_level) in state_dict:
            max_level += 1

        finegrained_num_classes_pretrained = state_dict[
            _hierarchical_bias_k(max_level)
        ].shape[0]
        num_classes = model.head.bias.shape[0]
        if num_classes == finegrained_num_classes_pretrained == 10_000:
            # Probably fine-tuning on iNat21
            logger.warn(
                "Assuming that you pre-trained on iNat21 and are now fine-tuning on iNat21."
            )
            state_dict["head.weight"] = state_dict[_hierarchical_weight_k(max_level)]
            state_dict["head.bias"] = state_dict[_hierarchical_bias_k(max_level)]
        else:
            for i in range(max_level):
                del state_dict[_hierarchical_weight_k(i)]
                del state_dict[_hierarchical_bias_k(i)]
            logger.warning(
                "Error in loading classifier head, using default initialization."
            )
    elif not pretrained_hierarchical and current_hierarchical:
        # UNTESTED
        assert "head.bias" in state_dict, "Should have a single pre-trained linear head"
        assert not hasattr(
            model.head, "bias"
        ), "Should not have a single random linear head"

        # Delete the head.bias and head.weight keys then do nothing since the linear
        # layer is already correctly initialized from scratch so it can fine-tune.
        del state_dict["head.weight"]
        del state_dict["head.bias"]

    elif pretrained_hierarchical and current_hierarchical:
        assert (
            "head.bias" not in state_dict
        ), "Should not have a single pre-trained linear head"
        assert not hasattr(
            model.head, "bias"
        ), "Should not have a single random linear head"

        # Check if the two models have the exact same number of levels, and the same
        # number of classes in each level
        matches = True
        level = 0
        while matches and _hierarchical_bias_k(level) in state_dict:
            # Check that the current model has the right attribute
            if level > len(model.head.heads):
                matches = False
                continue

            if not hasattr(model.head.heads[level], "bias"):
                matches = False
                continue

            if (
                model.head.heads[level].bias.shape
                != state_dict[_hierarchical_bias_k(level)].shape
            ):
                matches = False
                continue

            if (
                model.head.heads[level].weight.shape
                != state_dict[_hierarchical_weight_k(level)].shape
            ):
                matches = False
                continue

            level += 1

        if not matches:
            # UNTESTED
            logger.warning(
                "Not using pre-trained hierarchical head because the shapes do not match."
            )
            # Delete the keys from the state dict because the pre-trained model and
            # the current model do not match in size.
            for i in range(level):
                del state_dict[_hierarchical_weight_k(i)]
                del state_dict[_hierarchical_bias_k(i)]
        else:
            logger.info("Using pre-trained hierarchical head.")


def load_pretrained(config, model, logger):
    logger.info(
        f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......"
    )
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location="cpu")
    state_dict = checkpoint["model"]

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_position_index" in k
    ]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_coords_table" in k
    ]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k
    ]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                relative_position_bias_table_pretrained_resized = (
                    torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(
                            1, nH1, S1, S1
                        ),
                        size=(S2, S2),
                        mode="bicubic",
                    )
                )
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2
                ).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [
        k for k in state_dict.keys() if "absolute_pos_embed" in k
    ]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                    -1, S1, S1, C1
                )
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                    0, 3, 1, 2
                )
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode="bicubic"
                )
                absolute_pos_embed_pretrained_resized = (
                    absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                )
                absolute_pos_embed_pretrained_resized = (
                    absolute_pos_embed_pretrained_resized.flatten(1, 2)
                )
                state_dict[k] = absolute_pos_embed_pretrained_resized

    handle_linear_head(config, model, state_dict, logger)

    msg = model.load_state_dict(state_dict, strict=False)
    for key in msg.missing_keys:
        assert (
            "relative_coords_table" in key
            or "relative_position_index" in key
            or "attn_mask" in key
        ), f"Should only reinitialize relative positional information, not '{key}'"
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(
    config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_accuracy": max_accuracy,
        "scaler": loss_scaler.state_dict(),
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


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
    print(f"All checkpoints found in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def batch_size(tensor_or_list):
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.size(0)
    elif isinstance(tensor_or_list, list):
        sizes = [tensor.size(0) for tensor in tensor_or_list]
        assert all(size == sizes[0] for size in sizes)
        return sizes[0]


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
