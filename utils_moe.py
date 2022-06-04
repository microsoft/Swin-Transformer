# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist


def split_moe_model_state_dict(moe_keys, model_state_dict):
    moe_model_state_dict = {}
    non_moe_model_state_dict = {}
    for (k, v) in model_state_dict.items():
        if k in moe_keys:
            moe_model_state_dict[k] = v
        else:
            non_moe_model_state_dict[k] = v
    return moe_model_state_dict, non_moe_model_state_dict


def merge_moe_model_state_dict(moe_model_state_dict, non_moe_model_state_dict):
    model_state_dict = {}
    model_state_dict.update(moe_model_state_dict)
    model_state_dict.update(non_moe_model_state_dict)
    return model_state_dict


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    global_rank = dist.get_rank()
    logger.info(f"==============> Rank[{global_rank}] Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.endswith(f'.pth'):
        if config.TRAIN.MOE.SAVE_MASTER:
            resume_path = config.MODEL.RESUME + f'.global'
        else:
            resume_path = config.MODEL.RESUME + f'.rank{global_rank}'
        logger.info(f"===> Rank[{global_rank}] Re-formatting checkpoint name to {resume_path}......")
    else:
        resume_path = config.MODEL.RESUME

    checkpoint = torch.load(resume_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=>Rank[{global_rank}] loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    global_rank = dist.get_rank()
    logger.info(f"==============> Rank[{global_rank}] Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    if config.MODEL.PRETRAINED.endswith(f'.pth'):
        if config.TRAIN.MOE.SAVE_MASTER:
            pretrained_path = config.MODEL.PRETRAINED + f'.global'
        else:
            pretrained_path = config.MODEL.PRETRAINED + f'.rank{global_rank}'
        logger.info(f"===> Rank[{global_rank}] Re-formatting checkpoint name to {pretrained_path}......")
    else:
        pretrained_path = config.MODEL.PRETRAINED

    if pretrained_path.endswith(f'.rank{global_rank}'):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if os.path.exists(pretrained_path.replace(f'.rank{global_rank}', f'.master')):
            checkpoint_master = torch.load(pretrained_path.replace(f'.rank{global_rank}', f'.master'),
                                           map_location='cpu')
            state_dict = merge_moe_model_state_dict(checkpoint['model'], checkpoint_master['model'])
        else:
            state_dict = checkpoint['model']
    elif pretrained_path.endswith(f'.pth.global'):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint['model']
    else:
        raise NotImplementedError(f"{config.MODEL.PRETRAINED} file error...")

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
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
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
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
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger,
                    zero_redundancy=False):
    global_rank = dist.get_rank()

    if zero_redundancy:
        if config.TRAIN.MOE.SAVE_MASTER:
            save_state = {'model': model.state_dict()}
            if global_rank == 0:
                save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth.global')
                logger.info(f"{save_path} saving......")
                torch.save(save_state, save_path)
                logger.info(f"{save_path} saved !!!")
        else:
            moe_model_state_dict, non_moe_model_state_dict = \
                split_moe_model_state_dict(model._ddp_params_and_buffers_to_ignore, model.state_dict())
            save_state = {'model': moe_model_state_dict}
            save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth.rank{global_rank}')
            logger.info(f"{save_path} saving......")
            torch.save(save_state, save_path)
            logger.info(f"{save_path} saved !!!")
            if global_rank == 0:
                save_state_master = {'model': non_moe_model_state_dict}
                save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth.master')
                logger.info(f"{save_path} saving......")
                torch.save(save_state_master, save_path)
                logger.info(f"{save_path} saved !!!")
    else:
        save_state = {'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'lr_scheduler': lr_scheduler.state_dict(),
                      'max_accuracy': max_accuracy,
                      'scaler': loss_scaler.state_dict(),
                      'epoch': epoch,
                      'config': config}
        if config.TRAIN.MOE.SAVE_MASTER:
            if global_rank == 0:
                save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth.global')
                logger.info(f"{save_path} saving......")
                torch.save(save_state, save_path)
                logger.info(f"{save_path} saved !!!")
        else:
            save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth.rank{global_rank}')
            logger.info(f"{save_path} saving......")
            torch.save(save_state, save_path)
            logger.info(f"{save_path} saved !!!")


def auto_resume_helper(output_dir, save_master=False):
    global_rank = dist.get_rank()
    checkpoints = os.listdir(output_dir)
    if not save_master:
        master_checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith(f'pth.rank0')]
    else:
        master_checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith(f'pth.global')]
    print(f"All master checkpoints founded in {output_dir}: {master_checkpoints}")
    if len(master_checkpoints) > 0:
        latest_master_checkpoint = max([os.path.join(output_dir, d) for d in master_checkpoints], key=os.path.getmtime)
        latest_checkpoint = latest_master_checkpoint.replace('pth.rank0', f'pth.rank{global_rank}')
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def hook_scale_grad(scale, tensor):
    return tensor / scale
