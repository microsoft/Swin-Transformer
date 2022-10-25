# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .simmim import build_simmim
from .swin_mlp import SwinMLP
from .swin_transformer import SwinTransformer
from .swin_transformer_moe import SwinTransformerMoE
from .swin_transformer_v2 import SwinTransformerV2


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp

            layernorm = amp.normalization.FusedLayerNorm
        except ImportError:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn

        layernorm = nn.LayerNorm

    if is_pretrain:
        model = build_simmim(config)
        return model

    if model_type == "swin":
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            norm_layer=layernorm,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            fused_window_process=config.FUSED_WINDOW_PROCESS,
        )
    elif model_type == "swinv2":
        model = SwinTransformerV2(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWINV2.PATCH_SIZE,
            in_chans=config.MODEL.SWINV2.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWINV2.EMBED_DIM,
            depths=config.MODEL.SWINV2.DEPTHS,
            num_heads=config.MODEL.SWINV2.NUM_HEADS,
            window_size=config.MODEL.SWINV2.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
            qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWINV2.APE,
            patch_norm=config.MODEL.SWINV2.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES,
        )
    elif model_type == "swin_moe":
        model = SwinTransformerMoE(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN_MOE.PATCH_SIZE,
            in_chans=config.MODEL.SWIN_MOE.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN_MOE.EMBED_DIM,
            depths=config.MODEL.SWIN_MOE.DEPTHS,
            num_heads=config.MODEL.SWIN_MOE.NUM_HEADS,
            window_size=config.MODEL.SWIN_MOE.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN_MOE.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN_MOE.QKV_BIAS,
            qk_scale=config.MODEL.SWIN_MOE.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN_MOE.APE,
            patch_norm=config.MODEL.SWIN_MOE.PATCH_NORM,
            mlp_fc2_bias=config.MODEL.SWIN_MOE.MLP_FC2_BIAS,
            init_std=config.MODEL.SWIN_MOE.INIT_STD,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            pretrained_window_sizes=config.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES,
            moe_blocks=config.MODEL.SWIN_MOE.MOE_BLOCKS,
            num_local_experts=config.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS,
            top_value=config.MODEL.SWIN_MOE.TOP_VALUE,
            capacity_factor=config.MODEL.SWIN_MOE.CAPACITY_FACTOR,
            cosine_router=config.MODEL.SWIN_MOE.COSINE_ROUTER,
            normalize_gate=config.MODEL.SWIN_MOE.NORMALIZE_GATE,
            use_bpr=config.MODEL.SWIN_MOE.USE_BPR,
            is_gshard_loss=config.MODEL.SWIN_MOE.IS_GSHARD_LOSS,
            gate_noise=config.MODEL.SWIN_MOE.GATE_NOISE,
            cosine_router_dim=config.MODEL.SWIN_MOE.COSINE_ROUTER_DIM,
            cosine_router_init_t=config.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T,
            moe_drop=config.MODEL.SWIN_MOE.MOE_DROP,
            aux_loss_weight=config.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT,
        )
    elif model_type == "swin_mlp":
        model = SwinMLP(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
            in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
            depths=config.MODEL.SWIN_MLP.DEPTHS,
            num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
            window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN_MLP.APE,
            patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
