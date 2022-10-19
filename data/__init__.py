from .build import build_loader as _build_loader
from .data_simmim_ft import build_loader_finetune
from .data_simmim_pt import build_loader_simmim


def build_loader(config, simmim=False, is_pretrain=False):
    if not simmim:
        return _build_loader(config)
    if is_pretrain:
        return build_loader_simmim(config)
    else:
        return build_loader_finetune(config)
