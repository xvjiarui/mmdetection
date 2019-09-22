import torch


def convert_non_inplace(module):
    for m in module.modules():
        if hasattr(m, 'inplace'):
            m.inplace = False
