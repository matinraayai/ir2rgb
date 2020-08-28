import os
import torch
from torch.nn import Module


def save_network(network: Module, network_label, epoch_label, save_dir: str):
    save_filename = f'{epoch_label}_net_{network_label}.pth'
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.state_dict(), save_path)


def concat(tensors, dim=0):
    if tensors[0] is not None and tensors[1] is not None:
        if isinstance(tensors[0], list):
            tensors_cat = []
            for i in range(len(tensors[0])):
                tensors_cat.append(concat([tensors[0][i], tensors[1][i]], dim=dim))
            return tensors_cat
        return torch.cat([tensors[0], tensors[1]], dim=dim)
    elif tensors[0] is not None:
        return tensors[0]
    else:
        return tensors[1]
