import torch
from functools import reduce


def dice_coeff(pred, target):
    """
    :param pred: (torch.tensor) hard label (in 4D with the batch size) found from the network
    :param target: (torch.tensor) real label (in 4D with the batch size)
    :return: (int) average dice from the batch
    """
    intersection = (pred == target).type(torch.int).sum()
    union = reduce(lambda x, y: x * y, pred.shape, 1)
    return (intersection) / union

