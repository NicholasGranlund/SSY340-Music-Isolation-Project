import torch


def dice_coeff(pred, target):
    """
    :param pred: (torch.tensor) hard label (in 4D with the batch size) found from the network
    :param target: (torch.tensor) real label (in 4D with the batch size)
    :return: (int) average dice from the batch
    """
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
