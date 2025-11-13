import torch.nn.functional as F

# Multi-Scale BCEDice Loss
# BCEDiceLoss1 + BCEDiceLoss2 + BCEDiceLoss3 + BCEDiceLoss4

def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice
