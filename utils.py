import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
from torch.autograd import Variable
from .metric_tool import SegEvaluator, ConfuseMatrixMeter
# from .metric_tool import SegEvaluator



def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            for f, g in m.named_children():
                print('initialize: ' + f)
                if isinstance(g, nn.Conv2d):
                    nn.init.kaiming_normal_(g.weight, mode='fan_in', nonlinearity='relu')
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
                elif isinstance(g, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(g.weight)
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
                elif isinstance(g, nn.Linear):
                    nn.init.kaiming_normal_(g.weight, mode='fan_in', nonlinearity='relu')
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
        elif isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.AdaptiveMaxPool2d) or isinstance(m, nn.ModuleList) or isinstance(m, nn.BCELoss):
            a=1
        else:
            pass


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice

def WeightedBCEWithLogitsDiceLoss(inputs, targets):
    pred = torch.where(inputs > 0.5, torch.ones_like(inputs), torch.zeros_like(inputs)).long()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=targets.cpu().numpy())
    scores = salEvalVal.get_scores()
    w_00, w_11 = scores['w0'], scores['w1']
    weight = torch.zeros_like(targets)
    weight = torch.fill_(weight, w_11)
    weight[targets > 0] = w_00
    bce = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # return bce + 1 - dice
    return bce 


def BCE(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    return bce


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = iter
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr  # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))


        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        target = target.long()
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the models.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)



class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        if self.n_classes == 1:
            # 对于二分类，不进行 one-hot 编码，保持原始形状
            return input_tensor
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if self.n_classes > 1 and softmax:
            inputs = torch.softmax(inputs, dim=1)
        elif self.n_classes == 1 and softmax:
            inputs = torch.sigmoid(inputs)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1.0] * self.n_classes
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} shape do not match'

        if self.n_classes == 1:
            # 直接计算二分类的 Dice 损失
            loss = self._dice_loss(inputs, target)
            return loss
        else:
            # 多分类情况下计算每个类别的 Dice 损失并加权平均
            loss = 0.0
            for i in range(self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                loss += dice * weight[i]
            return loss / self.n_classes


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    # target = target.long()
    # if target.dim() == 4:
    #     target = torch.squeeze(target, dim=1)
    # if input.shape[-1] != target.shape[-1]:
    #     input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input, target)


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(WeightedBCEWithLogitsLoss,self).__init__()

    def forward(self, input, target):

        evaluator = SegEvaluator(1)
        evaluator.reset()
        pred = torch.where(torch.sigmoid(input) > 0.5, 1, 0)
        evaluator.add_batch(gt_image=target.cpu().numpy(), pre_image=pred.cpu().numpy())
        w_00,w_11 = evaluator.loss_weight()
        weight1 = torch.zeros_like(target)
        weight1 = torch.fill_(weight1, w_00)
        weight1[target > 0] = w_11
        print(weight1)
        loss = F.binary_cross_entropy_with_logits(input, target,weight=weight1,reduction="mean")

        return loss

class CombinedLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_iou=1.0, smooth=1e-5):
        """
        初始化加权损失函数
        :param weight_bce: BCE 损失的权重
        :param weight_iou: IoU 损失的权重
        :param smooth: 平滑因子，避免除零错误
        """
        super(CombinedLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_iou = weight_iou
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 计算 BCE 损失

    def forward(self, preds, targets):
        """
        计算加权的 BCE 和 IoU 损失
        :param preds: 模型预测值 (batch_size, 1, H, W)，通过 Sigmoid 激活函数计算得到
        :param targets: 真实标签 (batch_size, 1, H, W)，0 表示背景，1 表示前景
        :return: 加权损失
        """
        # 二元交叉熵损失 (BCE)
        bce_loss = self.bce_loss(preds, targets.float())

        # 计算 IoU 损失
        iou_loss = self.iou_loss(preds, targets)

        # 综合损失
        total_loss = self.weight_bce * bce_loss + self.weight_iou * iou_loss
        return total_loss

    def iou_loss(self, preds, targets):
        """
        计算加权的 IoU 损失
        :param preds: 模型预测值 (batch_size, 1, H, W)
        :param targets: 真实标签 (batch_size, 1, H, W)
        :return: IoU 损失
        """
        # 对预测值进行 Sigmoid 激活
        preds = torch.sigmoid(preds)

        # 计算预测和真实标签的交集和并集
        intersection = torch.sum(preds * targets)
        union = torch.sum(preds + targets) - intersection

        # IoU 损失计算：1 - IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou





