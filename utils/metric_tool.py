import numpy as np


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x + 1e-6) ** -1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    tp = hist[1, 1]
    fn = hist[1, 0]
    fp = hist[0, 1]
    tn = hist[0, 0]
    # recall
    recall = tp / (tp + fn + np.finfo(np.float32).eps)
    # precision
    precision = tp / (tp + fp + np.finfo(np.float32).eps)
    # F1 score
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    return f1


def cm2score(confusion_matrix):
    hist = confusion_matrix
    tp = hist[1, 1]
    fn = hist[1, 0]
    fp = hist[0, 1]
    tn = hist[0, 0]
    # acc
    oa = (tp + tn) / (tp + fn + fp + tn + np.finfo(np.float32).eps)
    # recall
    recall = tp / (tp + fn + np.finfo(np.float32).eps)
    # precision
    precision = tp / (tp + fp + np.finfo(np.float32).eps)
    # F1 score
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    # IoU
    iou = tp / (tp + fp + fn + np.finfo(np.float32).eps)
    # pre
    pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
    # kappa
    kappa = (oa - pre) / (1 - pre)
    # w_00,w_11
    w_00 = tp / (tp + fp + fn)
    w_11 = tn / (tn + fn + fp)
    score_dict = {'Kappa': kappa, 'IoU': iou, 'F1': f1, 'OA': oa, 'recall': recall, 'precision': precision, 'Pre': pre, 'w0': w_00, 'w1': w_11}
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""

    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix

class SegEvaluator:
    def __init__(self, class_num=4):
        if class_num == 1:
            class_num = 2
        self.num_class = class_num
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def kappa(self,OA):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = np.sum(self.confusion_matrix)
        pe = np.dot(pe_rows, pe_cols) / (sum_total ** 2)
        #po = self.pixel_oa()
        po = OA
        return (po - pe) / (1 - pe)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix


    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.mat=self.confusion_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def loss_weight(self):
        TN = self.confusion_matrix[0][0]
        FP = self.confusion_matrix[0][1]
        FN = self.confusion_matrix[1][0]
        TP = self.confusion_matrix[1][1]
        w_00 = TP / (TP + FP + FN)
        w_11 = TN / (TN + FN + FP)
        return w_00, w_11

    def matrix(self,class_index):
        metric = {}
        precision_cls = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        recall_cls = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        OA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        iou_per_class = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        metric['0_IoU'] = iou_per_class[0]
        metric['1_IoU'] = iou_per_class[1]
        metric['IoU'] = np.nanmean(iou_per_class)
        metric['Precision'] = precision_cls[class_index]  #precision / self.num_class
        metric['Recall'] = recall_cls[class_index]          #recall / self.num_class
        metric['OA'] = OA
        metric['F1'] = (2 * precision_cls[class_index] * recall_cls[class_index]) / (precision_cls[class_index] + recall_cls[class_index])
        Kappa = self.kappa(OA)
        metric['Kappa'] = Kappa
        return metric
