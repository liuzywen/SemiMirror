import numpy as np
import scipy.ndimage as nd

import torch
import torch.nn as nn
import torch.nn.functional as F

# from engine.logger import get_logger

# logger = get_logger()

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, reduction='mean', ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        if weight:
            self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                                 reduction=reduction, ignore_index=ignore_index)
        else:
            self.loss = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, input, target):
        return self.loss((1 - F.softmax(input, 1))**2 * F.log_softmax(input, 1), target)


class FocalMSE(nn.Module):
    def __init__(self, gamma=2):
        super(FocalMSE, self).__init__()
        self.gamma = gamma

        self.loss = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss_no_reduction = self.loss(pred, target)
        weight = (1-pred)**self.gamma
        weighted_loss = torch.mean(loss_no_reduction * weight)
        return weighted_loss


class RCELoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', weight=None, class_num=37, beta=0.01):
        super(RCELoss, self).__init__()
        self.beta = beta
        self.class_num = class_num
        self.ignore_label = ignore_index
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)
        self.criterion2 = nn.NLLLoss(reduction='none', ignore_index=ignore_index, weight=weight)

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        max_pred, max_id = torch.max(pred, dim=1)		# pred (b, h, w)
        target_flat = target.view(b, 1, h, w)
        mask = (target_flat.ne(self.ignore_label)).float()
        target_flat = (mask * target_flat.float()).long()
        # convert to onehot
        label_pred = torch.zeros(b, self.class_num, h, w).cuda().scatter_(1, target_flat, 1)
        # print(label_pred.shape, max_id.shape)

        prob = torch.exp(pred)
        prob = F.softmax(prob, dim=1)      # i add this

        weighted_pred = F.log_softmax(pred, dim=1)
        loss1 = self.criterion(weighted_pred, target)

        label_pred = torch.clamp(label_pred, min=1e-9, max=1.0-1e-9)

        label_pred = torch.log(label_pred)
        loss2 = self.criterion2(label_pred, max_id)
        loss2 = torch.mean(loss2*mask)
        # print(loss1, loss2)
        loss = loss1 + self.beta*loss2
        # print(loss1, loss2)
        # print(loss)
        return loss

class BalanceLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', weight=None):
        super(BalanceLoss, self).__init__()
        self.ignore_label = ignore_index
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)

    def forward(self, pred, target):
        # prob = torch.exp(pred)
        # # prob = F.softmax(prob, dim=1)      # i add this
        # weighted_pred = pred * (1 - prob) ** 2
        # loss = self.criterion(weighted_pred, target)

        prob = torch.exp(pred)
        prob = F.softmax(prob, dim=1)      # i add this
        weighted_pred = F.log_softmax(pred, dim=1) * (1 - prob) ** 2
        loss = self.criterion(weighted_pred, target)
        return loss

class berHuLoss(nn.Module):
    def __init__(self, delta=0.2, ignore_index=0, reduction='mean'):
        super(berHuLoss,self).__init__()
        self.delta = delta
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        valid_mask = (1 - target.eq(self.ignore_index)).float()
        valid_delta = torch.abs(pred - target) * valid_mask
        max_delta = torch.max(valid_delta)
        delta = self.delta * max_delta

        f_mask = (1 - torch.gt(target, delta)).float() * valid_mask
        s_mask = (1 - f_mask ) * valid_mask
        f_delta =  valid_delta * f_mask
        s_delta = ((valid_delta **2) + delta **2) / (2 * delta) * s_mask

        loss = torch.mean(f_delta + s_delta)
        return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, ignore_label, gamma=2.0, alpha=0.25,
                 reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, h, w = target.size()
        pred = pred.view(b, -1, 1)
        pred_sigmoid = pred.sigmoid()
        target = target.view(b, -1).float()
        mask = (target.ne(self.ignore_label)).float()
        target = mask * target
        onehot = target.view(b, -1, 1)

        max_val = (-pred_sigmoid).clamp(min=0)

        pos_part = (1 - pred_sigmoid) ** self.gamma * (
                pred_sigmoid - pred_sigmoid * onehot)
        neg_part = pred_sigmoid ** self.gamma * (max_val + (
                (-max_val).exp() + (-pred_sigmoid - max_val).exp()).log())

        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(
            dim=-1) * mask
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.5, 0.5]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        # print('1',pred.size())
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)#将~valid_mask中为True，prob的位置置为1
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]#根据每个像素的类别标签，选择对应的概率值
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()#返回mask_prob排序后的序列
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)     # 概率小于阈值的挖出来
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

def bce2d(input, target):
    b, c, h, w = input.size()

    log_p = input.permute(0, 2, 3, 1).contiguous().view(-1)      # (b, h, w, c) ==> (b, -1)
    target = target.view(-1)

    pos_index = (target == 1)
    neg_index = (target == 0)
    ignore_index = (target > 1)

    weight = torch.zeros(log_p.size()).cuda().float()
    pos_num = pos_index.sum().float()
    neg_num = neg_index.sum().float()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    weight[ignore_index] = 0
    # print(weight.max(), pos_num, neg_num)

    loss = F.binary_cross_entropy_with_logits(log_p, target.float(), weight, reduction='mean')
    return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _one_hot_mask_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # print(inputs.size(), target.size())
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            mask = self._one_hot_mask_encoder(mask)
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes


class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=256, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

class CriterionOhem(nn.Module):
    def __init__(
        self,
        aux_weight,
        thresh=0.7,
        min_kept=100000,
        ignore_index=255,
        use_weight=False,
    ):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh, min_kept, use_weight
        )
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss

def compute_unsupervised_loss_by_threshold(predict, target, logits, thresh=0.95):
    batch_size, num_class, h, w = predict.shape
    thresh_mask = logits.ge(thresh).bool() * (target != 255).bool()
    target[~thresh_mask] = 255
    loss = F.cross_entropy(predict, target, ignore_index=255, reduction="none")
    return loss.mean(), thresh_mask.float().mean()
