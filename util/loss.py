import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass


class DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def compute_unsupervised_loss_conf_weight(predict, target, percent, pred_teacher):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        conf, ps_label = torch.max(prob, dim=1)
        conf = conf.detach()
        conf_thresh = np.percentile(
            conf[target != 255].cpu().numpy().flatten(), 100 - percent
        )
        thresh_mask = conf.le(conf_thresh).bool() * (target != 255).bool()
        conf[thresh_mask] = 0
        target[thresh_mask] = 255
        weight = batch_size * h * w / (torch.sum(target != 255) + 1e-6)

    loss_ = weight * F.cross_entropy(predict, target, ignore_index=255, reduction='none')  # [10, 321, 321]
    ## v1
    # loss = torch.mean(conf * loss_)
    ## v2
    # conf = conf / conf.sum() * (torch.sum(target != 255) + 1e-6)
    # loss = torch.mean(conf * loss_)
    ## v3
    conf = (conf + 1.0) / (conf + 1.0).sum() * (torch.sum(target != 255) + 1e-6)
    loss = torch.mean(conf * loss_)
    return loss


def compute_unsupervised_Bloss_conf_weight(predict, target, epoch, warm_epoch):
    batch_size, num_class, h, w = predict.shape
    target = target.float()
    # with torch.no_grad():
    #     # drop pixels with high entropy
    #     # prob = torch.softmax(pred_teacher, dim=1)
    #     conf = torch.sigmoid(pred_teacher)
    #     # conf, ps_label = torch.max(prob, dim=1)
    #     conf = conf.detach()
    #     conf_thresh = np.percentile(
    #         conf[target != 255].cpu().numpy().flatten(), 100 - percent
    #     )
    #     thresh_mask = conf.le(conf_thresh).bool() * (target != 255).bool()
    #     conf[thresh_mask] = 0
    #     target[thresh_mask] = 255
    #     weight = batch_size * h * w / (torch.sum(target != 255) + 1e-6)
    #     # weight = weight.float()
    current = np.clip(epoch, 0.0, warm_epoch)
    phase = 1.0 - current / warm_epoch
    weight = float(np.exp(-5.0 * phase * phase)) * 0.1
    use_target = (target != 255 )
    loss_ = weight * F.binary_cross_entropy_with_logits(predict, target, reduction='none') * use_target  # [10, 321, 321]
    loss = torch.mean(loss_)
    return loss

def compute_L_UL_loss(loss_type, pred, gt, confidence = 1, is_label = True, unlabel_weigt = 0.1):
    assert isinstance(pred,list)
    assert isinstance(gt, list)
    assert len(pred) == len(gt)
    loss = 0
    if is_label:
        for i in range(len(pred)):
            # print(pred[i].shape)
            loss = loss + loss_type(pred[i], gt[i])
        loss = loss / len(pred)
        return loss
    else:
        for i in range(len(pred)):
            loss = loss + loss_type(pred[i] * confidence[i], gt[i])
        loss = loss / len(pred)
        return loss*unlabel_weigt

class StructLoss(nn.Module):
    def __init__(self):
        super(StructLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

