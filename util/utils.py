import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataset.data import MirrorDataset,test_dataset,SalObjSTDataset
from pathlib import Path
from tensorboardX import SummaryWriter
import time
import logging
import sys
from util import statistic
import numpy as np
def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    ret = net.load_state_dict(state['net'])
    print(ret)
    optimizer.load_state_dict(state['opt_1'])

@torch.no_grad()
def pred_unlabel(net, pred_loader, batch_size, epoch):
    unimg, unlab, unmask, labs, undepth = [], [], [], [], []
    plab_dice = 0
    drop_sample_count = 0
    net.eval()
    for (step, data) in enumerate(pred_loader):
        img, depth = data
        img, depth = img.cuda(), depth.cuda()
        out = net(img, depth)
        thres = 0.5
        plab0 = get_mask(out[1], thres=thres)
        plab1 = get_mask(out[0], thres=thres)
        mask = (plab0 == plab1).long()
        plab = plab1
        # plab = (plab0 == plab3)
        ratio = mask.sum().item() / ( mask.shape[1] * mask.shape[2])
        if ratio > 0.85:
            unimg.append(img)
            undepth.append(depth)
            unlab.append(plab)
            unmask.append(mask)
        else:
            drop_sample_count += 1
    new_loader = DataLoader(SalObjSTDataset(unimg, unlab, unmask, undepth, trainsize=256), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return new_loader, drop_sample_count, plab_dice

def to_cuda(tensors, device=None):
    res = []
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            res.append(to_cuda(t, device))
        return res
    elif isinstance(tensors, (dict,)):
        res = {}
        for k, v in tensors.items():
            res[k] = to_cuda(v, device)
        return res
    else:
        if isinstance(tensors, torch.Tensor):
            if device is None:
                return tensors.cuda()
            else:
                return tensors.to(device)
        else:
            return tensors

def get_current_consistency_weight(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

def get_model_and_dataloader(cfg):
    cfg_dataset = cfg["dataset"]

    train_size=cfg_dataset['train']['trainsize']
    data_root = cfg_dataset['train']['data_root']
    batch_size = cfg_dataset['batch_size']
    u_batch_size = int(cfg_dataset['batch_size'] * cfg_dataset['ration'])
    strong_aug_nums = cfg_dataset['strong_aug']['num_augs']
    flag_use_rand_num = cfg_dataset['strong_aug']['flag_use_random_num_sampling']

    trainset_lab = MirrorDataset(data_root,trainsize=train_size,split='label', strong_aug_nums=strong_aug_nums, flag_use_rand_num=flag_use_rand_num)
    lab_loader = DataLoader(trainset_lab, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    trainset_unlab = MirrorDataset(data_root,trainsize=train_size,split='unlabel', strong_aug_nums=strong_aug_nums, flag_use_rand_num=flag_use_rand_num)
    unlab_loader = DataLoader(trainset_unlab, batch_size=u_batch_size, shuffle=True, num_workers=0, pin_memory=True,drop_last=True)

    test_root = cfg_dataset['val']['data_root']
    test_image_root=test_root + '/RGB/'
    test_gt_root=test_root + '/GT/'
    test_depth_root=test_root + '/depth/'
    test_loader=test_dataset(test_image_root, test_gt_root,test_depth_root, train_size)

    return lab_loader, unlab_loader, test_loader

def config_log(save_path, tensorboard=False):
    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S')) if tensorboard else None
    save_path = str(Path(save_path) / 'log.txt')
    formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s')
    logger = logging.getLogger(save_path.split('\\')[-2])
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(save_path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    sh = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, writer

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        return self

class Measures():
    def __init__(self, keys, writer, logger):
        self.keys = keys
        self.measures = {k: AverageMeter() for k in self.keys}
        self.writer = writer
        self.logger = logger

    def reset(self):
        [v.reset() for v in self.measures.values()]

def get_mask(out, thres=0.5):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).float()
    masks = masks[:, 1, :, :]#B H W
    return masks

class PretrainMeasures(Measures):
    def __init__(self, writer, logger):
        keys = ['loss_ce', 'loss_dice', 'loss_con', 'loss_rad', 'loss_all', 'train_dice']
        super(PretrainMeasures, self).__init__(keys, writer, logger)

    def update(self, out, lab, *args):
        args = list(args)
        masks = get_mask(out)
        train_dice = statistic.dice_ratio(masks, lab)
        args.append(train_dice)

        dict_variables = dict(zip(self.keys, args))
        for k, v in dict_variables.items():
            self.measures[k].update(v)

    def log(self, epoch, step):
        log_string, params = 'Epoch : {}', []
        for k in self.keys:
            log_string += ', ' + k + ': {:.4f}'
            params.append(self.measures[k].val)
        self.logger.info(log_string.format(epoch, *params))

        for k, measure in self.measures.items():
            k = 'pretrain/' + k
            self.writer.add_scalar(k, measure.avg, step)
        self.writer.flush()

def save_net_opt(net, optimizer_1, optimizer_2, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt_1': optimizer_1.state_dict(),
        'opt_2':optimizer_2.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr
