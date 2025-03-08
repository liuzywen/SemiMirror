import copy

import torch

import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
# from models.Swin_Transformer import SwinTransformer,SwinNet
import yaml
from dataset.data import test_dataset
from model.model_helper import ModelBuilder
import time
parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config/sal_config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)
parser.add_argument('--testsize', default=416, type=int)
parser.add_argument('--teacher', default=False, type=bool)
opt = parser.parse_args()

args = parser.parse_args()
cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

'''创建模型'''
model = ModelBuilder(cfg["net"])
if opt.teacher:
    checkpoint = torch.load(cfg['test']['checkpoint_root'])['teacher_state']
else:
    checkpoint = torch.load(cfg['test']['checkpoint_root'])['model_state']
model.load_state_dict(checkpoint, strict=True)
model.cuda()

res_dir = './result/RGBD_Mirror/'
model.eval()
#
dataset_path = cfg['test']['data_root']
# test

test_datasets = ['test']
fps = 0
for dataset in test_datasets:
    save_path = './test_maps/semi_RGBD/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    maesum = 0
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        depth = depth.repeat(1,3,1,1)
        outs= model(image,depth, name)
        out = outs['pred']
        res = F.softmax(out, dim=1)
        res = torch.max(res, dim=1)[1].float()
        res = res.unsqueeze(1)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.squeeze(1)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        maesum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res*255)
    mae_rgb = maesum / test_loader.size
    print('Test Done! mae:', mae_rgb)

