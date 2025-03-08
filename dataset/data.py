import math

import torch.utils.data as data
import os
# from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
from dataset.randaugment import RandomAugment
from copy import deepcopy
import torch
from PIL import Image, ImageOps, ImageFilter
from . import augs_TIBA as img_trsform

def cv_random_flip(img, label, depth,edge):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        # label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if label is not None:
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            # edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            label = None
            # edge = None

        if edge is not None:
            # label = label.transpose(Image.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # label = None
            edge = None
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    #     if label is not None:
    #         label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     else:
    #         label = None
    #     if edge is not None:
    #         edge = edge.transpose(Image.FLIP_TOP_BOTTOM)
    #     else:
    #         edge = None

    return img, label, depth, edge


def randomCrop(image, label, depth, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    if label is not None:
        if edge is not None:
            return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge.crop(random_region)
        else:
            return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge
    else:
        return image.crop(random_region), label, depth.crop(random_region), edge


def randomCrop_unlabel(image, label, depth, edge):
    border = 60
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    if label is not None:
        if edge is not None:
            return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge.crop(random_region)
        else:
            return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge
    else:
        return image.crop(random_region), label, depth.crop(random_region), edge


def randomRotation(image, label, depth, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        if label is not None:
            label = label.rotate(random_angle, mode)
        if edge is not None:
            edge = edge.rotate(random_angle, mode)
    return image, label, depth, edge

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)



def normalize(img, depth=None, mask=None, trainsize=256):
    img = transforms.Compose([
        transforms.Resize((trainsize, trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if depth is not None:
        depth = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(depth)
    if mask is not None:
        # mask = torch.from_numpy(np.array(mask))
        mask = transforms.Compose([transforms.Resize((trainsize, trainsize)),transforms.ToTensor()])(mask)
        # x = np.array(mask)
        return img, depth, mask.long()
    return img, depth


def resize(img, depth, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))
    # print('x', w, h ,long_side)
    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    depth = depth.resize((ow, oh), Image.BILINEAR)
    if mask is not None:
        mask = mask.resize((ow, oh), Image.NEAREST)
    return img, depth, mask


def crop(img, depth, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)
    if mask is not None:
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    # masks =
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    depth = depth.crop((x, y, x + size, y + size))
    if mask is not None:
        mask = mask.crop((x, y, x + size, y + size))

    return img, depth, mask


def hflip(img, depth, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        if mask is not None:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, depth, mask

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class MirrorDataset(data.Dataset):
    def __init__(self, image_root,trainsize,split, strong_aug_nums = 2, flag_use_rand_num = True):
        self.trainsize = trainsize
        self.split=split
        self.choice = ''

        if self.split == 'label':
            data_path=image_root+'/'+ self.split+'/RGB/'
            image_root=image_root+'/'+self.split
        else:
            data_path = image_root + '/' + self.split +'/' + self.choice + '/RGB/'  # split='label' 'unlabel'
            image_root = image_root + '/' + self.split +'/' + self.choice
        print('dataset:',data_path)

        self.index=[i_index for i_index in os.listdir(data_path)]
        self.images=['/'.join([image_root,'RGB',i_index]) for i_index in self.index]

        if self.split == 'label':
            self.gts=['/'.join([image_root,'GT',i_index[:-4]+'.png']) for i_index in self.index]
            self.depths = []
            for i_index in self.index:
                if os.path.isfile(image_root + '/depth/' + i_index.strip()[:-4] + '.bmp'):
                    self.depths.append('/'.join([image_root, 'depth', i_index.strip()[:-4] + '.bmp']))
                elif os.path.isfile(image_root + '/depth/' + i_index.strip()[:-4] + '.png'):
                    self.depths.append('/'.join([image_root, 'depth', i_index.strip()[:-4] + '.png']))
            self.edges = [None for i_index in self.index]
        else:
            self.gts = [None for i_index in self.index]
            self.depths = []
            for i_index in self.index:
                if os.path.isfile(image_root + '/depth/' + i_index.strip()[:-4] + '.bmp'):
                    self.depths.append('/'.join([image_root, 'depth', i_index.strip()[:-4] + '.bmp']))
                elif os.path.isfile(image_root + '/depth/' + i_index.strip()[:-4] + '.png'):
                    self.depths.append('/'.join([image_root, 'depth', i_index.strip()[:-4] + '.png']))
            assert len(self.depths) == len(self.images)

            self.edges = [None for i_index in self.index]

        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        if self.gts[0] is not None:
            self.gts = sorted(self.gts)
        if self.edges[0] is not None:
            self.edges = sorted(self.edges)
        self.filter_files()
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
        self.edges_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()]
        )
        self.img_transform_strong = img_trsform.strong_img_aug(strong_aug_nums,
        flag_using_random_num=flag_use_rand_num)


    def __getitem__(self, index):

        image = self.rgb_loader(self.images[index % self.size])
        depth = self.binary_loader(self.depths[index % self.size])
        if self.gts[index % self.size] is not None:
            gt = self.binary_loader(self.gts[index % self.size])
        else:
            gt = None
        if self.edges[index % self.size] is not None:
            edge = self.binary_loader(self.edges[index % self.size])
        else:
            edge = None
        if self.split == 'label':
            image, gt, depth, edge = cv_random_flip(image, gt, depth, edge)
            image, gt, depth, edge = randomCrop(image, gt, depth, edge)
            image, gt, depth, edge = randomRotation(image, gt, depth, edge)

            depth_strong = deepcopy(depth)
            image_strong = deepcopy(image)

            depth = self.depths_transform(depth)
            image = self.img_transform(image)

            image_strong, depth_strong = self.img_transform_strong(image_strong, depth_strong)
            image_strong = self.img_transform(image_strong)
            depth_strong = self.depths_transform(depth_strong)

            gt = self.gt_transform(gt)
            name = self.images[index % self.size].split('/')[-1]
            if name.endswith('.jpg'):
                name = name.split('.jpg')[0] + '.png'
            return {'image':image, 'gt':gt.long(), 'depth':depth, 'name': name,
                    'image_strong': image_strong, 'depth_strong':depth_strong}
        else:
            image, gt, depth, edge = cv_random_flip(image, gt, depth, edge)
            image, gt, depth, edge = randomCrop(image, gt, depth, edge)
            image, gt, depth, edge = randomRotation(image, gt, depth, edge)
            depth_strong = deepcopy(depth)
            depth = self.depths_transform(depth)
            image_strong = deepcopy(image)
            image_weak = self.img_transform(image)
            image_strong,_ = self.img_transform_strong(image_strong, depth_strong)
            image_strong = self.img_transform(image_strong)
            depth_strong = self.depths_transform(depth_strong)
            name = self.images[index % self.size].split('/')[-1]
            if name.endswith('.jpg'):
                name = name.split('.jpg')[0]+'.png'

        if gt is not None:
            gt = self.gt_transform(gt)
            if edge is not None:
                edge = self.edges_transform(edge)
                return {'image_weak': image_weak,  'gt': gt.long(), 'depth': depth, 'edge':edge, 'name': name, 'image_strong': image_strong, 'depth_strong':depth_strong}
            else:
                return {'image_weak': image_weak,  'gt': gt.long(), 'depth': depth, 'name': name, 'image_strong': image_strong, 'depth_strong':depth_strong}
        else:
            return {'image_weak': image_weak,  'depth': depth, 'name': name, 'image_strong': image_strong, 'depth_strong':depth_strong}


    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        edges = []
        for img_path, gt_path, depth_path, edge_path in zip(self.images, self.gts, self.depths, self.edges):
            # print(img_path)
            img = Image.open(img_path)
            depth = Image.open(depth_path)
            if gt_path is not None:
                gt = Image.open(gt_path)
                if img.size == gt.size and gt.size == depth.size:
                    gts.append(gt_path)
            else:
                gts.append(None)

            if edge_path is not None:
                edge=Image.open(edge_path)
                if img.size == edge.size and edge.size == depth.size:
                    edges.append(edge_path)
            else:
                edges.append(None)
            images.append(img_path)
            depths.append(depth_path)

        self.images = images
        self.gts = gts
        self.depths = depths
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth, edge):
        assert img.size == gt.size and gt.size == depth.size and edge.size == img.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), \
                   depth.resize((w, h),Image.NEAREST), edge.resize((w,h), Image.NEAREST)
        else:
            return img, gt, depth, edge

    def __len__(self):
        if self.split == 'label':
            return int(self.size*15)
        else:
            return int(self.size)

def get_loader(image_root, gt_root, depth_root, edge_root, batchsize, trainsize,split,shuffle=True, num_workers=0, pin_memory=True):
    dataset = MirrorDataset(image_root, gt_root, depth_root, edge_root, trainsize,split)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

class SalObjSTDataset(data.Dataset):
    def __init__(self, imgs, plabs, masks, depths, trainsize):
        self.trainsize=trainsize
        # trans_pil=transforms.ToPILImage()
        self.img = [img for img in imgs]
        self.plab = [lab for lab in plabs]
        self.mask = [mask for mask in masks]
        self.depth = [depth for depth in depths]
        self.num = len(self.img)

    def __getitem__(self, idx):
        #
        imgs = self.img[idx]
        plabs=self.plab[idx]
        masks=self.mask[idx]
        depths = self.depth[idx]
        return imgs, plabs.long(), masks, depths

    def __len__(self):
        return self.num