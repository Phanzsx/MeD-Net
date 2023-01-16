import os
import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import h5py
from sklearn.cluster import KMeans
import numpy as np
import random
import math
from utils import Trainer
# from wvit import mtransformer
# from nwvit import mtransformer
# from nnvit_dual import mtransformer
from mmvit import mtransformer
# from mmvit_single import mtransformer
from pretrained_model_mmvit import Pre_model


class MyDataset(Dataset):
    def __init__(self, mode, txt, img_mean, img_std, num_img_s):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.mode = mode
        self.img_mean = img_mean
        self.img_std = img_std
        self.num_img_s = num_img_s

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        data = np.load(data_root + fn + '/data.npz')
        img = data['img']
        position = data['position']

        data = np.load(singledata_root + fn + '/index.npz')
        index = data['index']
        img_c = []
        for i in range(self.num_img_s):
            data = np.load(singledata_root + fn + '/' + str(index[i, 0]) + '.mat.npz')
            img_c.append(data['img_s'])
            data.close()
        img_c = np.array(img_c)
        
        # min = -1350
        # max = 150
        img = torch.from_numpy(img)

        img1 = img.unsqueeze(1)
        min = -200
        max = 200
        img1[img1<min] = min
        img1[img1>max] = max
        img1 = (img1-min)/(max-min)
        img1.sub_(0.2715).div_(0.2910)
        
        img2 = img.unsqueeze(1)
        min = -600
        max = -200
        img2[img2<min] = min
        img2[img2>max] = max
        img2 = (img2-min)/(max-min)
        img2.sub_(0.5203).div_(0.4160)
        
        img3 = img.unsqueeze(1)
        min = -1000
        max = -600
        img3[img3<min] = min
        img3[img3>max] = max
        img3 = (img3-min)/(max-min)
        img3.sub_(0.7696).div_(0.2375)
        
        img = torch.cat((img1, img2, img3), 1)

        position = torch.from_numpy(position)

        min = -1350
        max = 150
        img_c[img_c<min] = min
        img_c[img_c>max] = max
        img_c = (img_c-min)/(max-min)
        img_c = torch.from_numpy(img_c)
        img_c.sub_(0.6395).div_(0.2295)
        # if random.randint(0, 1):
        #     img_c.flip(dims=[2])
        
        n = img.shape
        if n[0] > num_patch:
            # head
            # img_local = img_local[0:num_patch]
            # img_global = img_global[0:num_patch]
            # position = position[0:num_patch]

            # middle
            # start = (n[0]-num_patch)//2
            # img = img[start:num_patch+start]
            # position = position[start:num_patch+start]

            #uniform
            u = np.linspace(0, n[0]-1, num_patch)
            u = u.astype('int')
            img = img[u]
            position = position[u]
        else:
            img = torch.cat((img, torch.zeros(num_patch-n[0], 3, 48, 48)), 0)
            position = torch.cat((position, torch.zeros(num_patch-n[0], 3)), 0)
        
        # img = torch.from_numpy(img)
        # img = self.transform(img)
        # position = torch.from_numpy(position)

        #random position
        if 0:#self.mode == 'train':
            # kmeans
            kmeans = KMeans(n_clusters=10).fit(position)
            cluster = kmeans.labels_
            randorder = np.arange(10)
            np.random.shuffle(randorder)
            order = np.empty((0))
            for i in randorder:
                order = np.hstack((order, np.where(cluster == i)[0]))
            
            # full rand
            # order = torch.randperm(num_patch)
            # img = img[order]
            # position = position[order]

        # pos1 = (position[:, 0] - 1)*16 + position[:, 1] - 1
        # pos2 = position[:, 2] - 1

        pos0 = position[:, 0] - 1
        pos1 = position[:, 1] - 1
        pos2 = position[:, 2] - 1

        return img.float(), pos0.long(), pos1.long(), pos2.long(), img_c.float(), label

    def __len__(self):
        return len(self.imgs)

def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(batch_size, img_mean, img_std, num_img_s, i):
    train_data = MyDataset(
        mode = 'train',
        txt='{}train_label_{}.txt'.format(label_root, i+1),
        img_mean=img_mean,
        img_std=img_std, 
        num_img_s=num_img_s)
    test_data = MyDataset(
        mode = 'test',
        txt='{}test_label_{}.txt'.format(label_root, i+1),
        img_mean=img_mean,
        img_std=img_std,
        num_img_s=num_img_s)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    # model = mtransformer(patch_size=16, num_patch=num_patch, dim=512, num_classes=2, pool='pool')
    # model = mtransformer(num_patch=num_patch, stride=32, dim=256, num_classes=3, type='mean')
    model = mtransformer(num_patch=num_patch, stride=16, dim=512, num_input=num_img_s, num_classes=num_classes, pool='mean')

    # pretrained on MedMNIST
    pretrained = Pre_model(num_classes=11, num_patch=28)
    pretrained = nn.DataParallel(pretrained.cuda(), device_ids=[0])
    pretrained.load_state_dict(torch.load('./temp/pretrained_model_mmvit/train16models.pth'))
    pretrained.module.fc = nn.Identity()
    # pretrained.module.patch_embed.proj = nn.Identity()
    pretrained_dict = pretrained.state_dict()
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    # model.cuda()
    model = nn.DataParallel(model.cuda(), device_ids=[0, 1])
    # optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 0)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=10)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15], gamma=1)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20)
    trainer = Trainer(model, optimizer, save_dir=save_root, num_classes=num_classes)
    trainer.loop(6, train_loader, test_loader, scheduler, save_freq=1)

    # optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
    # # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 0)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 25, 35], gamma=0.1)
    # trainer = Trainer(model, optimizer, save_dir=save_root)
    # trainer.loop(30, train_loader, test_loader, scheduler, save_freq=1, k_fold=k_fold)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 3'
    init_seeds(seed=617)
    import warnings
    warnings.filterwarnings("ignore")

    batch_size = 8
    num_patch = 512
    num_img_s = 5
    num_classes = 3
    class_sub = ['meli_covid_other_3']
    data_root = './data/patch/'
    singledata_root = './data/slice/'
    label_root = './data/label/' + class_sub[0] + '/'
    for i in range(5):
        save_root = './temp/ablation/k=5/' + class_sub[0] + '#res34_WCE/' + str(i+1) + '/'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        main(batch_size, 0.5657, 0.2249, num_img_s, i)
