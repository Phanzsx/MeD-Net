import os
import torch
import torch.nn as nn
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Seg_tools import Trainer
from natsort import natsorted
import scipy.io as sio
from torchvision import transforms
from ScoatNet import ScoatNet
import scipy.io as sio

# segmentation of lesions (3 type of lesion mix as one)


class MyDataset(Dataset):
    def __init__(self, datalist, transform=None):
        self.WC = -400
        self.WW = 1200
        self.datalist = datalist
        self.transform = transform

    def __getitem__(self, index):
        root = self.datalist[index][0]
        fn = self.datalist[index][1]

        img_np = np.load(root)
        img_np = self._wind_transfer(img_np)
        minWindow = self.WC - .5 * self.WW
        img_np = (img_np - minWindow) / self.WW

        label_root = root.replace('img', 'les_gt')
        label_np = np.load(label_root)
        label_np[label_np>0] = 1
        img = self.transform(img_np)
       
        img_o = transforms.ToTensor()(img_np)
        seg = torch.from_numpy(label_np)

        if not img.shape[1] == 512 or not img.shape[2] == 512 or not seg.shape[0] == 512 or not seg.shape[1] == 512:
            img = self.resize_ct(img)
            img_o = self.resize_ct(img_o)
            seg = torch.unsqueeze(seg.float(), 0)
            seg = self.resize_ct(seg)[0]
            seg[seg>.5] = 1
            seg[seg<1] = 0

        return img.float(), seg.float(), img_o.float(), root[:-4]


    def __len__(self):
        return len(self.datalist)

    def _wind_transfer(self, ct_array):
        # ct_array 
        minWindow = self.WC - .5 * self.WW
        maxWindow = self.WC + .5 * self.WW
        ct_array[ct_array < minWindow] = minWindow
        ct_array[ct_array > maxWindow] = maxWindow
        return ct_array
    def resize_ct(self, img):
        img = torch.unsqueeze(img, 0)
        img = torch.nn.functional.interpolate(img, size=[512, 512], mode='bilinear')
        img = torch.squeeze(img, 0)
        return img


def main(batch_size, trainlist, testlist, img_mean, img_std):

    train_data = MyDataset(
        datalist=trainlist,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[img_mean], std=[img_std])
        ]))
    test_data = MyDataset(
        datalist=testlist,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[img_mean], std=[img_std])
        ]))
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=32)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=32)

    model = ScoatNet(num_classes=1, input_channels=1, deep_supervision=False,
                        blockname='Res')

    model = nn.DataParallel(model.cuda(), device_ids=[0,1,2,3])
    # model.load_state_dict(torch.load(save_root + 'train20models.pth'))

    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    trainer = Trainer(model, optimizer, save_dir=save_root)
    trainer.loop(20, train_loader, test_loader, scheduler, save_freq=1)

    optimizer = optim.Adam(params=model.parameters(), lr=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    trainer = Trainer(model, optimizer, save_dir=save_root)
    trainer.loop(10, train_loader, test_loader, scheduler, save_freq=1)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    batch_size = 16
    
    trainlist, testlist = [], []


# load Melioidosis train & test
    list_root = './dataset/1.2.840.113704.7.32.07.5.1.4.85515.30000018082902320696700029153/img/'
    for root, dirs, files in os.walk(list_root):
        for name in files:
            trainlist.append([os.path.join(root, name), name])
    for root, dirs, files in os.walk(list_root):
        for name in files:
            testlist.append([os.path.join(root, name), name])

    trainlist = natsorted(trainlist)

    testlist = natsorted(testlist)

    save_root = './lesions_seg/scoatnet' + '/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    main(batch_size, trainlist, testlist, 0.5018, 0.3699)
