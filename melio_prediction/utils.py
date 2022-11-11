import os
import torch
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import pdb
from torchvision import transforms
from PIL import Image
import time


class Trainer(object):
    def __init__(self, model, optimizer, save_dir=None, save_freq=1, num_classes=None):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.numclasses = num_classes

    def _loop(self, data_loader, ep, is_train=True):
        loop_loss_class, correct = [], []
        tp, tn, fp, fn = 0, 0, 0, 0
        tensor2img = transforms.ToPILImage()
        mode = 'train' if is_train else 'test'
        for img, pos0, pos1, pos2, img_c, label in tqdm(data_loader):
            img, pos0, pos1, pos2, img_c, label = img.cuda(), \
                pos0.cuda(), pos1.cuda(), pos2.cuda(), img_c.cuda(), label.cuda()
            if is_train:
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)
            n = img.size()
            out_class = self.model(img, pos0, pos1, pos2, img_c)
            if self.numclasses == 3:
                weight = torch.FloatTensor([0.5, 0.2, 0.3]).cuda()
                loss_class = F.cross_entropy(out_class, label, weight)
            elif self.numclasses == 2:
                loss_class = F.cross_entropy(out_class, label)
            # loss_class = CrossEntropyLabelSmooth(2)(out_class, label)
            loop_loss_class.append(loss_class.detach() / len(data_loader))
            out = (out_class.data.max(1)[1] == label.data).sum()
            correct.append(float(out) / len(data_loader.dataset))
            tp, tn, fp, fn = self.matrix(out_class.data.max(1)[1], label.data, tp, tn, fp, fn)
            if is_train:
                self.optimizer.zero_grad()
                loss_class.backward()
                self.optimizer.step()
        
        sen = tp / (tp + fn + 0.001)
        spe = tn / (tn + fp + 0.001)
        print(mode + ': loss_class: {:.6f}, Acc: {:.6%}, Sen: {:.6f}, Spe: {:6f}'.format(
            sum(loop_loss_class), sum(correct), sen, spe))
        return sum(loop_loss_class), sum(correct), sen, spe

    def matrix(self, pre, gt, tp, tn, fp, fn):
        n = pre.size()
        for i in range(n[0]):
            if gt[i] == 0:
                if pre[i] == 0:
                    tp += 1
                else:
                    fn += 1
            else:
                if pre[i] == 0:
                    fp += 1
                else:
                    tn += 1
        return tp, tn, fp, fn

    def train(self, data_loader, ep):
        self.model.train()
        index = self._loop(data_loader, ep)
        return index

    def test(self, data_loader, ep):
        self.model.eval()
        index = self._loop(data_loader, ep, is_train=False)
        return index

    def loop(self, epochs, train_data, test_data, scheduler=None, save_freq=5):
        loss_acc = []
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print('epoch {}'.format(ep))
            train_index = self.train(train_data, ep)
            test_index = self.test(test_data, ep)
            loss_acc.append(train_index + test_index)
            if not ep % save_freq:
                self.save(ep)
            np.savetxt(self.save_dir + 'loss.txt' , loss_acc, '%.6f')

    def save(self, epoch, **kwargs):
        if self.save_dir:
            # name = f"weight-{epoch}-" + "-".join([f"{k}_{v}" for k, v in kwargs.items()]) + ".pkl"
            # torch.save({"weight": self.model.state_dict()},
            #            os.path.join(self.save_dir, name))
            name = self.save_dir + 'train' + str(epoch) + 'models.pth'
            # torch.save(self.model.state_dict(), name)
            torch.save(self.model, name)