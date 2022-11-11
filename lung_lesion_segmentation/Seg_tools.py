import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from losses import *
from matrics import *


class Trainer(object):
    def __init__(self, model, optimizer, save_dir=None, save_freq=1):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _loop(self, data_loader, ep, is_train=True):
        loop_loss,  loop_iou, loop_dice = [], [], []
        mode = 'train' if is_train else 'test'
        for data, seg, data_o, fn in tqdm(data_loader):
            data, seg = data.cuda(), seg.cuda()

            if is_train:
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)
            
            out = self.model(data)

            n = out.size(0)
            loss = F.binary_cross_entropy(torch.sigmoid(out.view(n, -1)), seg.view(n, -1)) \
                + 0.5 * DICELoss(out, seg)

            loop_loss.append(loss.data / len(data))
            loop_dice.append(dice_coef(out, seg))
            loop_iou.append(iou_score(out, seg))
            
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print(mode + ': loss_seg: {:.6f}, iou: {:.6f}, dice: {:.6f}'.format(
            sum(loop_loss)/len(loop_loss), sum(loop_iou)/len(loop_iou), sum(loop_dice)/len(loop_dice)))
        return sum(loop_loss)/len(loop_loss), sum(loop_iou)/len(loop_iou), sum(loop_dice)/len(loop_dice)

    def train(self, data_loader, ep):
        self.model.train()
        loss, iou, dice = self._loop(data_loader, ep)
        return loss, iou, dice

    def test(self, data_loader, ep):
        self.model.eval()
        loss, iou, dice = self._loop(data_loader, ep, is_train=False)
        return loss, iou, dice

    def loop(self, epochs, train_data, test_data, scheduler=None, save_freq=5):
        loss_acc = []
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print('epoch {}'.format(ep))
            train_loss, train_iou, train_dice = self.train(train_data, ep)
            test_loss, test_iou, test_dice = self.test(test_data, ep)
            loss_acc.append(train_loss.cpu(), test_loss.cpu(), train_iou, test_iou, train_dice, test_dice)

            if not ep % save_freq:
                self.save(ep)

            if epochs == 20:
                np.savetxt(self.save_dir + 'loss01.txt', loss_acc, '%.6f')
            else:
                np.savetxt(self.save_dir + 'loss001.txt', loss_acc, '%.6f')

    def save(self, epoch, **kwargs):
        if self.save_dir:
            name = self.save_dir + 'train' + str(epoch) + 'models.pth'
            torch.save(self.model.state_dict(), name)

