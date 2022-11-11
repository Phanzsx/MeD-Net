import os
import torch
from tqdm import tqdm
from torchvision import transforms


class Trainer(object):
    def __init__(self, model, save_dir=None, is_lung_seg=True):
        self.model = model
        self.save_dir = save_dir
        self.is_lung_seg = is_lung_seg

    def _loop(self, data_loader, is_train=True):
        for data, seg, data_o, fn in tqdm(data_loader):

            data, seg = data.cuda(), seg.cuda()

            if is_train:
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)
            
            out = self.model(data)
            n = out.size(0)
            for j in range(n):
                self.write_img(torch.squeeze(out[j]), fn[j])

    def test(self, data_loader):
        self.model.eval()
        self._loop(data_loader, is_train=False)

    def write_img(self, out, fn, **kwargs):
        out = torch.sigmoid(out)
        if self.is_lung_seg:
            out[out>.5] = 1
            out[out<1] = 0

            save_dir =  fn[:-6].replace('/dataset/','/output/lungSeg/') # lung seg
        else:   
            save_dir =  fn[:-6].replace('/dataset/','/output/lesSeg/') # lesions Seg

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        name = fn[-2:] # lung
        tensor2img = transforms.ToPILImage()
        out_img = tensor2img(out)
        out_img.save(save_dir + name + '.png')


