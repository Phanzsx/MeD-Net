import warnings
warnings.filterwarnings('ignore')
import numpy as np
import h5py
from sklearn import svm
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import permutation_importance
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import scipy.io as sio
import scipy.stats as stats
import csv
import matplotlib.pyplot as plt
import os
import shap
import pickle
from xgboost import XGBClassifier
from matplotlib.font_manager import FontProperties
import pandas as pd


large = 22; med = 16; small = 12
params = {'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,
        #   'axes.prop_cycle': (cycler('color', ['#000000', '#08298A', '#848484', '#FF0080', '#088A85', 
        #                                        '#B45F04', '#FF8000', '#AEB404', '#8904B1', '#04B45F',
        #                                        '#01DFD7', '#B40431'])
        #    + cycler('ls', ['-','-','--','--','-','-','--','--','-','-','--','--']))
        }
plt.rcParams.update(params)
plt.rcParams['axes.unicode_minus'] = False
# matplotlib inline
font = FontProperties(fname='C:\Windows\Fonts\Arial.ttf', size=16)


class mlp(nn.Module):
    def __init__(self, input_channel=27, dims=[21, 30], num_classes=2):
        super().__init__()
        layers = []
        # layers.append(nn.LayerNorm(input_channel))
        layers.append(nn.Linear(input_channel, dims[0]))
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # layers.append(nn.LayerNorm(dims[i+1]))
            # layers.append(nn.BatchNorm1d(dims[i+1]))
            # layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y


class MLPClassifier(object):
    def __init__(self, input_channel=None, dims=[128, 128, 32]):
        self.input_channel = input_channel
        self.model = mlp(input_channel=input_channel, dims=dims)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5], gamma=0.1)
        # torch.manual_seed(0)
    
    def fit(self, traindata, label, epoch=20):
        torch.set_grad_enabled(True)
        traindata = torch.from_numpy(traindata).float()
        label = torch.from_numpy(label).long()
        for i in range(epoch):
            # print('epoch {}'.format(i+1))
            self.scheduler.step()
            out = self.model(traindata)
            loss = F.cross_entropy(out, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            acc = float((out.data.max(1)[1] == label.data).sum()) / out.size(0)
            # print('train_loss: {:.6f}, acc: {:.6%}'.format(loss, acc))
        return self

    def predict_proba(self, testdata):
        torch.set_grad_enabled(False)
        testdata = torch.from_numpy(testdata).float()
        out = self.model(testdata)
        return out.data.numpy()

    def predict(self, testdata):
        torch.set_grad_enabled(False)
        testdata = torch.from_numpy(testdata).float()
        out = self.model(testdata)
        out = out.data.max(1)[1].numpy()
        return out


def main(dims, ex):
    df = pd.read_excel('feat_list.xlsx')
    feat_name = df.values
    temp = np.load('impute_data.npz')
    temp = temp['data']

    num = len(feat_name)
    gt = np.empty((0))
    pre_all = np.empty((0))
    pro_all = np.empty((0, 2))
    feat_im = np.zeros((num))
    values = np.empty((0, num))
    data = np.empty((0, num))
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    num_fold = 1

    for index in kf.split(temp):
        trainlabel = temp[index[0], 1]
        testlabel = temp[index[1], 1]
        traindata = temp[index[0], 2:]
        testdata = temp[index[1], 2:]

        gt = np.concatenate((gt, testlabel))

        # imp = SimpleImputer(strategy='mean')
        # traindata = imp.fit_transform(traindata)
        # testdata = imp.fit_transform(testdata)

        trainmin = np.nanmin(traindata, 0)
        trainmax = np.nanmax(traindata, 0)

        traindata = (traindata-trainmin)/(trainmax-trainmin+0.00001)
        testdata = (testdata-trainmin)/(trainmax-trainmin+0.00001)

        ros = RandomOverSampler(random_state=0)
        rus = RandomUnderSampler(random_state=0)
        # traindata, trainlabel = ros.fit_resample(traindata, trainlabel)
        traindata, trainlabel = rus.fit_resample(traindata, trainlabel)

        # choose the model
        # clf = svm.SVC(probability=True, kernel='poly')
        # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
        #         max_depth=3)
        # clf = RandomForestClassifier(n_estimators=100, max_depth=3,
        #         min_samples_split=5)
        # clf = HistGradientBoostingClassifier(max_iter=100)
        # clf = AdaBoostClassifier(n_estimators=100)
        # clf = XGBClassifier()
        clf = MLPClassifier(input_channel=traindata.shape[1], dims=dims)

        # clf.fit(traindata, trainlabel)

        # load model
        # clf = torch.load('./temp/mlp_w_radio_{}/{}.pth'.format(ex, num_fold))
        # clf = joblib.load('./temp/rf_w_radio/{}.model'.format(num_fold))
        # clf = joblib.load('./temp/gb_w_radio/{}.model'.format(num_fold))

        save_root = './temp/mlp_{}_{}/'.format('w_radio', ex)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        # save model
        # joblib.dump(clf, save_root + '{}.model'.format(num_fold))
        torch.save(clf, save_root + '{}.pth'.format(num_fold))

        pre = clf.predict(testdata)
        pro = clf.predict_proba(testdata)
        pre_all = np.concatenate((pre_all, pre))
        pro_all = np.concatenate((pro_all, pro))
        
        # feat_im = feat_im + clf.feature_importances_
        # feat_im = feat_im + clf.feat_importances(traindata, trainlabel)


        # explainer = shap.TreeExplainer(clf)
        # shap_values = explainer(traindata)
        # values = np.concatenate((values, shap_values.values[:, :]), 0)
        # data = np.concatenate((data, traindata), 0)

        # traindata = testdata
        # e = shap.DeepExplainer(clf.model, torch.from_numpy(traindata).float())
        # torch.set_grad_enabled(True)
        # shap_values = e.shap_values(torch.from_numpy(traindata).float())
        # values = np.concatenate((values, shap_values[1]), 0)
        # data = np.concatenate((data, traindata), 0)

        num_fold += 1

    # np.savez('./result/MLP_2.npz', pre=pre_all, pro=pro_all, gt = gt)
    # sio.savemat('./result/MLP_2.mat', {'pre': pre_all, 'pro': pro_all, 'gt': gt})

    n = len(gt)
    CM = confusion_matrix(gt, pre_all)
    CM_h = CM/np.sum(CM, 1).reshape(2, 1)
    tp = CM[1, 1]
    fn = CM[1, 0]
    fp = CM[0, 1]
    tn = CM[0, 0]

    acc = (tp + tn) / (tp + tn + fn + fp)
    sen = tp / (tp + fn)
    spe = tn / (fp + tn)

    fpr, tpr, _ = roc_curve(gt, pro_all[:, 1])
    roc_auc = auc(fpr, tpr)

    print(acc, sen, spe)
    print(CM_h)
    print(roc_auc)
    return feat_im, acc, sen, spe, roc_auc

if __name__ == '__main__':
    main(dims=[128, 32, 32], ex=0)