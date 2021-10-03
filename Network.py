import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import sys
import warnings
import torch.nn.functional as F

import numpy as np
import torchvision
from torchvision import datasets,models,transforms
from torch.utils.data import DataLoader
from torch.autograd import Function,Variable

import matplotlib.pyplot as plt
import time
import os
import copy

warnings.filterwarnings("ignore")

model_ft = models.resnet101(pretrained=True)

class fcn(nn.Module):
    def __init__(self,model):
        super(fcn,self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.conv = nn.Conv2d(2048,2048,1)
        self.bn = nn.BatchNorm2d(2048)

    def forward(self,input):
        output = self.features(input)
        output = self.conv(output)
        output = self.bn(output)
        return output


class attention_branch_2(nn.Module):
    def __init__(self):
        super(attention_branch_2, self).__init__()
        self.conv = nn.Conv2d(2048, 128, 1)
        self.adaptive_max_pooling = nn.AdaptiveMaxPool1d(1)
        self.fc_1 = nn.Linear(128, 12)
        self.fc_2 = nn.Linear(12, 128)
        self.fc_w = nn.Linear(12, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(128)
        self.lbn = nn.LayerNorm(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_fea_p = self.conv(x)
        x_fea_p = self.bn(x_fea_p)
        x_fea_p = self.relu(x_fea_p)
        x = torch.mean(x_fea_p.view(-1, x_fea_p.size(1), x_fea_p.size(2) * x_fea_p.size(3)), dim=2)
        x_2_p = self.adaptive_max_pooling(x.view(-1,1,x.size(1)))
        x_2_p = x_2_p.view(-1,1)
        x_channel_atten = self.fc_1(x)
        x_channel_atten = self.relu(x_channel_atten)
        x_w = self.fc_w(x_channel_atten)
        x_channel_atten = self.fc_2(x_channel_atten)
        x_channel_atten = F.softmax(x_channel_atten, dim = 1)
        x_channel_atten = x_channel_atten.view(-1, x_channel_atten.size(1), 1, 1)
        x_spacial_atten = x_channel_atten * x_fea_p
        x_spacial_atten = torch.sum(x_spacial_atten, dim=1).unsqueeze(1)
        x_spacial_atten = self.sigmoid(x_spacial_atten)
        
        return x_w, x_spacial_atten,x_2_p

class class_branch_2(nn.Module):
    def __init__(self):
        super(class_branch_2, self).__init__()
        self.fc_class = nn.Sequential(
            nn.Linear(2048,256),
            nn.Dropout(0.3),
            nn.Linear(256,2),
        )

    def forward(self, x_spacial_atten_p,x_spacial_atten_n,x_w,x_fea):
        x_spacial_atten = torch.cat((x_spacial_atten_p, x_spacial_atten_n), dim=1)
        x_spacial_atten = torch.sum(x_spacial_atten, dim=1).unsqueeze(1)
        x = torch.mul(x_spacial_atten,x_fea)
        x_c_2 = x + x_fea
        x_c_2 = torch.mean(x_c_2.view(-1, x_c_2.size(1), x_c_2.size(2) * x_c_2.size(3)), dim=2)
        x_c_2 = self.fc_class(x_c_2)
        return x_c_2

class fin(nn.Module):
    def __init__(self,model):
        super(fin, self).__init__()
        self.fcn = fcn(model)
        self.p_branch = attention_branch_2()
        self.n_branch = attention_branch_2()
        self.classify_2 = class_branch_2()


    def forward(self, x):
        x_fea = self.fcn(x)
        x_w_p,x_spacial_atten_p, x_2_p = self.p_branch(x_fea)
        x_w_n,x_spacial_atten_n, x_2_n = self.n_branch(x_fea)
        x_c_2_1 = torch.cat((x_2_p, x_2_n), dim=1)
        x_w = torch.cat((x_w_p, x_w_n), dim=1)
        x_w = F.softmax(x_w,dim=1)
        x_c_2_2 = self.classify_2(x_spacial_atten_p,x_spacial_atten_n,x_w,x_fea)
        x_c_2_2 = x_w + x_c_2_2

        return x_c_2_1,x_c_2_2,x_w

def main():
    model = fin(model = model_ft)
    input = torch.randn(16,3,448,448)
    output1,output2,output3 = model(input)
    print(output1.size())
    print(output2.size())


if __name__ == '__main__':
    main()
