import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils.config import cfg


class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        model = models.resnet101(pretrained=True)       
        for param in model.parameters():
            param.requires_grad = False        
        self.define_module(model)       

    def define_module(self, model):
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool 
        self.embedding = nn.Conv2d(1024,256,1)       
        # self.attention = MultiHeadSelfAttention(cfg.RNN_ATT.n_heads, 2048, 1024) 
        # self.att = multi_attention(1024,1024,1)
        # self.fc = nn.Linear(2048,1024)        
        # self.bnorm = nn.BatchNorm1d(2048)


    def forward(self, x):
        # x = nn.functional.interpolate(x,size=(244, 244), mode='bilinear', align_corners=False)    # (3, 244, 244)
        x = self.conv1(x)    # (64, 122, 122)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)     #(256, 61, 61)     
        x = self.layer2(x)     #(512, 31, 31)        
        x = self.layer3(x)        #(1024, 16, 16)  
        feature = self.embedding(x)  
        feature = F.normalize(feature,p=2,dim=1)
        # global_feature = nn.functional.normalize(global_feature, p=2, dim=1)    
        # global_feature = nn.functional.normalize(global_feature, p=2, dim=1)  
        return feature #,features,loc_feature