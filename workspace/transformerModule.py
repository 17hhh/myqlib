import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

# 股票内信息融合模块
class Attention(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        # self.feature_layer = nn.Linear(d_feat, d_model)
        # self.pos_encoder = PositionalEncoding(d_model)
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model)


    def forward(self, x):
        # x = self.pos_encoder(x)  
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        kt = torch.transpose(k, 1, 2)
        s = (q @ kt) / math.sqrt(self.d_model)
        v = s @ v
        return v
    
# 前馈网络模块
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        # position:(max_len,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term: (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer 是一个用于向模型（通常是nn.Module的子类）注册一个不需要梯度的张量（tensor）的方法。
        # 注册的张量将会保存在state_dict中，并在模型保存和加载时被持久化。
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]：天数、T历史窗口、F特征数
        return x + self.pe[: x.size(0), :]

class AddNormLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNormLayer, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, src):
        input = self.dropout(input)
        return self.norm(input+ src) 

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        # 预测股票移动趋势
        self.linear2 = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)