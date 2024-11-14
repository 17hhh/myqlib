import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, d_model=512, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_state, attention_mask=None):
        # hidden_state 形状: (batch_size, seq_len, hidden_size)
        dim = int(self.d_model/self.n_heads)
        batch_size = hidden_state.size(0)  # 获取批量大小
        # 计算 Q、K、V，线性变换
        query = self.Wq(hidden_state)  # (batch_size, seq_len, hidden_size)
        # print (query)

        key = self.Wk(hidden_state)    # (batch_size, seq_len, hidden_size)
        value = self.Wv(hidden_state)  # (batch_size, seq_len, hidden_size)
        # print (query)
        # 分割多头，将每个头的维度拆分出来
        query = self.split_head(query, dim)  # (batch_size, num_heads, seq_len, head_dim)
        
        key = self.split_head(key,dim)      # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_head(value,dim)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数，使用缩放点积注意力机制
        # attention_scores 形状: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(dim, dtype=torch.float32))
        
        # 添加注意力掩码（seq_len, seq_len），掩码位置（1）的值为负无穷
        if attention_mask is not None:
            attention_scores += attention_mask * -1e9
        
        # 对注意力分数进行归一化，得到注意力概率
        attention_probs = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        # drop the attention_probs
        attention_probs = self.dropout(attention_probs)
        # 计算注意力输出，通过注意力概率加权值
        output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 对多头注意力输出进行拼接
        # output.transpose(1, 2) 将 num_heads 和 seq_len 维度转置
        # 将形状调整为 (batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).reshape(batch_size, -1, dim * self.n_heads)
        
        # 通过线性层将拼接后的输出变换为所需的输出维度
        # output = self.o_linear(output)  # (batch_size, seq_len, hidden_size)
        
        return output
    def split_head(self, x, dim):
        batch_size = x.size(0)  # 获取批量大小
        # x 形状: (batch_size, seq_len, hidden_size)
        # 将 hidden_size 分割为 num_heads 和 head_dim
        return x.reshape(batch_size, -1, self.n_heads, dim).transpose(1, 2)
        # 返回形状: (batch_size, num_heads, seq_len, head_dim)



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
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        # position:(max_len,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term: (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # self.register_buffer 是一个用于向模型（通常是nn.Module的子类）注册一个不需要梯度的张量（tensor）的方法。
        # 注册的张量将会保存在state_dict中，并在模型保存和加载时被持久化。
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]：天数、T历史窗口、F特征数
        # self.pe[: x.size(0), :]=> [T, 1, F]
        print(self.pe.shape)
        print(x.shape)
        return x + self.pe[:x.shape[1], :]
    
class AddNormLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNormLayer, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, src):
        input = self.dropout(input)
        return self.norm(input+ src) 

class TransformerModule(nn.Module):
    def __init__(self, d_ff=158, d_model=512, dropout=0.2, n_layers=6):
        super(TransformerModule, self).__init__()
        self.d_model = d_model
        # self.layers = nn.Sequential(
        #     nn.Linear(d_ff, d_model),
        #     PositionalEncoding(d_model, max_len=10000),
        #     Attention(d_model, dropout=0.1),
        #     AddNormLayer(d_model, dropout=0.1),
        #     nn.Linear(d_model, 1)
        # )
        self.feature_layer = nn.Linear(d_ff, d_model)
        self.posi = PositionalEncoding(d_model, max_len=10000)
        self.addNorm = AddNormLayer(d_model, dropout=0.1)
        self.attn = Attention(d_model, dropout=0.1)

    def forward(self, src):
        # output = self.layers(src)
        feature = self.feature_layer(src.float())
        posi_out=self.posi(feature)
        attn_out=self.attn(posi_out)
        addNorm_out = self.addNorm(attn_out, feature.float())
        return addNorm_out


def train_transformer(dataloader):
    model = TransformerModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    for batch_idx,data in dataloader:
        feature = data[:, :, 0:-1]
        label = data[:, -1, -1]
        pred = model(feature.float())
        # print(pred)
        mask = ~torch.isnan(label)
        loss = torch.mean((pred[mask] - label[mask]) ** 2)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()
        print(f'batch_idx:{batch_idx}, loss: {loss.item()}')
    return float(np.mean(losses))
