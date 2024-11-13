import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

class MultiHeadAttention2(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 每个头的维度，二者必须整除
        
        # 初始化 Q、K、V 的投影矩阵，将输入词向量线性变换为 Q、K、V，维度保持一致
        self.q_linear = nn.Linear(hidden_size, hidden_size) 
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        # 输出线性层，将拼接后的多头注意力输出变换为所需的输出维度，这里维度保持一致
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state, attention_mask=None):
        # hidden_state 形状: (batch_size, seq_len, hidden_size)
        batch_size = hidden_state.size(0)  # 获取批量大小
        # 计算 Q、K、V，线性变换
        query = self.q_linear(hidden_state)  # (batch_size, seq_len, hidden_size)
        # print (query)

        key = self.k_linear(hidden_state)    # (batch_size, seq_len, hidden_size)
        value = self.v_linear(hidden_state)  # (batch_size, seq_len, hidden_size)
        # print (query)
        # 分割多头，将每个头的维度拆分出来
        query = self.split_head(query)  # (batch_size, num_heads, seq_len, head_dim)
        
        key = self.split_head(key)      # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_head(value)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数，使用缩放点积注意力机制
        # attention_scores 形状: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 添加注意力掩码（seq_len, seq_len），掩码位置（1）的值为负无穷
        if attention_mask is not None:
            attention_scores += attention_mask * -1e9
        
        # 对注意力分数进行归一化，得到注意力概率
        attention_probs = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # 计算注意力输出，通过注意力概率加权值
        output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 对多头注意力输出进行拼接
        # output.transpose(1, 2) 将 num_heads 和 seq_len 维度转置
        # 将形状调整为 (batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.num_heads)
        
        # 通过线性层将拼接后的输出变换为所需的输出维度
        # output = self.o_linear(output)  # (batch_size, seq_len, hidden_size)
        
        return output

    def split_head(self, x):
        batch_size = x.size(0)  # 获取批量大小
        # x 形状: (batch_size, seq_len, hidden_size)
        # 将 hidden_size 分割为 num_heads 和 head_dim
        return x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 返回形状: (batch_size, num_heads, seq_len, head_dim)


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        # if mask is not None:
        #     score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v
        return v, score


class MultiHeadAttention1(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention1, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # print (x)
        # 1. dot product with weight matrices
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        
        # print (q)
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor



# 股票内信息融合模块
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, d_model=512, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        # self.attn_dropout = []
        # if dropout > 0:
        #     for i in range(self.n_heads):
        #         self.attn_dropout.append(nn.Dropout(p=dropout))
        # self.attn_dropout = nn.ModuleList(self.attn_dropout)

        self.norm = LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)
    # def forward(self, hidden_state, attention_mask=None):
    #     # hidden_state 形状: (batch_size, seq_len, hidden_size)
    #     dim = int(self.d_model/self.n_heads)
    #     batch_size = hidden_state.size(0)  # 获取批量大小
    #     # 计算 Q、K、V，线性变换
    #     query = self.Wq(hidden_state)  # (batch_size, seq_len, hidden_size)
    #     # print (query)

    #     key = self.Wk(hidden_state)    # (batch_size, seq_len, hidden_size)
    #     value = self.Wv(hidden_state)  # (batch_size, seq_len, hidden_size)
    #     # print (query)
    #     # 分割多头，将每个头的维度拆分出来
    #     query = self.split_head(query, dim)  # (batch_size, num_heads, seq_len, head_dim)
        
    #     key = self.split_head(key,dim)      # (batch_size, num_heads, seq_len, head_dim)
    #     value = self.split_head(value,dim)  # (batch_size, num_heads, seq_len, head_dim)

    #     # 计算注意力分数，使用缩放点积注意力机制
    #     # attention_scores 形状: (batch_size, num_heads, seq_len, seq_len)
    #     attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(dim, dtype=torch.float32))
        
    #     # 添加注意力掩码（seq_len, seq_len），掩码位置（1）的值为负无穷
    #     if attention_mask is not None:
    #         attention_scores += attention_mask * -1e9
        
    #     # 对注意力分数进行归一化，得到注意力概率
    #     attention_probs = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

    #     # 计算注意力输出，通过注意力概率加权值
    #     output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_dim)
        
    #     # 对多头注意力输出进行拼接
    #     # output.transpose(1, 2) 将 num_heads 和 seq_len 维度转置
    #     # 将形状调整为 (batch_size, seq_len, hidden_size)
    #     output = output.transpose(1, 2).reshape(batch_size, -1, dim * self.n_heads)
        
    #     # 通过线性层将拼接后的输出变换为所需的输出维度
    #     # output = self.o_linear(output)  # (batch_size, seq_len, hidden_size)
        
    #     return output
    # def split_head(self, x,dim):
    #     batch_size = x.size(0)  # 获取批量大小
    #     # x 形状: (batch_size, seq_len, hidden_size)
    #     # 将 hidden_size 分割为 num_heads 和 head_dim
    #     return x.reshape(batch_size, -1, self.n_heads, dim).transpose(1, 2)
    #     # 返回形状: (batch_size, num_heads, seq_len, head_dim)


    def forward(self, x):
        
        q = self.Wq(x)
        # print(q)
        k = self.Wk(x)
        # print(k)
        v = self.Wv(x)
        dim = int(self.d_model/self.n_heads)
        att_output = []
        for i in range(self.n_heads):
            if i == self.n_heads-1:
                qh = q[:,:,i*dim:]
                kh = k[:,:,i*dim:]
                vh = v[:,:,i*dim:]
            else:
                qh = q[:,:,i*dim:(i+1)*dim]
                kh = k[:,:,i*dim:(i+1)*dim]
                vh = v[:,:,i*dim:(i+1)*dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh,kh.transpose(1,2))/torch.sqrt(torch.tensor(dim, dtype=torch.float32)), dim=-1)
            # if self.attn_dropout:
            #     atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh,vh))
        att_output = torch.cat(att_output, dim=-1)
        output = att_output
        # print(output)
        # print('--------------------------------')
        return output
    
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
        # pe: (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer 是一个用于向模型（通常是nn.Module的子类）注册一个不需要梯度的张量（tensor）的方法。
        # 注册的张量将会保存在state_dict中，并在模型保存和加载时被持久化。
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]：天数、T历史窗口、F特征数
        # self.pe[: x.size(0), :]=> [T, 1, F]
        return x + self.pe[: x.size(0), :]
    
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
