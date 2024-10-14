import qlib  
from qlib.data import D  
from qlib.contrib.model.torch_utils import TorchDataset  
from torch import nn  
from torch.nn import TransformerEncoder, TransformerEncoderLayer  
from torch.optim import Adam  
from torch.utils.data import DataLoader  
import numpy as np  

  
  
# 初始化qlib  
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')  
  
# 配置数据集  
instruments = D.instruments(market='all')  
fields = ['$close', '$volume', 'Ref($close, -1)', 'Ref($close, -2)']  
data = D.features(instruments=instruments, fields=fields, start_time='2010-01-01', end_time='2020-01-01')  
  
# 假设我们直接使用close价格作为目标，这里需要更复杂的特征工程  
labels = data['$close'].shift(-1).fillna(method='bfill')  # 下一个交易日的收盘价作为标签  
  
# 划分数据集  
train_data = data.loc['2010-01-01':'2018-12-31']  
test_data = data.loc['2019-01-01':'2020-01-01']  
train_labels = labels.loc['2010-01-01':'2018-12-31']  
test_labels = labels.loc['2019-01-01':'2020-01-01']  
  
# 转换为PyTorch Dataset  
def pad_sequence(sequences, max_len=100):  
    """ 对序列进行填充或截断，使其长度统一为max_len """  
    padded_seqs = np.zeros((len(sequences), max_len, len(fields) - 1))  # 减去1是因为我们不考虑$close本身作为特征  
    for i, seq in enumerate(sequences):  
        seq_len = len(seq)  
        if seq_len > max_len:  
            padded_seqs[i, :max_len] = seq[-max_len:]  
        else:  
            padded_seqs[i, :seq_len] = seq  
    return padded_seqs  
  
train_features = pad_sequence(train_data.values[:-1], max_len=100)  # 去掉最后一行，因为它没有对应的标签  
train_labels = train_labels.values[:-1]  
  
# 假设我们已经有了一个分割好的segment列表（这里为了简化，我们手动创建一个）  
# 在实际应用中，你应该使用qlib提供的工具来生成segments  
segments = [np.arange(i, min(i + 100, len(train_features))) for i in range(0, len(train_features), 100)]  
train_dataset = TorchDataset.from_list(  
    [train_features, train_labels],  
    segment=segments  
)  



#   2024-09-24
# 自己定义Transformer模型  
# 假设的PyTorch Transformer模型定义  

class StockTransformer(nn.Module):  
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward):  
        super().__init__()  
        self.d_model = d_model
        self.nhead = nhead

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)