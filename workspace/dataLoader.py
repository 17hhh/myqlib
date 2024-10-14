import qlib  
# from qlib.config import C  
from qlib.data import D  
  
# 初始化Qlib。这里假设你已经按照Qlib的文档设置了数据路径。  
# 如果你的数据路径不同，请相应地修改provider_uri。  
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')  
  
# 列出所有可用的股票代码（这里以A股市场为例）
# my_instruments=D.instruments(market='all')  
# instruments = D.list_instruments(instruments=my_instruments)  
  
# # 选择你想要加载的字段，这里以收盘价（$close）为例  
# fields = ['$close','$high','$volume']  
  
# # 设置数据加载的时间范围  
# start_time = '2020-01-01'  
# end_time = '2023-01-01'  
  
# # 使用D.features函数加载数据  
# # 注意：这里返回的是一个pandas DataFrame，其中包含了指定时间段内所有选定股票和字段的数据  
# data = D.features(instruments=instruments,  
#                   fields=fields,  
#                   start_time=start_time,  
#                   end_time=end_time)  
  
# # 查看数据的前几行以确认加载正确  
# print(data.head())  

from qlib.data.dataset.loader import QlibDataLoader
MACD_EXP = '(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close'
fields = [MACD_EXP] # MACD
names = ['MACD']
labels = ['Ref($close, -2)/Ref($close, -1) - 1'] # label
label_names = ['LABEL']
data_loader_config = {
    "feature": (fields, names),
    "label": (labels, label_names)
}
data_loader = QlibDataLoader(config=data_loader_config)
df = data_loader.load(instruments='csi300', start_time='2010-01-01', end_time='2017-12-31')
print(df.shift(-1))