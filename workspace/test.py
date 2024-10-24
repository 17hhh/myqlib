# 初始化配置
import qlib
from qlib.constant import REG_CN
data_uri = 'home/.qlib/qlib_data/cn_data'
qlib.init(provider_uri=data_uri, region=REG_CN)

from qlib.data.dataset.loader import QlibDataLoader
# 加载原始特征，比如收盘价、最高价
qdl = QlibDataLoader(config=(['$close', '$high'],['close', 'high'])) 
qdl.load(instruments=['SH600519'], start_time='20190101', end_time='20191231') # 可以通过freq参数设置周期，默认freq='day'