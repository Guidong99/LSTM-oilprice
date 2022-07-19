```python
#包载入
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

#网络搭建
class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()
        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
#             nn.Tanh(),
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim)
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x) 
        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim) 
        y = self.reg(y)                                   
        y = y.view(seq_len, batch_size, -1)               
        return y
    
    def output_y_hc(self, x, hc):
        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)
        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc

#归一化与反归一化
def minmaxscaler(x):
    minx = np.amin(x)
    maxx = np.amax(x)
    return (x - minx)/(maxx - minx), (minx, maxx)

def preminmaxscaler(x, minx, maxx):
    return (x - minx)/(maxx - minx)

def unminmaxscaler(x, minx, maxx):
    return x * (maxx - minx) + minx

#加载数据
df=pd.read_csv('/Users/guidongzhang/Desktop/Data Mining/oilprice/BrentOilPrices.csv')
data=np.array(df['Price'])
data=data[100:250]
data=data[:,np.newaxis]
data_x=data[:-1, :]
data_y=data[1:, :]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#数据预处理  归一化
train_size = 100#定义训练集为100条
train_x, train_x_minmax = minmaxscaler(data_x[:train_size])
train_y, train_y_minmax = minmaxscaler(data_y[:train_size])

#转换为tensor格式
train_x = torch.tensor(train_x, dtype=torch.float32, device=device)
train_y = torch.tensor(train_y, dtype=torch.float32, device=device)

#对时序数据进行切割、重组
window_len = 40
batch_x,batch_y=list(),list()
for i in range(len(train_x),window_len,-3):
    batch_x.append(train_x[i-window_len:i])
    batch_y.append(train_y[i-window_len:i])
    
#多线模式一
batch_x = pad_sequence(batch_x)
batch_y = pad_sequence(batch_y)

#训练数据

net = RegLSTM(inp_dim=1, out_dim=1, mid_dim=16, mid_layers=2)#定义模型
loss = nn.MSELoss()#定义损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)#定义优化器


# # 开始训练
print("Training......")
for e in range(1000):
    out = net(batch_x)

    Loss = loss(out, batch_y)
    
    optimizer.zero_grad()#梯度清零
    
    Loss.backward()#反向传播
    
    optimizer.step()#梯度更新

    if e % 10 == 0:
        print('Epoch: {:4}, Loss: {:.5f}'.format(e, Loss.item()))

#定义参数和初始值状态
mid_layers=2
eval_size=1
mid_dim=16
zero_ten = torch.zeros((mid_layers, eval_size, mid_dim), dtype=torch.float32)

#预测
test_len = 40
for i in range(train_size, len(new_data_x)):  # 要预测的是i
    test_x = new_data_x[i-test_len:i, np.newaxis, :]
    test_x = preminmaxscaler(test_x, train_x_minmax[0], train_x_minmax[1])
    batch_test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
    if i == train_size:
        test_y, hc = net.output_y_hc(batch_test_x, (zero_ten, zero_ten))
    else:
        test_y, hc = net.output_y_hc(batch_test_x, hc)
#     test_y = net(batch_test_x)
    predict_y = test_y[-1].item()
    predict_y = unminmaxscaler(predict_y, train_x_minmax[0], train_y_minmax[1])
    new_data_x[i] = predict_y
    
#绘制图
plt.plot(new_data_x, 'r', label='pred')
plt.plot(data_x, 'b', label='real', alpha=0.3)
plt.legend(loc='best')
```

