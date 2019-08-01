

```python
import torch
import torch.nn as nn
import numpy as np
torch.__version__
```




    '1.0.0'



# 3.1 logistic回归实战
在这一章里面，我们将处理一下结构化数据，并使用logistic回归对结构化数据进行简单的分类。
## 3.1.1 logistic回归介绍
logistic回归是一种广义线性回归（generalized linear model），与多重线性回归分析有很多相同之处。它们的模型形式基本上相同，都具有 wx + b，其中w和b是待求参数，其区别在于他们的因变量不同，多重线性回归直接将wx+b作为因变量，即y =wx+b,而logistic回归则通过函数L将wx+b对应一个隐状态p，p =L(wx+b),然后根据p 与1-p的大小决定因变量的值。如果L是logistic函数，就是logistic回归，如果L是多项式函数就是多项式回归。

说的更通俗一点，就是logistic回归会在线性回归后再加一层logistic函数的调用。

logistic回归主要是进行二分类预测，我们在激活函数时候讲到过 Sigmod函数，Sigmod函数是最常见的logistic函数，因为Sigmod函数的输出的是是对于0~1之间的概率值，当概率大于0.5预测为1，小于0.5预测为0。

下面我们就来使用公开的数据来进行介绍

## 3.1.2 UCI German Credit  数据集

UCI German Credit是UCI的德国信用数据集，里面有原数据和数值化后的数据。

German Credit数据是根据个人的银行贷款信息和申请客户贷款逾期发生情况来预测贷款违约倾向的数据集，数据集包含24个维度的，1000条数据，

在这里我们直接使用处理好的数值化的数据，作为展示。

[地址](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/)

## 3.2 代码实战
我们这里使用的 german.data-numeric是numpy处理好数值化数据，我们直接使用numpy的load方法读取即可


```python
data=np.loadtxt("german.data-numeric")
```

数据读取完成后我们要对数据做一下归一化的处理


```python
n,l=data.shape
for j in range(l-1):
    meanVal=np.mean(data[:,j])
    stdVal=np.std(data[:,j])
    data[:,j]=(data[:,j]-meanVal)/stdVal
```

打乱数据


```python
np.random.shuffle(data)
```

区分训练集和测试集，由于这里没有验证集，所以我们直接使用测试集的准确度作为评判好坏的标准

区分规则：900条用于训练，100条作为测试

german.data-numeric的格式为，前24列为24个维度，最后一个为要打的标签（0，1），所以我们将数据和标签一起区分出来


```python
train_data=data[:900,:l-1]
train_lab=data[:900,l-1]-1
test_data=data[900:,:l-1]
test_lab=data[900:,l-1]-1
```

下面我们定义模型，模型很简单


```python
class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.fc=nn.Linear(24,2) # 由于24个维度已经固定了，所以这里写24
    def forward(self,x):
        out=self.fc(x)
        out=torch.sigmoid(out)
        return out
        
```

测试集上的准确率


```python
def test(pred,lab):
    t=pred.max(-1)[1]==lab
    return torch.mean(t.float())
```

下面就是对一些设置


```python
net=LR() 
criterion=nn.CrossEntropyLoss() # 使用CrossEntropyLoss损失
optm=torch.optim.Adam(net.parameters()) # Adam优化
epochs=1000 # 训练1000次

```

下面开始训练了


```python
for i in range(epochs):
    # 指定模型为训练模式，计算梯度
    net.train()
    # 输入值都需要转化成torch的Tensor
    x=torch.from_numpy(train_data).float()
    y=torch.from_numpy(train_lab).long()
    y_hat=net(x)
    loss=criterion(y_hat,y) # 计算损失
    optm.zero_grad() # 前一步的损失清零
    loss.backward() # 反向传播
    optm.step() # 优化
    if (i+1)%100 ==0 : # 这里我们每100次输出相关的信息
        # 指定模型为计算模式
        net.eval()
        test_in=torch.from_numpy(test_data).float()
        test_l=torch.from_numpy(test_lab).long()
        test_out=net(test_in)
        # 使用我们的测试函数计算准确率
        accu=test(test_out,test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i+1,loss.item(),accu))
```

    Epoch:100,Loss:0.6313,Accuracy：0.76
    Epoch:200,Loss:0.6065,Accuracy：0.79
    Epoch:300,Loss:0.5909,Accuracy：0.80
    Epoch:400,Loss:0.5801,Accuracy：0.81
    Epoch:500,Loss:0.5720,Accuracy：0.82
    Epoch:600,Loss:0.5657,Accuracy：0.81
    Epoch:700,Loss:0.5606,Accuracy：0.81
    Epoch:800,Loss:0.5563,Accuracy：0.81
    Epoch:900,Loss:0.5527,Accuracy：0.81
    Epoch:1000,Loss:0.5496,Accuracy：0.80
    

训练完成了，我们的准确度达到了80%
