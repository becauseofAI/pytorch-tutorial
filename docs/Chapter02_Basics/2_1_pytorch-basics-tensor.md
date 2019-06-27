
# PyTorch 基础 : 张量
在第一章中我们已经通过官方的入门教程对PyTorch有了一定的了解，这一章会详细介绍PyTorch 里面的基础知识。
全部掌握了这些基础知识，在后面的应用中才能更加快速进阶，如果你已经对PyTorch有一定的了解，可以跳过此章


```python
# 首先要引入相关的包
import torch
import numpy as np
#打印一下版本
torch.__version__
```




    '1.0.0'



## 张量(Tensor)
张量的英文是Tensor，它是PyTorch里面基础的运算单位,与Numpy的ndarray相同都表示的是一个多维的矩阵。
与ndarray的最大区别就是，PyTorch的Tensor可以在 GPU 上运行，而 numpy 的 ndarray 只能在 CPU 上运行，在GPU上运行大大加快了运算速度。

下面我们生成一个简单的张量


```python
x = torch.rand(2, 3)
x
```




    tensor([[0.6904, 0.7419, 0.8010],
            [0.1722, 0.2442, 0.8181]])



以上生成了一个，2行3列的的矩阵，我们看一下他的大小：


```python
# 可以使用与numpy相同的shape属性查看
print(x.shape)
# 也可以使用size()函数，返回的结果都是相同的
print(x.size())
```

    torch.Size([2, 3])
    torch.Size([2, 3])
    

张量（Tensor）是一个定义在一些向量空间和一些对偶空间的笛卡儿积上的多重线性映射，其坐标是|n|维空间内，有|n|个分量的一种量， 其中每个分量都是坐标的函数， 而在坐标变换时，这些分量也依照某些规则作线性变换。r称为该张量的秩或阶（与矩阵的秩和阶均无关系）。 (来自百度百科)

下面我们来生成一些多维的张量：


```python
y=torch.rand(2,3,4,5)
print(y.size())
y
```

    torch.Size([2, 3, 4, 5])
    




    tensor([[[[0.9071, 0.0616, 0.0006, 0.6031, 0.0714],
              [0.6592, 0.9700, 0.0253, 0.0726, 0.5360],
              [0.5416, 0.1138, 0.9592, 0.6779, 0.6501],
              [0.0546, 0.8287, 0.7748, 0.4352, 0.9232]],
    
             [[0.0730, 0.4228, 0.7407, 0.4099, 0.1482],
              [0.5408, 0.9156, 0.6554, 0.5787, 0.9775],
              [0.4262, 0.3644, 0.1993, 0.4143, 0.5757],
              [0.9307, 0.8839, 0.8462, 0.0933, 0.6688]],
    
             [[0.4447, 0.0929, 0.9882, 0.5392, 0.1159],
              [0.4790, 0.5115, 0.4005, 0.9486, 0.0054],
              [0.8955, 0.8097, 0.1227, 0.2250, 0.5830],
              [0.8483, 0.2070, 0.1067, 0.4727, 0.5095]]],
    
    
            [[[0.9438, 0.2601, 0.2885, 0.5457, 0.7528],
              [0.2971, 0.2171, 0.3910, 0.1924, 0.2570],
              [0.7491, 0.9749, 0.2703, 0.2198, 0.9472],
              [0.1216, 0.6647, 0.8809, 0.0125, 0.5513]],
    
             [[0.0870, 0.6622, 0.7252, 0.4783, 0.0160],
              [0.7832, 0.6050, 0.7469, 0.7947, 0.8052],
              [0.1755, 0.4489, 0.0602, 0.8073, 0.3028],
              [0.9937, 0.6780, 0.9425, 0.0059, 0.0451]],
    
             [[0.3851, 0.8742, 0.5932, 0.4899, 0.8354],
              [0.8577, 0.3705, 0.0229, 0.7097, 0.7557],
              [0.1505, 0.3527, 0.0843, 0.0088, 0.8741],
              [0.6041, 0.8797, 0.6189, 0.9495, 0.1479]]]])



在同构的意义下，第零阶张量 （r = 0） 为标量 （Scalar），第一阶张量 （r = 1） 为向量 （Vector）， 第二阶张量 （r = 2） 则成为矩阵 （Matrix），第三阶以上的统称为多维张量。

其中要特别注意的就是标量，我们先生成一个标量：



```python
#我们直接使用现有数字生成
scalar =torch.tensor(3.1433223)
print(scalar)
#打印标量的大小
scalar.size()
```

    tensor(3.1433)
    




    torch.Size([])



对于标量，我们可以直接使用 .item() 从中取出其对应的python对象的数值


```python
scalar.item()
```




    3.143322229385376



特别的：如果张量中只有一个元素的tensor也可以调用`tensor.item`方法


```python
tensor = torch.tensor([3.1433223]) 
print(tensor)
tensor.size()
```

    tensor([3.1433])
    




    torch.Size([1])




```python
tensor.item()
```




    3.143322229385376



### 基本类型
Tensor的基本数据类型有五种：
- 32位浮点型：torch.FloatTensor。 (默认)
- 64位整型：torch.LongTensor。
- 32位整型：torch.IntTensor。
- 16位整型：torch.ShortTensor。
- 64位浮点型：torch.DoubleTensor。

除以上数字类型外，还有
byte和chart型


```python
long=tensor.long()
long
```




    tensor([3])




```python
half=tensor.half()
half
```




    tensor([3.1426], dtype=torch.float16)




```python
int_t=tensor.int()
int_t
```




    tensor([3], dtype=torch.int32)




```python
flo = tensor.float()
flo
```




    tensor([3.1433])




```python
short = tensor.short()
short
```




    tensor([3], dtype=torch.int16)




```python
ch = tensor.char()
ch
```




    tensor([3], dtype=torch.int8)




```python
bt = tensor.byte()
bt
```




    tensor([3], dtype=torch.uint8)



### Numpy转换
使用numpy方法将Tensor转为ndarray


```python
a = torch.randn((3, 2))
# tensor转化为numpy
numpy_a = a.numpy()
print(numpy_a)
```

    [[ 0.46819344  1.3774964 ]
     [ 0.9491934   1.4543315 ]
     [-0.42792308  0.99790514]]
    

numpy转化为Tensor



```python
torch_a = torch.from_numpy(numpy_a)
torch_a
```




    tensor([[ 0.4682,  1.3775],
            [ 0.9492,  1.4543],
            [-0.4279,  0.9979]])



***Tensor和numpy对象共享内存，所以他们之间的转换很快，而且几乎不会消耗什么资源。但这也意味着，如果其中一个变了，另外一个也会随之改变。***

### 设备间转换
一般情况下可以使用.cuda方法将tensor移动到gpu，这步操作需要cuda设备支持


```python
cpu_a=torch.rand(4, 3)
cpu_a.type()
```




    'torch.FloatTensor'




```python
gpu_a=cpu_a.cuda()
gpu_a.type()
```




    'torch.cuda.FloatTensor'



使用.cpu方法将tensor移动到cpu


```python
cpu_b=gpu_a.cpu()
cpu_b.type()
```




    'torch.FloatTensor'



如果我们有多GPU的情况，可以使用to方法来确定使用那个设备，这里只做个简单的实例：


```python
#使用torch.cuda.is_available()来确定是否有cuda设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#将tensor传送到设备
gpu_b=cpu_b.to(device)
gpu_b.type()
```

    cuda
    




    'torch.cuda.FloatTensor'



### 初始化
Pytorch中有许多默认的初始化方法可以使用


```python
# 使用[0,1]均匀分布随机初始化二维数组
rnd = torch.rand(5, 3)
rnd
```




    tensor([[0.3804, 0.0297, 0.5241],
            [0.4111, 0.8887, 0.4642],
            [0.7302, 0.5913, 0.7182],
            [0.3048, 0.8055, 0.2176],
            [0.6195, 0.1620, 0.7726]])




```python
##初始化，使用1填充
one = torch.ones(2, 2)
one
```




    tensor([[1., 1.],
            [1., 1.]])




```python
##初始化，使用0填充
zero=torch.zeros(2,2)
zero
```




    tensor([[0., 0.],
            [0., 0.]])




```python
#初始化一个单位矩阵，即对角线为1 其他为0
eye=torch.eye(2,2)
eye
```




    tensor([[1., 0.],
            [0., 1.]])



### 常用方法
PyTorch中对张量的操作api 和 NumPy 非常相似，如果熟悉 NumPy 中的操作，那么 他们二者 基本是一致的：


```python
x = torch.randn(3, 3)
print(x)
```

    tensor([[ 0.6922, -0.4824,  0.8594],
            [ 0.4509, -0.8155, -0.0368],
            [ 1.3533,  0.5545, -0.0509]])
    


```python
# 沿着行取最大值
max_value, max_idx = torch.max(x, dim=1)
print(max_value, max_idx)
```

    tensor([0.8594, 0.4509, 1.3533]) tensor([2, 0, 0])
    


```python
# 每行 x 求和
sum_x = torch.sum(x, dim=1)
print(sum_x)
```

    tensor([ 1.0692, -0.4014,  1.8568])
    


```python
y=torch.randn(3, 3)
z = x + y
print(z)
```

    tensor([[-0.3821, -2.6932, -1.3884],
            [ 0.7468, -0.7697, -0.0883],
            [ 0.7688, -1.3485,  0.7517]])
    

正如官方60分钟教程中所说，以_为结尾的，均会改变调用值


```python
# add 完成后x的值改变了
x.add_(y)
print(x)
```

    tensor([[-0.3821, -2.6932, -1.3884],
            [ 0.7468, -0.7697, -0.0883],
            [ 0.7688, -1.3485,  0.7517]])
    

张量的基本操作都介绍的的差不多了，下一章介绍PyTorch的自动求导机制
