PyTorch是什么?
================

基于Python的科学计算包，服务于以下两种场景:

-  作为NumPy的替代品，可以使用GPU的强大计算能力
-  提供最大的灵活性和高速的深度学习研究平台
    

开始
---------------

### Tensors（张量）

Tensors与Numpy中的 ndarrays类似，但是在PyTorch中
Tensors 可以使用GPU进行计算.

```python
from __future__ import print_function
import torch
```

创建一个 5x3 矩阵, 但是未初始化:

```python
x = torch.empty(5, 3)
print(x)
```

    tensor([[0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000]])
    

创建一个随机初始化的矩阵:

```python
x = torch.rand(5, 3)
print(x)
```

    tensor([[0.6972, 0.0231, 0.3087],
            [0.2083, 0.6141, 0.6896],
            [0.7228, 0.9715, 0.5304],
            [0.7727, 0.1621, 0.9777],
            [0.6526, 0.6170, 0.2605]])
    

创建一个0填充的矩阵，数据类型为long:

```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])
    

创建tensor并使用现有数据初始化:

```python
x = torch.tensor([5.5, 3])
print(x)
```

    tensor([5.5000, 3.0000])
    

根据现有的张量创建张量。 这些方法将重用输入张量的属性，例如， dtype，除非设置新的值进行覆盖


```python
x = x.new_ones(5, 3, dtype=torch.double)      # new_* 方法来创建对象
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 覆盖 dtype!
print(x)                                      #  对象的size 是相同的，只是值和类型发生了变化
```

    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)
    tensor([[ 0.5691, -2.0126, -0.4064],
            [-0.0863,  0.4692, -1.1209],
            [-1.1177, -0.5764, -0.5363],
            [-0.4390,  0.6688,  0.0889],
            [ 1.3334, -1.1600,  1.8457]])
    

获取 size

***译者注：使用size方法与Numpy的shape属性返回的相同，张量也支持shape属性，后面会详细介绍***


```python
print(x.size())
```

    torch.Size([5, 3])


!!! Note
    ```torch.Size``` 返回值是 tuple类型, 所以它支持tuple类型的所有操作。

### Operations（操作）

操作有多种语法。 

我们将看一下加法运算。

加法：语法1

```python
y = torch.rand(5, 3)
print(x + y)
```

    tensor([[ 0.7808, -1.4388,  0.3151],
            [-0.0076,  1.0716, -0.8465],
            [-0.8175,  0.3625, -0.2005],
            [ 0.2435,  0.8512,  0.7142],
            [ 1.4737, -0.8545,  2.4833]])
    

加法：语法2

```python
print(torch.add(x, y))
```

    tensor([[ 0.7808, -1.4388,  0.3151],
            [-0.0076,  1.0716, -0.8465],
            [-0.8175,  0.3625, -0.2005],
            [ 0.2435,  0.8512,  0.7142],
            [ 1.4737, -0.8545,  2.4833]])
    

加法：提供输出tensor作为参数

```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

    tensor([[ 0.7808, -1.4388,  0.3151],
            [-0.0076,  1.0716, -0.8465],
            [-0.8175,  0.3625, -0.2005],
            [ 0.2435,  0.8512,  0.7142],
            [ 1.4737, -0.8545,  2.4833]])
    

加法：原地

```python
# adds x to y
y.add_(x)
print(y)
```

    tensor([[ 0.7808, -1.4388,  0.3151],
            [-0.0076,  1.0716, -0.8465],
            [-0.8175,  0.3625, -0.2005],
            [ 0.2435,  0.8512,  0.7142],
            [ 1.4737, -0.8545,  2.4833]])


!!! Note
    任何 以```_```结尾的操作都会用结果替换原变量.
    例如: ```x.copy_(y)```, ```x.t_()```, 都会改变 ```x```。

你可以使用与NumPy索引方式相同的操作来进行对张量的操作!

```python
print(x[:, 1])
```
    tensor([-2.0126,  0.4692, -0.5764,  0.6688, -1.1600])
    

改变大小：可以用```torch.view```改变张量的维度和大小

***译者注：torch.view 与Numpy的reshape类似***

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  #  size -1 从其他维度推断
print(x.size(), y.size(), z.size())
```
    torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
    

如果你有只有一个元素的张量，使用``.item()``来得到Python数据类型的数值

```python
x = torch.randn(1)
print(x)
print(x.item())
```

    tensor([-0.2368])
    -0.23680149018764496
    

**Read later:**

  100+ Tensor operations, including transposing, indexing, slicing,
  mathematical operations, linear algebra, random numbers, etc.,
  are described
  `here <https://pytorch.org/docs/torch>`_.

NumPy 转换
------------

Converting a Torch Tensor to a NumPy array and vice versa is a breeze.

The Torch Tensor and NumPy array will share their underlying memory
locations, and changing one will change the other.

### Torch Tensor 转成 NumPy Array

```python
a = torch.ones(5)
print(a)
```

    tensor([1., 1., 1., 1., 1.])
    

```python
b = a.numpy()
print(b)
```

    [1. 1. 1. 1. 1.]
    

See how the numpy array changed in value.


```python
a.add_(1)
print(a)
print(b)
```

    tensor([2., 2., 2., 2., 2.])
    [2. 2. 2. 2. 2.]
    

### NumPy Array 转成 Torch Tensor

使用from_numpy自动转化

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

    [2. 2. 2. 2. 2.]
    tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    


所有的 Tensor 类型默认都是基于CPU， CharTensor 类型不支持到
NumPy 的转换.  

CUDA 张量
------------

使用```.to``` 方法 可以将Tensor移动到任何设备中


```python
# is_available 函数判断是否有cuda可以使用
# ``torch.device``将张量移动到指定的设备中
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
    x = x.to(device)                       # 或者直接使用``.to("cuda")``将张量移动到cuda中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 也会对变量的类型做更改
```
    tensor([0.7632], device='cuda:0')
    tensor([0.7632], dtype=torch.float64)
    
