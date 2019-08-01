

```python
%matplotlib inline
```


数据并行（选读）
==========================
**Authors**: [Sung Kim](https://github.com/hunkim) and [Jenny Kang](https://github.com/jennykang)

在这个教程里，我们将学习如何使用 ``DataParallel`` 来使用多GPU。 

PyTorch非常容易就可以使用多GPU，用如下方式把一个模型放到GPU上：

```python

    device = torch.device("cuda:0")
    model.to(device)
```
 GPU:
然后复制所有的张量到GPU上：
```python

    mytensor = my_tensor.to(device)
```
请注意，只调用``my_tensor.to(device)``并没有复制张量到GPU上，而是返回了一个copy。所以你需要把它赋值给一个新的张量并在GPU上使用这个张量。

在多GPU上执行前向和反向传播是自然而然的事。
但是PyTorch默认将只使用一个GPU。

使用``DataParallel``可以轻易的让模型并行运行在多个GPU上。


```python

    model = nn.DataParallel(model)
```
这才是这篇教程的核心，接下来我们将更详细的介绍它。



导入和参数
----------------------

导入PyTorch模块和定义参数。





```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100
```

Device





```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

虚拟数据集
-------------

制作一个虚拟（随机）数据集，
你只需实现 `__getitem__`





```python
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
```

简单模型
------------
作为演示，我们的模型只接受一个输入，执行一个线性操作，然后得到结果。
说明：``DataParallel``能在任何模型（CNN，RNN，Capsule Net等）上使用。


我们在模型内部放置了一条打印语句来打印输入和输出向量的大小。

请注意批次的秩为0时打印的内容。





```python
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
```

创建一个模型和数据并行
-----------------------------

这是本教程的核心部分。

首先，我们需要创建一个模型实例和检测我们是否有多个GPU。
如果有多个GPU，使用``nn.DataParallel``来包装我们的模型。
然后通过m``model.to(device)``把模型放到GPU上。




```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)
```




    Model(
      (fc): Linear(in_features=5, out_features=2, bias=True)
    )



运行模型
-------------

现在可以看到输入和输出张量的大小。





```python
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```

    	In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    	In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    	In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    	In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
    

结果
-------

当没有或者只有一个GPU时，对30个输入和输出进行批处理，得到了期望的一样得到30个输入和输出，但是如果你有多个GPU，你得到如下的结果。


2 GPUs
~

If you have 2, you will see:

.. code:: bash

    # on 2 GPUs
    Let's use 2 GPUs!
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])

3 GPUs
~

If you have 3 GPUs, you will see:

.. code:: bash

    Let's use 3 GPUs!
        In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
        In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
        In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
        In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
        In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
        In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
        In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])

8 GPUs
~~

If you have 8, you will see:

.. code:: bash

    Let's use 8 GPUs!
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
        In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
        In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
        In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
        In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
        In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])




总结
-------

DataParallel会自动的划分数据，并将作业发送到多个GPU上的多个模型。
并在每个模型完成作业后，收集合并结果并返回。

更多信息请看这里：
https://pytorch.org/tutorials/beginner/former\_torchies/parallelism\_tutorial.html.



