# TVM d2l

computation：计算图，使用te.op构建，op={placeholder, compute...}

schedule：执行程序的顺序，包括读数据，优化操作等

build：构建可执行程序

tvm进行tvmndarray与npArray转化的方式如下：

```python
x = np.ones(2)
y = tvm.nd.array(x) #转化为tvm.nd.array
y.asnumpy() #再转化回去
```

Implementing an operator using TVM has three steps: 

1. <u>Declare the computation</u> by specifying input and output shapes and how each output element is computed. 

2. <u>Create a schedule</u> to (hopefully) fully utilize the machine resources. 

3. <u>Compile</u> to the hardware target. 

In addition, we can save the compiled module into disk so we can load it back later.

runtime to run the inference

两种提高泛化性的方法：`n = te.var(name='n')` 来非限制<u>长度</u>；传入dtype来非限制<u>类型</u>。

