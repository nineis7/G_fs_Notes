- [ ] Conv-BN-ReLU算子融合
- [ ] TVM算子融合
- [ ] OPU算子融合
- [ ] 算子融合常见总结



[TOC]



## 参考文献

1. 

# MLC

---

mlc指将算法从开发形态变成部署形态

mlc goal：

- 整合与最小化依赖
- 利用硬件自身自带的加速特性
- 优化



mlc关键元素：Tensor and TensorFunction，mlc所做的便是张量函数之间的组合变换（with different abstractions），张量函数可以是单一算子，也可以是多算子的融合，更可以是端到端的整个过程；抽象与实现，抽象指定做什么，实现指定怎么做。

call_tir(prim_func, inputs, shape, dtype) ，引入call_tir是因为元张量函数(即prim_func)本身具有destination passing的约定，而out的数组的开辟便由call_tir来完成。这是一个底层转高层的过程，将底层函数用标准计算图的形式来表示出来，如下图所示：

![side-effect](pics/MLC/side-effect.png)

下图不适用call_tir时计算图的表示比较混乱，出现了显式lv0等中间结果节点，即出现了side-effect。

具体来说，计算图通常具有以下性质：

- 框的每个输入边对应于操作的输入；
- 每个出边对应于操作的输出；
- 每个操作可以任意重新排序，直到边缘的拓扑顺序。

使用call_tir代码如下：

```python
lv0 = R.call_tir(linear0, (x, w0, b0), (1, 128), dtype="float32")

def lnumpy_call_tir(prim_func, inputs, shape, dtype):
    # A.开辟内存空间
    # B.遵循destination passing传统
    res = np.empty(shape, dtype=dtype)
    prim_func(*inputs, res)
    return res
```

使用dataflow来标注可以成为计算图并进行优化的部分

之所以要对{参数，中间节点，输出节点}做节点类别区分，是因为优化时可能会对中间节点进行消除（例如linear和relu合并成linear_relu之后linear向relu传的中间节点便被消除），而输出的结果节点可能会被其他计算图所调用，存在依赖关系，要进行全局分析。

`w_torch = torch.from_dlpack(x)`其中x type为tvm.nd.NDArray，x_torch为torchTensor，而x和x_torch共享内存，相当于不重新生成内存的情况下进行了两个类型对一片空间的指向。

将参数绑定为附加到 IRModule 的常量通常会降低API的复杂程度

```python
MyModuleWithParams = relax.transform.BindParams("main", nd_params)(MyModuleMixture)

def main(x: Tensor((1, 784), "float32")) -> Tensor(None, "float32", ndim = 2):
        # block 0
        with R.dataflow():
            lv0 = R.call_tir(linear0, (x, meta[relay.Constant][0], meta[relay.Constant][1]), (1, 128), dtype="float32")
            lv1 = R.call_tir("env.relu", (lv0,), (1, 128), dtype="float32")
            out = R.call_tir("env.linear", (lv1, meta[relay.Constant][2], meta[relay.Constant][3]), (1, 10), dtype="float32")
            R.output(out)
        return out
```

其中meta为dictionary，为模组中的元数据，w0,w1等参数就被转化为了`meta[relay.Constant][0]`，`meta[relay.Constant][1]`，由此可以达到减少参数的个数的目的。

trace记录了之前所作的所有变换操作

程序变换方式之一：随机变换，通过loop来将搜索空间内可能全部算出结果，求出最优解；



# AI编译器原理

---

AI编译器相较传统编译器更highlevel一点，是建基于传统编译器之上的，针对计算图的优化。

AI编译器通常会降低计算精度，因为深度学习对计算精度不那么敏感。

Frontend前端优化

- 节点级优化，如：Zero-dim-tensor elimination， Nop Elimination
- 块级优化，如代数简化、常量折叠、算子融合
- 数据流级优化，如CSE、DCE



一个变量的内存地址如果正好等于它的长度的整数倍，则称为自然对齐。

NCHW：通道优先，更适合需要对每个通道单独运算的操作，如MaxPooling，计算时需要的存储更多，适合GPU

NHWC：不同通道中同一位置优先顺序存储，更适合那些对不同通道同一位置的数据进行运算的操作，如Conv1x1，更适合多核CPU运算

![NHWC](pics/MLC/NHWC.png)

![NHWC_example](pics/MLC/NHWC_example.png)

相比于NCHW将所有数据读取完后一次性计算，NHWC每次计算出最终结果的一部分。

数据排布转换可通过`Tensor.reshape([N, H, W, C1, C0]).transpose([0, 3, 1, 2, 4])`样完成。

![Fractal Z](pics/MLC/Fractal Z.png)

z为横向优先，n为纵向优先

#### 内存分配和优化算法

1. 替代操作(Inplace operation)：如果一块内存不再需要，且下一个操作是element-wise（即元素在张量中对应位置相同），即可以用后一处内存来覆盖前面的内存；
2. 内存共享(Memory sharing)：两个数据使用内存大小相同，且前一个数据不再需要后后一个数据可以覆盖它。比如先后两个Conv操作，当前一个Conv的操作完全结束后后一个Conv操作便可以复用前一个Conv的内存空间。

#### AI编译器中的常量折叠

是将计算图中可以预先确定输出值的节点替换为常量，并对计算图进行一些结构简化的操作。

![常量折叠](pics/MLC/常量折叠.png)

![BN折叠](pics/MLC/BN折叠.png)

![BN折叠2](pics/MLC/BN折叠2.png)

常量折叠分类：

1. 如传统编译器类似的常量折叠
2. 常量折叠与数据形状shape有关，通过计算图已有信息推断出形状结果之后，用来代替原来的节点
3. 常量折叠与已知常量的代数化简有关

![TensorFlow常量折叠PASS](pics/MLC/TensorFlow常量折叠PASS.png)

#### 公共子表达式消除

![CSEinAI](pics/MLC/CSEinAI.png)

![CSEinAI2](pics/MLC/CSEinAI2.png)

其中采用逆后序表示是为了确定得到拓扑排序，访问节点的依赖节点都已被访问。（前序遍历不一定得到拓扑排序[ 请教拓扑排序中的一点疑问？](https://www.zhihu.com/question/28549004)）有向无环图的拓扑顺序就是所有顶点的逆后序排列。DFS逆转即为将本来结果{DBCA}转为{ACBD}

#### DCE

AI中的DCE通常应用在其他PASS之后，也就是其他PASS后出现不被访问的节点，此时删除这些死节点，避免不必要的计算和存储。例如无用的控制流、推理时删除仅训练相关的反向图/子图。

 算法：A. DFS找到死节点；B. 建立节点使用拓扑序列（节点间的调用和依赖关系），将节点更新并统一消除。

#### 代数化简

分为算术化简、运行化简、广播化简

算术化简分为{结合律化简、分配律化简、交换律化简}

![代数化简1](pics/MLC/代数化简1.png)

广播化简：位置替换（将shape相同的放在一起计算）