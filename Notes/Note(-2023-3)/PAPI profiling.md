[TOC]

# PAPI profiling



### PAPI 安装

PAPI目前需要cmake version>=3.23, 这里选择版本为3.24，由于是从tvm/docker中安装，故只需修改[tvm/docker/install/ubuntu_install_cmake_source.sh](https://github.com/apache/tvm/blob/main/docker/install/ubuntu_install_cmake_source.sh)，将版本改为3.24.1重新build docker即可。

PAPI的安装也有两种方法：

#### 法1：修改Dockerfile.ci_gpu

第一种是如cmake一样，在Dockerfile.ci_gpu中添加[tvm/docker/install/ubuntu_install_papi.sh](https://github.com/apache/tvm/blob/main/docker/install/ubuntu_install_papi.sh)即可（code内的dockerfile没有加上papi，需要去掉注释），有两处需要修改的地方：

```
1. In tvm/docker/install/ubuntu_install_papi.sh
export PAPI_CUDA_ROOT=/usr/local/cuda -> export PAPI_CUDA_ROOT=/usr/local/cuda本地的版本号
cuda本地版本号需要进入/usr/local下查看

2. In Dockerfile.ci_gpu
RUN bash /install/ubuntu_install_papi.sh "cuda rocm" 将rocm删去，不需要
```

等待docker build完成。

#### 法2：手动安装

手动安装的过程与法1相同：

```
git clone --branch papi-6-0-0-1-t https://bitbucket.org/icl/papi.git
cd papi/src
export PAPI_CUDA_ROOT=/usr/local/cuda版本号（需自行查看）
# export PAPI_ROCM_ROOT=/opt/rocm 可以不安装
./configure --with-components="cuda"
make && make install
```

这里`./configure --with-components="cuda"` 即为了能够在gpu下进行PAPI profiling。

安装完PAPI后在config.cmake中修改`set(USE_PAPI ON)`并重新build tvm。

### PAPI使用

在[tvm官方文档](https://tvm.apache.org/docs/how_to/profile/papi.html)中给出了PAPI profiling的示例

![PAPI_tu](../assets/pics\PAPI\PAPI_tu.png)

其中`mod, params = mlp.get_workload(1)`的作用与

```python
mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
```

作用相同，mod为IRmodule，其并没有经过参数绑定，故main function的input包括了所有的权重和偏置等参数，以下是mlp的mod和params，mlp workload来自tvm/python/tvm/realy/testing/mlp.py：

```
def @main(%data: Tensor[(1, 1, 28, 28), float32] /* ty=Tensor[(1, 1, 28, 28), float32] */, %fc1_weight: Tensor[(128, 784), float32] /* ty=Tensor[(128, 784), float32] */, %fc1_bias: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %fc2_weight: Tensor[(64, 128), float32] /* ty=Tensor[(64, 128), float32] */, %fc2_bias: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %fc3_weight: Tensor[(10, 64), float32] /* ty=Tensor[(10, 64), float32] */, %fc3_bias: Tensor[(10), float32] /* ty=Tensor[(10), float32] */) -> Tensor[(1, 10), float32] {
  %0 = nn.batch_flatten(%data) /* ty=Tensor[(1, 784), float32] */;
  %1 = nn.dense(%0, %fc1_weight, units=128) /* ty=Tensor[(1, 128), float32] */;
  %2 = nn.bias_add(%1, %fc1_bias, axis=-1) /* ty=Tensor[(1, 128), float32] */;
  %3 = nn.relu(%2) /* ty=Tensor[(1, 128), float32] */;
  %4 = nn.dense(%3, %fc2_weight, units=64) /* ty=Tensor[(1, 64), float32] */;
  %5 = nn.bias_add(%4, %fc2_bias, axis=-1) /* ty=Tensor[(1, 64), float32] */;
  %6 = nn.relu(%5) /* ty=Tensor[(1, 64), float32] */;
  %7 = nn.dense(%6, %fc3_weight, units=10) /* ty=Tensor[(1, 10), float32] */;
  %8 = nn.bias_add(%7, %fc3_bias, axis=-1) /* ty=Tensor[(1, 10), float32] */;
  nn.softmax(%8) /* ty=Tensor[(1, 10), float32] */
}
```

```
{'fc1_weight': <tvm.nd.NDArray shape=(128, 784), cpu(0)>
array([[ 0.0079186 ,  0.03490832,  0.01667042, ..., -0.01185768,
         0.05561841,  0.05159181],
       [-0.064497  , -0.05574198, -0.03176317, ..., -0.02593908,
        -0.06998283, -0.043977  ],
       [-0.02303805, -0.01052136,  0.01475026, ..., -0.08096418,
        -0.06978533, -0.04436897],
       ...,
       [ 0.05179032, -0.03124079,  0.07870501, ...,  0.04822593,
         0.03519785, -0.00536821],
       [ 0.07242752, -0.02499434, -0.03771518, ..., -0.0059433 ,
        -0.01991812,  0.04865525],
       [ 0.03747288, -0.03090882,  0.02881455, ..., -0.05079785,
        -0.03773598, -0.0132189 ]], dtype=float32), 'fc1_bias': <tvm.nd.NDArray shape=(128,), cpu(0)>
        
        ...
```

params为字典，前者为op_name，后者为类型和值。

目前PAPI均可进行llvm和cuda的profiling，也就是target和dev可以是llvm+cpu，也可以是cuda+cuda。



```python
data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"), device=dev)
```

这一句需要针对不同模型自定义输入shape和datatype，对于gpt使用

```python
data = tvm.nd.array(np.random.rand(16, 512).astype(int), device=dev)
```

如果自定义有报错，针对报错的datatype和shape自定调整。

在tvm中文文档中代码有一处bug而英文最新文档已更新，即vm.profile里传入的是data而不是[data]，无需添加中括号。该问题已在[Error using VM to profile : Downcast from runtime.ADT to runtime.NDArray failed - Questions - Apache TVM Discuss](https://discuss.tvm.apache.org/t/error-using-vm-to-profile-downcast-from-runtime-adt-to-runtime-ndarray-failed/13557)中得到反馈并解决。



实际的gpt PAPI profiling代码参见[papi.py](https://github.com/nineis7/tvm-gpt/blob/main/tutorials/papi.py)，结果参见[PAPI_profiling_gpt_cuda.txt](https://github.com/nineis7/tvm-gpt/blob/main/artifacts/PAPI_profiling/PAPI_profiling_gpt_cuda.txt)和[PAPI_profiling_gpt_llvm.txt](https://github.com/nineis7/tvm-gpt/blob/main/artifacts/PAPI_profiling/PAPI_profiling_gpt_llvm.txt)：

![PAPI_profiling_gpt_llvm](../assets/pics\PAPI\PAPI_profiling_gpt_llvm.png)

![PAPI_profiling_gpt_cuda](../assets/pics\PAPI\PAPI_profiling_gpt_cuda.png)