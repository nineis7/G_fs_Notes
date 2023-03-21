# TVM docker cuda环境配置最新方案

2023-3-21

从外部docker创建环境的方式会导致tvm的各种依赖不满足条件的问题，采用tvm/docker中的GPU images，网址为[tvm/docker at main · apache/tvm (github.com)](https://github.com/apache/tvm/tree/main/docker)。