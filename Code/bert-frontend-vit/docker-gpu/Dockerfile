FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu18.04

COPY utils/apt-install-and-clear.sh /usr/local/bin/apt-install-and-clear

RUN apt-get update --fix-missing

RUN apt-install-and-clear -y bzip2 wget sudo binutils git

#RUN  wget -q http://developer.download.nvidia.com/compute/redist/cudnn/v7.6.0/cudnn-10.0-linux-x64-v7.6.0.64.tgz && \
#    tar --no-same-owner -xzf cudnn-10.0-linux-x64-v7.6.0.64.tgz -C /usr/local && \
#    rm cudnn-10.0-linux-x64-v7.6.0.64.tgz && \
#    ldconfig

COPY install/ubuntu_install_conda.sh /install/ubuntu_install_conda.sh
RUN bash /install/ubuntu_install_conda.sh

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV CONDA_BLD_PATH /tmp

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

# Globally disable pip cache
RUN pip config set global.cache-dir false

COPY install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh

# LLVM
RUN mkdir /llvm && wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
RUN tar xvf clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
RUN rm clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && mv clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/ /llvm/ 

# Other tools
RUN apt-get install -y vim 

RUN mkdir /workspace

# Environment variables
ENV TVM_HOME=/workspace/bert-frontend/tvm
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
ENV LLVM_DIR=/llvm/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04

# Vision transformer pytorch
RUN pip3 install vit-pytorch

# Deep learning framework dependencies
#RUN pip3 install torchvision==0.4.2
#RUN pip3 install onnx==1.6.0
RUN pip3 install transformers
#RUN pip3 install torch
#RUN pip3 install tensorflow==1.15
#RUN pip3 install keras=2.2.4
#RUN pip3 install h5py=2.10.0

RUN pip3 install ipdb
RUN apt-get install -y zsh
RUN pip3 install pandas

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib:/usr/lib/x86_64-linux-gnu
COPY install/ubuntu_install_cmake_source.sh /install/ubuntu_install_cmake_source.sh
RUN bash /install/ubuntu_install_cmake_source.sh

RUN pip3 install onnx datasets
# swin
RUN pip3 install termcolor timm yacs
