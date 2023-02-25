FROM ubuntu:18.04

RUN apt-get update --fix-missing

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY install/ubuntu1804_install_python.sh /install/ubuntu1804_install_python.sh
RUN bash /install/ubuntu1804_install_python.sh

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
RUN pip3 install termcolor
RUN pip3 install timm
RUN pip3 install yacs

# Deep learning framework dependencies
#RUN pip3 install torchvision==0.4.2
#RUN pip3 install onnx==1.6.0
RUN pip3 install transformers
RUN pip3 install datasets
#RUN pip3 install torch
#RUN pip3 install tensorflow==1.15
#RUN pip3 install keras=2.2.4
#RUN pip3 install h5py=2.10.0
# RUN pip3 install pytorch_pretrained_vit

RUN pip3 install ipdb

