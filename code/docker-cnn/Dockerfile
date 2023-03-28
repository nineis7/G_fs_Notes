FROM ubuntu:16.04

ENV VER=00

RUN apt-get update && apt-get install -y --no-install-recommends \
        git make libgtest-dev cmake wget unzip libtinfo-dev libz-dev\
        libcurl4-openssl-dev libopenblas-dev g++ sudo\
        ocl-icd-opencl-dev libxml2-dev

# Python: basic dependencies
RUN sudo apt-get install -y software-properties-common
RUN sudo add-apt-repository -y ppa:jblgf0/python && sudo apt-get update && sudo rm /usr/bin/python3 && sudo apt-get install -y python3.7
RUN ln -s /usr/bin/python3.7 /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python
RUN sudo apt-get install -y python3-dev python3-setuptools gcc \
         libtinfo-dev zlib1g-dev build-essential cmake libboost-all-dev

RUN sudo apt-get install -y python3-pip
RUN sudo pip3 install --upgrade pip

# Deep learning framework dependencies
RUN sudo pip3 install absl-py==0.15.0 grpcio==1.42.0 importlib-metadata==4.11.3 markdown==3.3.4 numpy==1.21.5 protobuf==3.20.1 termcolor==1.1.0 werkzeug==0.16.1 install zipp==3.8.0 && \
    sudo pip3 install tensorflow==1.15
RUN sudo pip3 install torchvision==0.4.2
RUN sudo pip3 install onnx==1.6.0
#RUN sudo pip3 install keras==2.2.4
#RUN sudo pip3 install h5py==2.10.0

# TVM python dependencies
RUN sudo pip3 install numpy antlr4-python3-runtime==4.8 decorator attrs

# LLVM
RUN mkdir /llvm && wget http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
RUN tar xvf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
RUN rm clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && mv clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04/ /llvm/ 

# Other tools
RUN sudo apt-get install -y vim openssh-server

# Environment variables
ENV TVM_HOME=/workspace/opu-compiler/frontend
ENV PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}
ENV LLVM_DIR=/llvm/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04
ENV PATH=${LLVM_DIR}/bin:${PATH}

RUN echo "export VER=${VER}" >> /etc/profile
RUN echo "export TVM_HOME=${TVM_HOME}" >> /etc/profile
RUN echo "export PYTHONPATH=${PYTHONPATH}" >> /etc/profile
RUN echo "export LLVM_DIR=${LLVM_DIR}" >> /etc/profile
RUN echo "export PATH=${PATH}" >> /etc/profile
