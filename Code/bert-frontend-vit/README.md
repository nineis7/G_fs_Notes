# bert-frontend

reference:

tvm: [https://github.com/apache/tvm](https://github.com/apache/tvm)

bert: [https://tvm.apache.org/2020/07/14/bert-pytorch-tvm](https://tvm.apache.org/2020/07/14/bert-pytorch-tvm)

vit: [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)

### Build
```
git clone --recursive https://github.com/OPU-Lab/bert-frontend.git
```
For validation with transformer:
```
git checkout fuse_yw
git submodule update --init --recursive
```
build with docker
```
cd docker
sh build.sh docker-name
sh run.sh /absolute/mount/path docker-name
```
example:
```
sh build.sh tvm-bert
sh run.sh $(pwd)/../.. tvm-bert
```
build tvm
```
cd bert-frontend/tvm
mkdir build
cp cmake/config.cmake build/
cd build
cmake ..
make -j4
```

### Test
run both tvm frontend and json ir validation:
```
make script
```

### USE PAPI (with latest tvm repo)
https://tvm.apache.org/docs/how_to/profile/papi.html

update cmake to >= 3.23
```
sudo apt update
sudo apt install build-essential libtool autoconf unzip wget

sudo apt remove --purge --auto-remove cmake

version=3.24
build=1
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/

./bootstrap
make -j$(nproc)
sudo make install
```
build tvm with PAPI
```
git clone https://ming-hsuan-tu@bitbucket.org/icl/papi.git
git checkout tags/papi-6-0-0-1-t -b <branch>

cd papi/src
./configure --prefix=/home/papi_install
make && make install

# modify path to papi pkgconfig in config.cmake
SET(USE_PAPI /home/papi_install/lib/pkgconfig)
```

need to add extended privileges to the container with ```--privileged```, then:\
ref: https://stackoverflow.com/questions/54049196/getting-read-only-filesystem-error-inside-a-docker-container

```
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```