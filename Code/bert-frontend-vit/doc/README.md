```
mkdir ws
cd ws
git clone https://github.com/OPU-Lab/bert-frontend.git --recursive -b vit
git clone https://github.com/OPU-Lab/Compiler-Backend.git
```
run docker
```
sh build.sh
```


BERT

### compiler frontend

check ``tvm/src/relay/transform/hw_info.h`` and change ``#define VIT`` to ``#define BERT``

build
```
cd tvm
mkdir build
cd build
cp ../cmake/config.cmake .
cmake ..
make -j9
```

run frontend
```
python3 driver.py 2>&1 | tee driver_bert.log
```

run json ir validation
```
python3 utils/run_ir_json.py --config=artifacts/OPU_IR_bert.json --input_sequence_length=64 
```


### compiler backend

install boost
```
wget -O boost_1_55_0.tar.gz https://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz/download
tar xzvf boost_1_55_0.tar.gz
cd boost_1_55_0/
./bootstrap.sh
./b2 install

```
update env variables
```
export LLVM_INCLUDE_DIRS=${LLVM_DIR}/include
export LLVM_LIBRARY_DIRS=${LLVM_DIR}/lib
```
build
```
mkdir build;cd build;cmake ..;make -j4
```


run backend under ``ws``
```
./Compiler-Backend/build/backend -i bert-frontend/artifacts/OPU_IR_bert.json -opt 1 -plc 8
./Compiler-Backend/data-layout-generator/build/data-layout-gen dram_layout.json bert-frontend/dump_q/
./Compiler-Backend/build/backend -i bert-frontend/artifacts/OPU_IR_bert.json -opt 2 -plc 8
./Compiler-Backend/data-layout-generator/build/merge-dram-bin 
```
