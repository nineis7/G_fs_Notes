set -e
set -u
set -o pipefail

v=3.14
version=3.14.7
wget https://cmake.org/files/v${v}/cmake-${version}.tar.gz
tar xvf cmake-${version}.tar.gz
cd cmake-${version}
./bootstrap
make -j$(nproc)
make install
cd ..
rm -rf cmake-${version} cmake-${version}.tar.gz
