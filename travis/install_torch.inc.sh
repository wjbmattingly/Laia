# This script is supposed to be sourced by travis, since it

export ROOT_TRAVIS_DIR=$(pwd);
export TORCH_ROOT="$HOME/torch";
export TORCH_INSTALL="$HOME/torch/install";

# Install OpenBlas, if necessary
ls $HOME/OpenBlasInstall/lib || (
    cd /tmp/;
    git clone https://github.com/xianyi/OpenBLAS.git -b master;
    cd OpenBLAS;
    make clean;
    make USE_THREAD=0 USE_THREADS=0 USE_OPENMP=0 NO_AFFINITY=1 -j$(getconf _NPROCESSORS_ONLN) 2>/dev/null >/dev/null
);
make USE_THREAD=0 USE_THREADS=0 USE_OPENMP=0 NO_AFFINITY=0 PREFIX=$HOME/OpenBlasInstall install;

git clone https://github.com/torch/distro.git "$TORCH_ROOT" --recursive
cd "$TORCH_ROOT"
git submodule update --init --recursive;
mkdir build && cd build;
export CMAKE_LIBRARY_PATH="$HOME/OpenBlasInstall/include:$HOME/OpenBlasInstall/lib:$CMAKE_LIBRARY_PATH";
cmake .. -DCMAKE_INSTALL_PREFIX="${TORCH_INSTALL}" -DCMAKE_BUILD_TYPE=Release -DWITH_${TORCH_LUA_VERSION}=ON
make && make install
if [[ $TORCH_LUA_VERSION != 'LUAJIT21' && $TORCH_LUA_VERSION != 'LUAJIT20' ]]; then
    ${TORCH_INSTALL}/bin/luarocks install luaffi;
fi
source "${TORCH_INSTALL}/bin/torch-activate";
cd "$ROOT_TRAVIS_DIR";
