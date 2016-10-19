# This script is supposed to be sourced by travis, since it exports environment
# variables.

export ROOT_TRAVIS_DIR="$(pwd)";
export TORCH_ROOT="$HOME/torch";

if [[ ! "$TORCH_ROOT" ]]; then clone https://github.com/torch/distro.git "$TORCH_ROOT" --recursive ; fi

cd "$TORCH_ROOT";
git pull && git submodule update && git submodule foreach git pull origin master;
cp -rf "$TORCH_ROOT" "${TORCH_ROOT}_${TORCH_LUA_VERSION}";
cd "${TORCH_ROOT}_${TORCH_LUA_VERSION}";
./install_torch.sh -b -s;
source "${TORCH_ROOT}_${TORCH_LUA_VERSION}/install/bin/torch-activate";
cd "$ROOT_TRAVIS_DIR";
