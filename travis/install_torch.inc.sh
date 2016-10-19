# This script is supposed to be sourced by travis, since it exports environment
# variables.

export ROOT_TRAVIS_DIR="$(pwd)";
export TORCH_ROOT="$HOME/torch";

# Make sure that the torch directory is a git clone.
[[ ! -d "$TORCH_ROOT/.git" ]] && rm -rf "$TORCH_ROOT";
# If the torch directory does not exist, clone a new repo.
[[ ! -d "$TORCH_ROOT" ]] && git clone https://github.com/torch/distro.git "$TORCH_ROOT" --recursive ;
# Update torch
cd "$TORCH_ROOT";
git pull && git submodule update && git submodule foreach git pull origin master;
# Copy torch for a clean install with a particular LUA_VERSION
cp -a "$TORCH_ROOT" "${TORCH_ROOT}_${TORCH_LUA_VERSION}";
cd "${TORCH_ROOT}_${TORCH_LUA_VERSION}";
./install.sh -b -s;
# Set environment variables.
source "${TORCH_ROOT}_${TORCH_LUA_VERSION}/install/bin/torch-activate";
# Back to travis dir.
cd "$ROOT_TRAVIS_DIR";
