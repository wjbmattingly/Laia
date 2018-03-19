# Laia: A deep learning toolkit for HTR

[![Build Status](https://travis-ci.com/jpuigcerver/Laia.svg?token=HF64eTvPxEUcjjUPXpgm&branch=master)](https://travis-ci.com/jpuigcerver/Laia)

Laia is a deep learning toolkit to transcribe handwritten text images.

If you find this toolkit useful in your research, please cite:

```
@misc{laia2016,
  author = {Joan Puigcerver and
            Daniel Martin-Albo and
            Mauricio Villegas},
  title = {Laia: A deep learning toolkit for HTR},
  year = {2016},
  publisher = {GitHub},
  note = {GitHub repository},
  howpublished = {\url{https://github.com/jpuigcerver/Laia}},
}
```

## Installation

Laia is implemented in [Torch](http://torch.ch/), and depends on the following:

- [CUDA >=7.5](https://developer.nvidia.com/cuda-downloads)
- [cuDNN >= 5.1](https://developer.nvidia.com/cudnn)

Note that currently we only support GPU. You need to use NVIDIA's cuDNN library. Register first for the CUDA Developer Program (it's free) and download the library from [NVIDIA's website](https://developer.nvidia.com/cudnn).

Once Torch is installed the following luarocks are required:

- You also have to  install the [cuDNN bindings for Torch](https://github.com/soumith/cudnn.torch)
- [Baidu's CTC](https://github.com/baidu-research/warp-ctc)
- [imgdistort](https://github.com/jpuigcerver/imgdistort)

And execute `luarocks install https://raw.githubusercontent.com/jpuigcerver/Laia/master/rocks/laia-scm-1.rockspec`.


## Installation via docker

To ease the installation, there is a public [docker image for Laia](https://hub.docker.com/r/mauvilsa/laia/). To use it first install [docker](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/releases), and configure docker so that it can be executed without requiring sudo, see [docker linux postinstall](https://docs.docker.com/engine/installation/linux/linux-postinstall/). Then the installation of Laia consists of first pulling the image and tagging it as laia:active.

    docker pull mauvilsa/laia:[SOME_TAG]
    docker tag mauvilsa/laia:[SOME_TAG] laia:active

Replace SOME_TAG with one of the tags available [here](https://hub.docker.com/r/mauvilsa/laia/tags/). Then copy the command line interface script to some directory in your path for easily use from the host.

    mkdir -p $HOME/bin
    docker run --rm -u $(id -u):$(id -g) -v $HOME:$HOME laia:active bash -c "cp /usr/local/bin/laia-docker $HOME/bin"

After this, all Laia commands can be executed by using the laia-docker command. For further details run.

    laia-docker --help


## Usage

### Training a Laia model using CTC:

Create an "empty" model using:
```bash
laia-create-model \
    "$CHANNELS" "$INPUT_HEIGHT" "$((NUM_SYMB+1))" "$MODEL_DIR/model.t7";
```
Or if installed via docker:
```bash
laia-docker create-model \
    "$CHANNELS" "$INPUT_HEIGHT" "$((NUM_SYMB+1))" "$MODEL_DIR/model.t7";
```
Positional arguments:
- `$CHANNELS`: number of channels of the input images.
- `$INPUT_HEIGHT`: height of the input images. Note: **ALL** image must have the same height.
- `$((NUM_SYMB+1))`: number of output symbols. Note: Include **ONE** additional element for the CTC blank symbol.
- `$MODEL_DIR/model.t7`: path to the output model.

For optional arguments check `laia-create-model -h` or `laia-create-model -H`.

Train the model using:
```bash
laia-train-ctc \
    "$MODEL_DIR/model.t7" \
    "$SYMBOLS_TABLE" \
    "$TRAIN_LST" "$TRAIN_GT" "$VALID_LST" "$VALID_GT";
```
Or if installed via docker:
```bash
laia-docker train-ctc \
    "$MODEL_DIR/model.t7" \
    "$SYMBOLS_TABLE" \
    "$TRAIN_LST" "$TRAIN_GT" "$VALID_LST" "$VALID_GT";
```
Positional arguments:
- `$MODEL_DIR/model.t7` is the path to the input model or checkpoint for training.
- `$SYMBOLS_TABLE` is the list of training symbols and their id.
- `$TRAIN_LST` is a file containing a list of images for training.
- `$TRAIN_GT` is a file containing the list of training transcripts.
- `$VALID_LST` is a file containing a list of images for validation.
- `$VALID_GT` is a file containing the list of validation transcripts.

For optional arguments check `laia-train-ctc -h` or `laia-create-model -H`.

### Transcribing

```bash
laia-decode "$MODEL_DIR/model.t7" "$TEST_LST";
```
Or if installed via docker:
```bash
laia-docker decode "$MODEL_DIR/model.t7" "$TEST_LST";
```
Positional arguments:
- `$MODEL_DIR/model.t7` is the path to the model.
- `$TEST_LST` is a file containing a list of images for testing.

For optional arguments check `laia-decode -h`.

### Example

For a more detailed example, see the Spanish Numbers
[README.md](egs/spanish-numbers/README.md) in `egs/spanish-numbers` folder, or
the IAM [README.md](egs/iam/README.md) in `egs/iam` folder.
