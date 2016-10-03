# Laia: A deep learning toolkit for HTR

Laia is a deep learning toolkit to transcribe handwritten text images.

If you find this toolkit useful in your research, please cite:

```
@inproceedings{laia,
  title={},
  author={},
  booktitle={},
  year={}
}
```

## Installation

Laia is implemented in [Torch](http://torch.ch/), and depends on the following:

- [CUDA >=7.5](https://developer.nvidia.com/cuda-downloads)
- [cuDNN >= 5.1](https://developer.nvidia.com/cudnn)
- [cuDNN.torch](https://github.com/soumith/cudnn.torch)
- [Baidu's CTC](https://github.com/baidu-research/warp-ctc)
- [imgdistort](https://github.com/jpuigcerver/imgdistort)

Note that currently we only support GPU. You need to use NVIDIA's cuDNN library. Register first for the CUDA Developer Program (it's free) and download the library from [NVIDIA's website](https://developer.nvidia.com/cudnn); you also have to  install the [cuDNN bindings for Torch](https://github.com/soumith/cudnn.torch).

## Usage

### Training a Laia model using CTC:

Create an "empty" model using:
```bash
create_model "$CHANNELS" "$INPUT_HEIGHT" "$((NUM_SYMB+1))" "$MODEL_DIR/model.t7"
```
Positional arguments:
- `$CHANNELS`: number of channels of the input images.
- `$INPUT_HEIGHT`: height of the input images. Note: **ALL** image must have the same height.
- `$((NUM_SYMB+1))`: number of output symbols. Note: Include **ONE** additional element for the CTC blank symbol.
- `$MODEL_DIR/model.t7`: path to the output model.

For optional arguments check `create_model.lua -h`.


Train the model using:
```bash
train_ctc "$MODEL_DIR/model.t7" \
          "$SYMBOLS_TABLE" \
          "$TRAIN_LST" "$TRAIN_GT" "$VALID_LST" "$VALID_GT"
```
Positional arguments:
- `$MODEL_DIR/model.t7` is the path to the input model or checkpoint for training.
- `$SYMBOLS_TABLE` is the list of training symbols and their id.
- `$TRAIN_LST` is a file containing a list of images for training.
- `$TRAIN_GT` is a file containing the list of training transcripts.
- `$VALID_LST` is a file containing a list of images for validation.
- `$VALID_GT` is a file containing the list of validation transcripts.

For optional arguments check `train.lua -h`.

### Transcribing
```bash 
decode "$MODEL_DIR/model.t7" "$TEST_LST"
```
Positional arguments:
- `$MODEL_DIR/model.t7` is the path to the model.
- `$TEST_LST` is a file containing a list of images for testing.

For optional arguments check `decode -h`.
-

### Example

For an more detailed example, see [README.md](egs/spanish-numbers/README.md) in `egs/spanish-numbers` folder.
