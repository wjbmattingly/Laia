# Laia: A deep learning toolkit for HTR based on Torch

## Create a model

```bash
th create_model.lua $INPUT_CHANNELS $INPUT_HEIGHT $OUTPUT_SIZE $OUTPUT_MODEL
```

## Train

```bash
th train.lua $INIT_MODEL $TRAIN_H5 $VALID_H5
```

## Decode

```bash
th decode.lua $MODEL $DATA_H5
```

## Requirements

- [CUDA >=7.5](https://developer.nvidia.com/cuda-downloads)
- [cuDNN >= 5.1](https://developer.nvidia.com/cudnn)
- [Torch7](http://torch.ch)
- [cuDNN.torch](https://github.com/soumith/cudnn.torch)
- [Baidu's CTC](https://github.com/baidu-research/warp-ctc)
- [imgdistort](https://github.com/jpuigcerver/imgdistort)