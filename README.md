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

- CUDA 7.5 + cuDNN
- Torch7
- cuTorch
- Torch cuDNN [Install it from https://github.com/soumith/cudnn.torch]
- Warp CTC [https://github.com/baidu-research/warp-ctc]
