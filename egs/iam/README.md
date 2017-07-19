# Step-by-step Training Guide Using IAM Database

The IAM Handwriting Database contains forms of handwritten English text which
can be used to train and test handwritten text recognizers and to perform
writer identification and verification experiments.

![Example](a01-122-02.jpg)

This folder contains the scripts to reproduce the results from the paper
"Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?", by Joan Puigcerver.

## Requirements
- [ImageMagick](https://www.imagemagick.org/):
  Needed for processing the images.
- [imgtxtenh](https://github.com/mauvilsa/imgtxtenh):
  Needed for processing the images.
- [SRILM](http://www.speech.sri.com/projects/srilm/):
  Needed to build the n-gram language model.
- [Kaldi](https://github.com/kaldi-asr/kaldi):
  Needed to decode using n-gram language models.
- [Custom Kaldi decoders](https://github.com/jpuigcerver/kaldi-decoders):
  Needed to decode using n-gram language models.
- [R](https://www.r-project.org/) _(optional, but higly recommended)_:
  Needed to compute confidence intervals.

## Pipeline

This section explains some details of the `run.sh` script. If you are too
lazy to read this (or you are in a hurry), just type:

```bash
./steps/run.sh
```

### Step 1. Download data.

```bash
./steps/download.sh --iam_user "$IAM_USER" --iam_pass "$IAM_PASS";
```

### Step 2. Prepare images.

```bash
./steps/prepare_images.sh;
```

### Step 3. Prepare IAM text data.

```bash
./steps/prepare_iam_text.sh --partition aachen;
```

### Step 4. Train the neural network.

```bash
./steps/train_lstm1d.sh --partition aachen --model_name "lstm1d_h128";
```

### Step 5. Decode using only the neural network.

```bash
./steps/decode_net.sh "train/aachen/lstm1d_h128.t7";
```

### Step 6. Output raw network label pseudo log-likelihoods.

```bash
./steps/output_net.sh --partition aachen "train/aachen/lstm1d_h128.t7";
```

### Step 7. Decode using external n-gram language model.

```bash
./steps/decode_lm.sh --partition aachen \
    decode/lkh/forms/aachen/{va,te}_lstm1d_h128.scp;
```

## Any problem?

If you have any issue reproducing the results of the paper, please contact the
author at joapuipe@prhlt.upv.es.
