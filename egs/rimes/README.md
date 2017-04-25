# Rimes dataset

This folder contains experiments using the [Rimes](http://www.a2ialab.com/doku.php?id=rimes_database:start) dataset.
In particular, it contains the scripts to reproduce the results from the paper
"Are multidimensional recurrent layers really necessary for Handwritten Text Recognition?".

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

### Step 1. Download data

You will first need to register to A2IA and download the Rimes dataset, used in
the ICDAR 2011 Competition. Please, visit:
http://www.a2ialab.com/doku.php?id=rimes_database:data:icdar2011:line:icdar2011competitionline

Place all the following files to `data/a2ia`:

- http://www.a2ialab.com/lib/exe/fetch.php?media=rimes_database:data:icdar2011:line:training_2011_gray.tar
- http://www.a2ialab.com/lib/exe/fetch.php?media=rimes_database:data:icdar2011:line:training_2011.xml
- http://www.a2ialab.com/lib/exe/fetch.php?media=rimes_database:data:icdar2011:line:eval_2011_gray.tar
- http://www.a2ialab.com/lib/exe/fetch.php?media=rimes_database:data:icdar2011:line:eval_2011_annotated.xml

The original training data contains some errors in the training ground-truth
that would break some of our scripts. We have included a patch of the training
XML that will fix these errors. Just execute the following command to apply
the patch (the command also checks that you have placed all the required data
to the correct location):

```bash
./steps/download.sh
```

### Step 2. Prepare data

Once the original data is downloaded and fixed, you'll need to process the
images and text data.

The processing on the text data is quite straightforward: We just extract the
text from the XML files and put them into plain text files containing the
character-level, word-level and tokenized transcripts.

The tokenization is done with a custom Python script, similar to NLTK's
Treebank tokenizer. The main difference with the NLTK's tokenizer is that
we split all characters from words composed by all-numbers and/or all-capitals.
In addition, when we tokenize a word, we keep track of the original form in
order to create a lexicon that keeps track of tokens that can/cannot be preceded
by a whitespace symbol (see `steps/rimes_tokenize.py` for more details).

The text line images are extracted using the bounding boxes present in the XML
files. We then enhance them using imgtxtenh, deskew them using ImageMagick's
convert and remove top and bottom white borders, leaving a fixed left and right
borders of 20 pixels. The line images are alse scaled at a fixed height of 128
pixels.

```bash
./steps/prepare.sh
```

### Step 3. Training

The training script replicates the final model described in the paper,
broadly speaking:

- 5 Convolutional blocks with 3x3 convolutions, LeakyReLU activations
  and batch normalization. The first 3 blocks include a MaxPooling layer,
  and the last 3 blocks use dropout with probability 0.2. The number of
  features in each block is 16, 32, 48, 64, and 80, respectively.
- 5 bidirectional LSTM recurrent layers with 256 hidden units and dropout with
  probability 0.5.
- A final linear layer with 99 output units (98 characters + CTC blank symbol).
- Training stops after 80 epochs without any improvement on the validation
  CER.

```bash
./steps/train_lstm1d.sh
```

__IMPORTANT:__ Be aware that this script may take a considerable amount of time
to run (46h on a NVIDIA Titan X) and GPU memory (10.3GB). If this is not
feasible to you, to reduce the batch size (the default is 16) by passing
`--batch_size $your_batch_size` to the script, and/or reduce the early stop
epochs with `--early_stop_epochs $your_max_stop_epochs` (the default is 80).


### Step 4. Simple decoding

### Step 5. Decoding with word n-gram LM