# Rimes dataset

This folder contains experiments using the [Rimes](http://www.a2ialab.com/doku.php?id=rimes_database:start) dataset.
In particular, it contains the scripts to reproduce the results from the paper
"Are multidimensional recurrent layers really necessary for Handwritten Text Recognition?".

## Requirements
- ImageMagick:
  Needed for processing the images.
- [imgtxtenh](https://github.com/mauvilsa/imgtxtenh):
  Needed for processing the images.
- SRILM:
  Needed to build the n-gram language model.
- Kaldi:
  Needed to decode using n-gram language models.
- [Custom Kaldi decoders](https://github.com/jpuigcerver/kaldi-decoders):
  Needed to decode using n-gram language models.
- R _(optional, but higly recommended)_:
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

### Step 3. Training

### Step 4. Decoding

### Step 5. Decoding with word n-gram LM