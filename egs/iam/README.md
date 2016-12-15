# Step-by-step Training Guide Using IAM Database

The IAM Handwriting Database contains forms of handwritten English text which
can be used to train and test handwritten text recognizers and to perform
writer identification and verification experiments.

![Example](a01-122-02.jpg)

## Requirements

- Laia
- ImageMagick's convert
- imgtxtenh (https://github.com/mauvilsa/imgtxtenh)

## Data preparation

The first step is to download the data from the IAM servers in Bern and put all images in the
data/imgs directory.

```bash
# Download data
wget -P data --user="$FKI_USER" --password="$FKI_PASSWORD" http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz;

# Put all images into a single directory.
mkdir -p data/imgs;
tar zxf data/lines.tgz -C data/imgs && \
find data/imgs -name "*.png" | xargs -I{} mv {} data/imgs && \
find data/imgs -name "???" | xargs rm -r;
```

We already provide the transcripts of each line in the appropiate format for
the different partitions, so you won't need to do any processing on the
transcripts (see the data/{htr,kws}/lang/{char,word}/{te,tr,va}.txt files).

Just execute the `steps/prepare.sh` script. This script does the following:
  - Obtain the char-level transcripts from the word transcripts.
  - Generate the character symbols table.
  - Enhance images with imgtxtenh and scale to 64px height.
  - Creates the lists of images for train / validation / test.


## Training

Once you have prepared the data for training, it's time to create your model.

```bash
laia-create-model \
  --cnn_type leakyrelu \
  --cnn_kernel_size 3 \
  --cnn_num_features 12 24 48 96 \
  --cnn_maxpool_size 2,2 2,2 1,2 1,2 \
  --cnn_batch_norm false \
  --rnn_num_layers 4 \
  --rnn_num_units 256 \
  --rnn_dropout 0.5 \
  --linear_dropout 0.5 \
  --log_level info \
  1 64 79 model_htr.t7;
```

The model has 4 ConvNet blocks (ConvNet + LeakyReLU + MaxPool), followed by
4 BLSTM layers with 256 units each and a final linear layer with 79 output
units (the number of characters in your training set plus one, used for the
CTC blank symbol).

Finally, you can train the model with `laia-train-ctc`:

```bash
laia-train-ctc \
  --use_distortions true \
  --batch_size "$batch_size" \
  --progress_table_output train_htr.dat \
  --early_stop_epochs 50 \
  --learning_rate 0.00027 \
  --log_also_to_stderr info \
  --log_level info \
  --log_file train_htr.log \
  model_htr.t7 data/htr/lang/char/symbs.txt \
  data/htr/tr.lst data/htr/lang/char/tr.txt  \
  data/htr/va.lst data/htr/lang/char/va.txt;
```

## Decoding

Once the training is finished (you can safely stop it typing CTRL + C), you
can easily decode the test set using `laia-decode`. The output of this
decoding will be a character-level transcript for each test line.

```bash
mkdir -p decode/htr/{char,word};
../../laia-decode \
  --batch_size "$batch_size" \
  --log_level info \
  --symbols_table data/htr/lang/char/symbs.txt \
  model_htr.t7 data/htr/te.lst > decode/htr/char/te.txt;
```

You can obtain the word-level transcripts using the following awk script.

```bash
awk '{
  printf("%s ", $1);
  for (i=2;i<=NF;++i) {
    if ($i == "<space>")
      printf(" ");
    else
      printf("%s", $i);
  }
  printf("\n");
}' decode/htr/char/te.txt > decode/htr/word/te.txt;
```

Finally, compute CER and WER on the test set.

```bash
compute-wer --mode=strict \
  ark:data/htr/lang/char/te.txt ark:decode/htr/char/te.txt |
grep WER | sed -r 's|%WER|%CER|g';

compute-wer --mode=strict \
  ark:data/htr/lang/word/te.txt ark:decode/htr/word/te.txt |
grep WER;
```

After 250 epochs of training, the model achives a CER equal to 6.7 and
WER equal to 22.9, on the test set of the HTR partition.

You can compare these results with other publications on this dataset
in the following table:

| System | CER (%)   | WER (%)    | Comment                                      |
|--------|-----------|------------|----------------------------------------------|
| A2IA [1] | **4.4 ± 0.1** | **10.9 ± 0.4** | 2D-LSTM + CTC, 3-gram word + 10-gram char LM |
| A2IA [1] | 7.3 ± 0.1 | 24.7 ± 0.5 | 2D-LSTM + CTC, No LM                         |
| RWTH [2] | 4.7 ± 0.1 | 12.2 ± 0.4 | 1D-LSTM + HMM + Framewise softmax, 3-gram word + 10-gram char LM |
|--------------------------------------------------------------------------------|
| Laia   | 6.7 ± 0.1 | 22.9 ± 0.5 | ConvNet + 1D-LSTM + CTC, No LM               |


[1] "Deep Neural Networks for Large Vocabulary Handwritten Text Recognition", by
Theodore Bluche.

[2] "Fast and robust training of recurrent neural networks for offline handwriting recognition"
by Patrick Doetsch, Michal Kozielski and Hermann Ney.


## Hyperparameter search

The hyperparameters of the network (number of layers, number of units / layer,
learning rate, etc) were tuned using Spearmint (https://github.com/HIPS/Spearmint),
a Bayesian optimization software.

If you have Spearmint installed, you can run the tunning again with the experiment
available in the spearmint directory.

## TL;DR

Execute `run.sh`.
