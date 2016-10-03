## Step-by-step Training Guide Using Spanish Numbers Dataset

Spanish Numbers Dataset is a small dataset of 485 images containing handwritten sentences of Spanish numbers (298 for training and 187 for testing).

###Example:
![Example](example.png "Example")

To train a new Laia model for the Spanish Numbers dataset just follow these steps. Given that this dataset does not provide validation partition, we will use the test partition as validation.

- Download the Spanish Numbers dataset:
```bash
mkdir -p data/;
wget -P data/ https://www.prhlt.upv.es/corpora/spanish-numbers/Spanish_Number_DB.tgz;
tar -xvzf data/Spanish_Number_DB.tgz -C data/;
```

- Execute `scripts/prepare.sh`. This script assumes that Spanish Numbers dataset is inside `data` folder. This script does the following:
  - Transforms the images from pbm to png.
  - Scales them to 64px height.
  - Creates the auxiliary files necessary for training.

- Execute the `create_model` script to create an "empty" laia model using:
```bash
../../create_model -cnn_type leakyrelu \
  1 64 20 model.t7;
```

- Use the `train_ctc` script to train the model:
```bash
../../train_ctc -batch_size 16 \
  -num_samples_epoch 4000 \
  -adversarial_weight 0.5 \
  -output_progress data/laia.log \
  model.t7 \
  data/lang/chars/symbs.txt \
  data/train.lst \
  data/lang/chars/train.txt \
  data/test.lst \
  data/lang/chars/test.txt;
```

**Note**: We use `-num_samples_epoch 4000` to increase the number of samples per epoch. Given that the amount of asked samples (4000) is higher than the number of available training samples (298), Laia will augment the training samples by using distortions. For more information see [imgdistort library](https://github.com/jpuigcerver/imgdistort).

After 30 epochs the model achieves a CER=~2% in validation.

## Decoding

TBD

## TL;DR

Execute `run.sh`.