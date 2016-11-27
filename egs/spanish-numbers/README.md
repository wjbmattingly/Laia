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

- Execute the `laia-create-model` script to create an "empty" laia model using:
```bash
../../laia-create-model \
  --cnn_batch_norm true \
  --cnn_type leakyrelu \
  -- 1 64 20 model.t7;
```

- Use the `laia-train-ctc` script to train the model:
```bash
../../laia-train-ctc \
  --adversarial_weight 0.5 \
  --batch_size "$batch_size" \
  --log_also_to_stderr info \
  --log_level info \
  --log_file laia.log \
  --progress_table_output laia.dat \
  --use_distortions true \
  --early_stop_epochs 100 \
  --learning_rate 0.0005 \
  model.t7 data/lang/chars/symbs.txt \
  data/train.lst data/lang/chars/train.txt \
  data/test.lst data/lang/chars/test.txt;
```

After 367 epochs the model achieves a CER=~1.67% in validation, with a 95% confidence interval in [1.12%, 2.17%].

## Decoding

TBD

## TL;DR

Execute `run.sh`.
