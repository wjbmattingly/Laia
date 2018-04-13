# Washington - IAM Historical Document Database

The Washington database was created from the George Washington Papers at the
Library of Congress and has been widely used in many publications in the field
of Handwritten Text Recognition and Keyword Spotting.

For additional information, visit the official webpage:
http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/washington-database

**IMPORTANT: You will need to register to the IAM webpage in order to download
the data.** Please, follow the instructions here:
http://www.fki.inf.unibe.ch/DBs/iamHistDB/iLogin/index.php

## Requirements

- Laia
- ImageMagick's convert
- Kaldi's compute-wer-boostci (optional)

## Results

### Character Error Rate

|          | Valid             | Test              |
|----------|-------------------|-------------------|
| CV1      | 5.35 [4.66, 6.03] | 5.24 [4.44, 6.04] |
| CV2      | 4.26 [3.53, 4.99] | 4.22 [3.56, 4.88] |
| CV3      | 4.85 [4.11, 5.59] | 6.08 [5.21, 6.94] |
| CV4      | 5.35 [4.60, 6.10] | 4.62 [4.00, 5.25] |
| **Avg.** | 4.95 [4.23, 5.68] | 5.04 [4.30, 5.78] |

### Word Error Rate

|          | Valid                | Test                 |
|----------|----------------------|----------------------|
| CV1      | 22.50 [19.90, 25.11] | 21.17 [18.38, 23.96] |
| CV2      | 18.77 [15.95, 21.60] | 17.50 [14.97, 20.03] |
| CV3      | 19.79 [17.29, 22.29] | 23.78 [21.02, 26.55] |
| CV4      | 22.37 [19.59, 25.14] | 19.88 [17.46, 22.30] |
| **Avg.** | 20.86 [18.18, 23.54] | 20.58 [17.96, 23.21] |

Bootstrapped confidence intervals at 95% are shown.

## TL;DR

Execute `run.sh' and wait for the results.