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

|       |    CV1 CER (%)    |    CV2 CER (%)    |    CV3 CER (%)    |    CV4 CER (%)    | Avg.              |
|-------|-------------------|-------------------|-------------------|-------------------|-------------------|
| Valid | 5.35 [4.66, 6.03] | 4.26 [3.53, 4.99] | 4.85 [4.11, 5.59] | 5.35 [4.60, 6.10] | 4.95 [4.23, 5.68] |
| Test  | 5.24 [4.44, 6.04] | 4.22 [3.56, 4.88] | 6.08 [5.21, 6.94] | 4.62 [4.00, 5.25] | 5.04 [4.30, 5.78] |

### Word Error Rate

|       |    CV1 WER (%)       |    CV2 WER (%)       |    CV3 WER (%)       |    CV4 WER (%)       | Avg.                 |
|-------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Valid | 22.50 [19.90, 25.11] | 18.77 [15.95, 21.60] | 19.79 [17.29, 22.29] | 22.37 [19.59, 25.14] | 20.86 [18.18, 23.54] |
| Test  | 21.17 [18.38, 23.96] | 17.50 [14.97, 20.03] | 23.78 [21.02, 26.55] | 19.88 [17.46, 22.30] | 20.58 [17.96, 23.21] |

Bootstrapped confidence intervals at 95% are shown.

## TL;DR

Execute `run.sh' and wait for the results.