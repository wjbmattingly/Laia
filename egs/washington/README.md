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

|       |    CV1 CER (%)    |    CV2 CER (%)    |    CV3 CER (%)    |    CV4 CER (%)    | Avg.              |
|-------|-------------------|-------------------|-------------------|-------------------|-------------------|
| Valid | 5.35 [4.66, 6.03] | 4.26 [3.53, 4.99] | 4.85 [4.11, 5.59] | 5.35 [4.60, 6.10] | 4.95 [4.23, 5.68] |
| Test  | 5.24 [4.44, 6.04] | 4.22 [3.56, 4.88] | 6.08 [5.21, 6.94] | 4.62 [4.00, 5.25] | 5.04 [4.30, 5.78] |

Bootstrapped confidence intervals at 95% are shown.

## TL;DR

Execute `run.sh' and wait for the results.