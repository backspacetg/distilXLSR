Models and codes for INTERSPEECH 2023 paper *DistilXLSR: A Light Weight Cross-Lingual Speech Representation Model*


### Pre-trained models:

| Models | Link |
|--------|------|
|DistilXLSR128|[google drive](https://drive.google.com/file/d/1eJ3zamDYFb6kDuRpR9hHYYR47yyEIXmj/view?usp=sharing)| 
|DistilXLSR53|[google drive](https://drive.google.com/file/d/1AN-PGQ6GxNueuklpezYSry6nuAJwrym1/view?usp=sharing)|

### Usage

Our code are based on the [fairseq](https://github.com/facebookresearch/fairseq) toolkits. You can copy `codes` folder into `fairseq/fairseq/models` and rename it as distilxlsr (or other names) and use DistilXLSR as other fairseq models such as Wav2vec 2.0 or HuBERT. 

#### Data Preparation

#### Training

#### Decoding

### Some additional experiment results: 

We trained conformer-based E2E models and DNN-HMM (rather than GMM-HMM) models on 5 Common Voice languages, with the same no more than 10 hours subset. 

| Models   | el   | nl    | eu    | ia    | pl    | average |
|----------|------|-------|-------|-------|-------|---------|
| XLSR53   | 10.7 | 12.4  | 29.5  | 27.1  | 25.5  | 21.04   |
| Proposed | 14.2 | 14.9  | 33.8  | 34.4  | 28.8  | **25.22** |
| DNN-HMM  | 43.4 | 10.26 | 25.77 | 71.71 | 21.48 | 34.524  |
| E2E      | 65.6 | 51.9  | 21.1  | 77.9  | 30.5  | 49.4    |
