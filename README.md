Models and codes for INTERSPEECH 2023 paper *DistilXLSR: A Light Weight Cross-Lingual Speech Representation Model*


### Pre-trained models:

| Models | Link |
|--------|------|
|DistilXLSR128|[google drive](https://drive.google.com/file/d/1eJ3zamDYFb6kDuRpR9hHYYR47yyEIXmj/view?usp=sharing)| 
|DistilXLSR53|[google drive](https://drive.google.com/file/d/1AN-PGQ6GxNueuklpezYSry6nuAJwrym1/view?usp=sharing)|
|Language Models|[google drive](https://drive.google.com/file/d/16wbbz-8B1Ncd_YPR2qM2I0Pdu2gFrl3g/view?usp=sharing)|

### Usage

Our code are based on the [fairseq](https://github.com/facebookresearch/fairseq) toolkits. You can copy `codes` folder into `fairseq/fairseq/models` and rename it as distilxlsr (or other names) and use DistilXLSR as other fairseq models such as Wav2vec 2.0 or HuBERT. Please refer to the[Wav2vec2 guideline](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) for further information about the usage of Wav2vec 2.0. 

#### Data Preparation

##### Formatting Datasets

The selected 10-hour subsets of 5 languages in the Common Voice dataset (Version 5.1) are provided in the `data` folder.  You can select the mp3 samples according to the `tsv` files, convert them to wav format, and save them in paths like `$output_path/$language/wav/$file_name` such as `/mnt/data/el/wav/common_voice_el_20583960.wav`. Please remember to change the first line of the `tsv` files which provides the root folder of all the samples. 

##### Training

Run `run_cv.sh` to fine-tune the DisilXLSR models on 5 languages. Training will take about 5 hours on a RTX-3090 GPU. 

##### Decoding

You can download the language models from the link in the table above. Unzip the models. 
Run stage 1 in `decode.sh` to decode the models. The [Sclite toolkit](https://github.com/usnistgov/SCTK) is used for scoring, so we should format the transcription files for Sclite, and stage 2 in `decode.sh` does this. After scoring, the results are printed on the screen. 

### Some additional experiment results: 

We trained conformer-based E2E models and DNN-HMM (rather than GMM-HMM) models on 5 Common Voice languages, with the same no more than 10 hours subset. 

| Models   | el   | nl    | eu    | ia    | pl    | average |
|----------|------|-------|-------|-------|-------|---------|
| XLSR53   | 10.7 | 12.4  | 29.5  | 27.1  | 25.5  | 21.04   |
| Proposed | 14.2 | 14.9  | 33.8  | 34.4  | 28.8  | **25.22** |
| DNN-HMM  | 43.4 | 10.26 | 25.77 | 71.71 | 21.48 | 34.524  |
| E2E      | 65.6 | 51.9  | 21.1  | 77.9  | 30.5  | 49.4    |
