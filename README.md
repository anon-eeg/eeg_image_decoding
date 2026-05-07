# MindAlign: Bridging EEG, Vision, and Language for Zero-Shot Visual Decoding

## Main Folder
The core implementation lives in [eeg_image_decoding/](eeg_image_decoding/):

- `eeg_encoders.py` and `modules.py`: model components and encoder blocks.
- `Pretrain.py`: pretraining entry point.
- `train.py`: training and evaluation entry point.
- `features/`: cached image and text feature files.
- `preprocess/`: preprocessing and feature extraction scripts.

## Datasets
The project uses the Things initiative datasets:

- Things-EEG2 and Things-MEG: https://things-initiative.org/#datasets

Place the preprocessed EEG data in:

```text
./Things-EEG2/Preprocessed_data_250Hz/
```

## Preprocessing
The preprocessing scripts are in [eeg_image_decoding/preprocess/](eeg_image_decoding/preprocess/):

The preprocessing guide is documented in [eeg_image_decoding/preprocess/README_preprocess.md](eeg_image_decoding/preprocess/README_preprocess.md).

## Feature Files
The repository already includes cached features under [eeg_image_decoding/features/](eeg_image_decoding/features/), including:


## Quick Start
1. Install the dependencies listed in [eeg_image_decoding/requirements.txt](eeg_image_decoding/requirements.txt).
2. Download the Things-EEG2 / Things-MEG data and place the preprocessed EEG data under `./Things-EEG2/Preprocessed_data_250Hz/`.
3. Generate or refresh features if needed:
   - `python eeg_image_decoding/preprocess/obtain_text_feature.py`
   - `python eeg_image_decoding/preprocess/pack_text_feature.py`
   - `python eeg_image_decoding/preprocess/extract_eeg_img_features.py`
   - `python eeg_image_decoding/preprocess/extract_meg_img_features.py`
4. Run pretraining with `python eeg_image_decoding/Pretrain.py`.
5. Run downstream training and evaluation with `python eeg_image_decoding/train.py`.
or `python eeg_image_decoding/train.py --no_pretrain` to skip loading pretrained weights.

Pretrained checkpoints are available for download from:

https://huggingface.co/datasets/anon-eeg/EEG-pretrained/tree/main

place them under `./results/mae_eeg_pretrain/checkpoints/`.
