# Preprocess Directory

1. EEG preprocessing: `preprocessing.py` calls functions in `preprocessing_utils.py` (`epoching`, `mvnn`, `save_prepr`) to perform channel selection, epoching, baseline correction, resampling, multivariate noise normalization (MVNN), and saving of preprocessed data.
Preprocessed EEG data saved under `Preprocessed_data_250Hz/sub-XX/` (pickle/np files containing `preprocessed_eeg_data`, `ch_names`, `times`).
2. Image feature extraction: `extract_eeg_img_features.py` and `extract_meg_img_features.py` extract image features using CLIP models for training/test sets.
3. Text features extraction: `obtain_text_feature.py` generate short captions for images (via LLMs) and encode them into text feature vectors. Text feature packaging: `pack_text_feature.py` aggregate text features into final matrices suitable for model input.
