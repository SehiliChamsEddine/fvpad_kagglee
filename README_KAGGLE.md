# Running the feature extraction on Kaggle

1. kaggle.com -> Create -> New Notebook -> File -> Import Notebook -> `kaggle_feature_extraction.ipynb`
   (it is inside this zip).
2. Right panel: Add Input -> Upload -> New Dataset -> upload `fvpad_kaggle.zip` (any title).
3. Settings: Accelerator = GPU (T4 x1 or P100), Internet = On.
4. Run All. It downloads the three course zips (~3 GB) and the CNN weights, then extracts
   embeddings for 33,442 images with the 5 CNNs (roughly 15-25 min on a T4).
5. Output tab: download `features_IDIAP.zip`, `features_SCUT.zip`, `features_PLUS.zip`
   and `extraction_log.txt`. Each zip is < 400 MB.
6. Put the four files in your Downloads folder (or attach them in the chat) and tell Claude.

The notebook uses the shipped manifests in outputs/manifests and outputs/folds unchanged,
so the image ids match the main project exactly.
