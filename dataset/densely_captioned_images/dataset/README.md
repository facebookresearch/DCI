# densely_captioned_images.dataset

This directory contains files that are used in the DCI release.

- **`dense_image.py`**: Defines the overarching data structure for Densely Captioned Images, which is used downstream by all other classes
- **`impl.py`**: Contains a torch `Dataset` wrapper around the CLIP-ready version of DCI
- **`loss.py`**: Defines the loss functions used by DCI train, which are relevant for computing the highest scored captions during test.
- **`spacy_negs.py`**: Utility functions for generating spacy-based negatives for captions in the DCI dataset. 
- **`utils.py`**: simple utilities that are relevant for the project.
- **`config.py`**: Config loader for options related to using DCI globally.
- **`scripts/download.py`**: Download script for the dataset and our released LoRA weights.
- **`scripts/run_clip_dense_cap_eval.py`**: Usage example for running DCI test, defaults to running on the CLIP baseline.
