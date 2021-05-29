# SIIM_Pneumothorax_segmentation
Unet based model for the SIIM Pneumothorax dataset, (based on the original Unet Architecture). This project is based on the challenge posted on Kaggle.

Statement: Segment the infected chest-Xray images in the dataset. The dataset can be downloaded from https://www.kaggle.com/seesee/siim-train-test.
The data is in DICOM format, with masks in RLE format.

LOSS : Binary Cross-Entropy, pixelwise classification.

The Original U-Net architecture is used.


Paper : https://arxiv.org/pdf/1505.04597


TODO : deal with imbalanced dataset
Done - Implemented focal loss to mitigate the data imbalance.


