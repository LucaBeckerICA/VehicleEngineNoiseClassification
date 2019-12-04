# Audio Feature Extraction for Vehicle Engine Noise Classification (VENC)

Source Code of the paper: AUDIO FEATURE EXTRACTION FOR VEHICLE ENGINE NOISE CLASSIFICATION (Becker et al.: Conference Paper, ICASSP 2020)

The folders contain the utilized acoustic dataset (/dataset), the low-level and modulation features (/features),
the trained classificaiton models (/models) and the source code for the feature computation and classification (/src).

In order to compute the features from the dataset, it is recommended to use the Per-channel Energy Normalization features
implementation from (src). The features shall be stored as a cell in a .mat file with the dimensionality {1, n}. For the
modulation approach, the MATLAB script 'modMFCCMaps.m' computes the Mod-PCEN features and stores them in a .mat file.

For the Siamese Training/Inference, it is only necessary to run either 'start_script_siamese.py' (Modulation Features)
or 'start_script_siamese_raw.py' (Non-Modulation Features).
