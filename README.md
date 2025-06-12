Har2Classifier
===
> Alzheimer's Disease Classifier through Image Harmonization.

Abstract
---
The Har2Classifier is aim to determine the probability of having AD(Alzheimer's Disease) through sMRI (structural Magnetic Resonance Imaing). It train and test in public dataset: ADNI (Alzheimer's Disease Neuroimaging Initiative). The validation AUC have reached 0.961 with F1-score 0.975.

## 1. Introduction & motivation

## 2. Prerequisites
We use pytorch 2.6.0 with cuda 12.8 through our experiments. Please refer to `requirements.txt` for more details
```
pip install -r requirements.txt
```

## Description of code
- `beta_encoder` -> The anatomy encoder originated from HACA3 ([Zuo et al. 2023](https://www.sciencedirect.com/science/article/pii/S0895611123001039)) harmonization model.
- `dataset.py` -> The dataset code for passing `beta`, `preprocessed`, and `harmonized` training data to train the model.
- `preprocess.py` -> the original image should do some basic preprocess for harmonization by HACA3 ([Zuo et al. 2023](https://www.sciencedirect.com/science/article/pii/S0895611123001039)).
- `SFCN.py` -> our backbone structure of our classifier. We focus on the #4 Dimension ([4, 8, 16, 32, 32, 8]).
- `train.py` -> the main code for our experiment. Contains our training process for classifier model.
- `utils.py` -> Recorder for training and code to transform DCM to NII.

## Data
We use a portion of [ADNI](https://adni.loni.usc.edu/data-samples/adni-data/) dataset, which contains 3500 3D sMRI and seperated for training and validaton.

## 3. Training
You can use our original train parameters to train or finetune the classifier
```
python3 train.py \
--epochs 150 \
--lr 0.0005 \
--train_batch_size 16 \
--val_batch_size 16 \
--weight_decay 0.001 \
```

Also, our trained checkpoints are in the checkpoints folder, you can use these weights for further purpose.