# Learnable Gammatone Filterbank (LGTFB) and Equal-loudness Normalization (EN) 

This repo contains code for the paper [CNN-based Learnable Gammatone Filterbank and Equal-loudness Normalization for Environmental Sound Classification](https://ieeexplore.ieee.org/document/9005226).

## Setup

### Hardware/Platform Specs

The models were trained and using TITAN X (Pascal) gpus. 
All models were developed and tested in
```
Python 2.7.6
TensorFlow 1.11.0
```

### Download Dataset

[ESC-50 dataset](https://github.com/karolpiczak/ESC-50) is necessary to be downloaded and placed in ./Data/

```
cd Data
wget https://github.com/karoldvl/ESC-50/archive/master.zip
unzip master.zip
rm master.zip
cd ..
```

### Data Augmentation
```
python data_augm.py
rm Data/ESC-50-master/audio_da/*.jams
```

### Data preparation
```
python data_prep.py
```

## Training
```
python run.py
```

### Bibtex
```
to be appeared
```
