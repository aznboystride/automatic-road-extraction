# Automatic Road Extraction Deep Learning

On going research project that aims to automatically segment roads from high resolution satellite imagery.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing and training purposes.

### Installing

```
git clone https://github.com/aznboystride/automatic-road-extraction

mkdir weights optimizers # training and testing program will write to these folders.

cp -r /home/ml_csulb_gmail_com/automatic-road-extraction/valid /home/ml_csulb_gmail_com/automatic-road-extra
ction/train .
```

## Adding a custom model

To add a model, navigate to `networks/` directory and create `<model_classname>.py`, where `model_classname` is the class of the model.

## Running train.py

Run `python train.py` to see the parameters required.

```
usage: train.py [-h] -lr LR -b BATCH -it ITERATIONS -dv DEVICES [-lw LWEIGHTS]
                [-au AUGMENT] [-ls LOSS] [-e EPOCH]
                model
```

### Example

`python train.py -ls BCESSIM -lr 1e-4 -b 16 -it 200 -dv 0,1,2,3 -au DilatedDense`
