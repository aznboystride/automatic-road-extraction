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

To add a model, navigate to `networks/` directory and create `<model_classname>.py`, where `model_classname` is the name of the model.

## Running train.py

Run `python train.py` to see the parameters required.

### Example

`python train.py `
