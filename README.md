# VaB

This a Pytorch implementation of our paper "The Victim and The Beneficiary: Exploiting a Poisoned Model to Train a Clean Model on Poisoned Data", ICCV23, Oral.

## Setup

### Environments

Please install the required packages according to requirement.txt

### Datasets

Download corresponding datasets and extract them to 'dataset'

1. Original CIFAR-10 will be automatically downloaded during training. You can download modified data in [Google Drive](https://drive.google.com/drive/folders/1KzUcys85Y9eYlWXFzxKSYW7UPzhcNbjr?usp=sharing) to implement "CL" and "Dynamic" attacks.
2. For the ImageNet Subset, we prepared four poisoned datasets corresponding to four attacks. Datasets and Codes will be released later.

## Usage

Run the following script to train a clean model on the poisoned data.

```shell
python Train_cifar10.py --trigger_type badnet --trigger_label 0 --trigger_path ./trigger/cifar10/cifar_1.png --posioned_portion 0.1 --model_name ResNet18
```

Please modify the attack settings as you want.

 