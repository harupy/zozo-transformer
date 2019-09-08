## ZOZO transformaer

## Motivation

Mercari proposed an interesting method to transofrom an image query feature.

https://sigir-ecom.github.io/ecom19Papers/paper3.pdf

## Dataset

Sample images are collected from [ZOZOTOWN](https://zozo.jp/).

## Feature Extractor

ResNet50 trained on Imagenet (The paper says they used MobileNetV2 for edge computing.)

## Result

I'm still trying to figure out which combination of features works the best.

![result.png](./output/result.png)
