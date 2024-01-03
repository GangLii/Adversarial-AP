# Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness
This is the official code for the NeurIPS 2023 spotlight paper "Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness" by Gang Li, Wei Tong, and Tianbao Yang.

## Training
Below is an example of running the method "AdAP_LPN" on CIFAR10 dataset.
```
CUDA_VISIBLE_DEVICES=0  python ./main_cifar10_resnet18.py --method=AdAP_LPN --gamma1=0.1 --gamma2=0.9 --Lambda=0.8
```
