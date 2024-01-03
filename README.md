# Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness
This is the official code for the NeurIPS 2023 spotlight paper "Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness" by Gang Li, Wei Tong, and Tianbao Yang.

## Training
Below is an example of running the method "AdAP_LPN" on CIFAR10 dataset.
```
CUDA_VISIBLE_DEVICES=0  python ./main_cifar10_resnet18.py \
--method=AdAP_LPN \
--gamma1=0.1 --gamma2=0.9 --Lambda=0.8
```
Below is an example of running the method "AdAP_LN" on CIFAR100 dataset.
```
CUDA_VISIBLE_DEVICES=0  python ./main_cifar100_resnet18.py \
--method=AdAP_LN \
--gamma1=0.1 --gamma2=0.9 --Lambda=0.8
```

## List of methods
This is a list of different optimization methods compared in the paper. We summarize them here for reference.
- 'AdAP_LPN' : AdAP_LPN
- 'AdAP_LN' : AdAP_LN
- 'AdAP_LZ' : AdAP_LZ
- 'AdAP_PZ' : AdAP_PZ
- 'AdAP_MM' : AdAP_MM
- 'TRADES' : TRADES
- 'MART' : MART
- 'PGD' : PGD
- 'AP' : AP Max.
- 'CE' : CE Min.
