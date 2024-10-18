# [Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness](https://proceedings.neurips.cc/paper_files/paper/2023/file/31f04c174a6af322e9417b7a9a91097a-Paper-Conference.pdf)
This is the official code for the NeurIPS 2023 spotlight paper "Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness" by Gang Li, Wei Tong, and Tianbao Yang.

## Environment
```
python==3.8
torch==1.9.1
torchvision==0.10.1
scikit-learn==1.3.0
opencv-python==4.6.0
```
## Training
git clone
```
git clone https://github.com/GangLii/Adversarial-AP.git
```
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

## Contact
If you have any questions, please open a new issue or contact Gang Li via <gang-li@tamu.edu>. If you find this repo helpful, please cite the following paper:
```
@inproceedings{li2023maximization,
  title={Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness},
  author={Li, Gang and Tong, Wei and Yang, Tianbao},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
