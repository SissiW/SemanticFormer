# 《SemanticFormer: Hyperspectral image classification via semantic transformer》 PRL 2024

[paper](https://www.sciencedirect.com/science/article/pii/S0167865523003793) &nbsp;&nbsp;

## Abstract 
Hyperspectral image (HSI) classification is an active research problem in computer vision and multimedia field.
Contrary to traditional image data, HSIs contain rich spectral, spatial and semantic information. Thus, how
to extract the discriminative features for HSIs by integrating spectral, spatial and semantic cues together
is the core issue to address HSI classification task. Existing works mainly focus on exploring spectral and
spatial information which usually fail to fully explore the rich semantic information in HSIs. To address this
issue, in this paper, we first propose a novel semantic Transformer scheme, named SemanticFormer, which
aims to learn discriminative visual representations for semantics by exploiting the interaction among different
semantic tokens. Using the proposed SemanticFormer, we then propose a novel heterogeneous network that
contains both spectral–spatial convolution network branch and SemanticFormer branch to extract spectral–
spatial and semantic features simultaneously for HSIs. Experiments on two widely used datasets demonstrate
the effectiveness of our SemanticFormer and HSI classification network method.

## Architecture
![overview](https://github.com/SissiW/QSFormer/blob/main/overview.png)

## Results on MiniImageNet and TieredImageNet
More experimental results can be found in the paper.
![results](https://github.com/SissiW/QSFormer/blob/main/mini_tiered_result.png?raw=true)

## Datasets
We perform the abundant experiments on four
publicly popular datasets for few-shot classification task,
such as miniImageNet, tieredImageNet, Fewshot-CIFAR100 and Caltech-UCSD Birds-200-2011.
These datasets can be downloaded to click Baidu Drive ([miniImageNet](https://pan.baidu.com/s/1yTn78HgbkrRh_3EClax5FA) (password: rqcs), [tieredImageNet](https://pan.baidu.com/s/1Z9ZsYkwAY11Z_Glzu4tChQ) (password: k5z6), [FC100](https://pan.baidu.com/s/1atEdnikzs8zfKXuO4xr1rQ) (password: 3cib), [CUB](https://pan.baidu.com/s/1defYYyFQL5ZV1Dzug5paHQ) (password: qkpc))

## Installation
python3.7+, pytorch>=1.7, qpth, CVXPY, OpenCV-python, tensorboard

## Download Pre-trained Models
[Baidu Drive](https://pan.baidu.com/s/1UWnpjNaaCTSUB2sOtJqZng)
提取码：yd8w

## Config
```
sh train_meta_QSFormer.sh
```


## Citation
If you find this project useful, please feel free to leave a star and cite our paper:
```
@article{wang2023few,
  title={Few-Shot Learning Meets Transformer: Unified Query-Support Transformers for Few-Shot Classification},
  author={Wang, Xixi and Wang, Xiao and Jiang, Bo and Luo, Bin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgements
This project is built upon [DeepEMD](https://github.com/icoz69/DeepEMD). We also reference some code from [DETR](https://github.com/facebookresearch/detr). Thanks to the contributors of these great codebases.
