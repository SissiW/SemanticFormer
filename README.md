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
![overview](https://github.com/SissiW/SemanticFormer/blob/main/overview.png)

## Results on Indian Pines, Salinas and Houston datasets
More analysis results can be found in the paper.

![Indian_Pines_dataset](https://github.com/SissiW/SemanticFormer/blob/main/indian_pines_table.png)

![Salinas_dataset](https://github.com/SissiW/SemanticFormer/blob/main/salinas_table.png)

![Houston_dataset](https://github.com/SissiW/SemanticFormer/blob/main/houston_table.png)

## Datasets
We perform the abundant experiments on three
publicly popular datasets for hyperspectral image classification. These datasets can be downloaded to click Baidu Drive ([India_Pines](https://pan.baidu.com/s/1ykVdu_E-0Ohdsz-NrMaiUA) (password: jt71), [Salinas](https://pan.baidu.com/s/1Zkb58yI2DkJroqFMfyFfVQ) (password: 30ai), [Houston](https://pan.baidu.com/s/10PIrC1bRREVsJSmqKdXC0Q) (password: 9wl9))

## Installation
python3.8, pytorch1.6, spectral==0.23.1, scipy, numpy, matplotlib

## Config
```
python Main.py
```


## Citation
If you find this project useful, please feel free to leave a star and cite our paper:
```
@article{LIU20241,
  title = {SemanticFormer: Hyperspectral image classification via semantic transformer},
  journal = {Pattern Recognition Letters},
  volume = {179},
  pages = {1-8},
  year = {2024},
  issn = {0167-8655},
  author = {Yan Liu and Xixi Wang and Bo Jiang and Lan Chen and Bin Luo}
}
```

## Acknowledgements
Our code is based on [CEGCN](https://github.com/qichaoliu/CNN_Enhanced_GCN). Thanks for their excellent work!
