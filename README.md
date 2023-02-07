# Dual Multi-scale Mean Teacher Network for Semi-supervised Infection Segmentation in Chest CT Volume for COVID-19

## Introduction

This is an official release of the paper **Dual Multi-scale Mean Teacher Network for Semi-supervised Infection Segmentation in Chest CT Volume for COVID-19**.

<div align="center" border=> <img src=arch.jpg width="700" > </div>

## News

- **[2/6 2023] We have uploaded the training codes.**
- **[10/25 2022] We have created the repo.**

## Code List

- [x] Network
- [x] Pre-processing
- [x] Pre-trained Weights
- [x] Test codes
- [x] Training Codes

For more details or any questions, please feel easy to contact us by email (jiachengw@stu.xmu.edu.cn).

## Usage

### Dataset

Please download the dataset of [COVID-19-P20](https://zenodo.org/record/3757476#.Y1iELaFBxD9) and [MOSMED](https://www.kaggle.com/datasets/andrewmvd/mosmed-covid19-ct-scans).

### Pre-processing

The [file](scripts/prepare_data.py) contains the pre-processing tools for both datasets. Please replace the data path with yours and then run,

```bash
$ python scripts/prepare_data.py
```

### Training

TODO

### Test

You could download the pre-trained weights from [BaiDu Disk](https://pan.baidu.com/s/10U6PBOg4bJ499axR_YtTkA) (g2hd). Please store it locally with correct path, i.e., **logs/mosmed/dmmtnet_multi_mt_0.1**. Then, please run,

```bash
$ python scripts/test.py --gpu 0 --arch dmmtnet --dataset mosmeed
```

## Citation

If you find DM2TNet useful in your research, please consider citing:

```
@article{wang2022dual,
  title={Dual Multi-scale Mean Teacher Network for Semi-supervised Infection Segmentation in Chest CT Volume for COVID-19},
  author={Wang, Liansheng and Wang, Jiacheng and Zhu, Lei and Fu, Huazhu and Li, Ping and Cheng, Gary and Feng, Zhipeng and Li, Shuo and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2211.05548},
  year={2022}
}
```
