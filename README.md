# Self-Support Few-Shot Semantic Segmentation

[Qi Fan](https://github.com/fanq15), [Wenjie Pei](https://wenjiepei.github.io/), [Yu-Wing Tai](https://scholar.google.com/citations?user=nFhLmFkAAAAJ&hl=zh-CN), [Chi-Keung Tang](https://scholar.google.com/citations?user=EWfpM74AAAAJ&hl=zh-CN)

![](./docs/pipeline.png)



The codebase contains the official code of our paper [Self-Support Few-Shot Semantic Segmentation](), ECCV 2022.


## Data preparation

### Download

**Pretrained model:** [ResNet-50](https://drive.google.com/file/d/11yONyypvBEYZEh9NIOJBGMdiLLAgsMgj/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1mX1yYvkcyOkAVjZZSIf6uMBPlooZCmpk/view?usp=sharing)

**Dataset:** [Pascal images and ids](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
| [Semantic segmentation annotations](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)

### File organization

```
├── ./pretrained
    ├── resnet50.pth
    └── resnet101.pth
    
├── [Your Pascal Path]
    ├── JPEGImages
    ├── SegmentationClass
    └── ImageSets
```


## Run the code

```
CUDA_VISIBLE_DEVICES=0,1 python -W ignore main.py \
  --dataset pascal --data-root [Your Pascal Path] \
  --backbone resnet50 --fold 0 --shot 1
```

You may change the ``backbone`` from ``resnet50`` to ``resnet101``, change the ``fold`` from ``0`` to ``1/2/3``, or change the ``shot`` from ``1`` to ``5`` for other settings.

## Acknowledgement

This codebase is built based on [MLC's baseline code](https://github.com/LiheYoung/MiningFSS).
We thank [MLC](https://arxiv.org/abs/1908.06391) and other FSS works for their great contributions.


## Citation

```bibtex
@inproceedings{fan2022ssp,
  title={Self-Support Few-Shot Semantic Segmentation},
  author={Fan, Qi and Pei, Wenjie and Tai, Yu-Wing and Tang, Chi-Keung},
  journal={ECCV},
  year={2022}
}
```
