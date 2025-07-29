### 📖 MASFNet: Multiscale Adaptive Sampling Fusion Network for Object Detection in Adverse Weather

<a href="https://ieeexplore.ieee.org/document/10955257" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%93%9A Paper-IEEE-blue"></a>&ensp;
<a href="https://huggingface.co/spaces/PolarisFTL/MASFNet" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demos-blue"></a>&ensp;
![visitors](https://visitor-badge.laobi.icu/badge?page_id=PolarisFTL.MASFNet) <br />
[Zhenbing Liu](https://www.guet.edu.cn/sai/2023/0601/c5277a107204/page.htm), [Tianle Fang](https://polarisftl.github.io/), [Haoxiang Lu](https://scholar.google.com/citations?user=bvC1s0UAAAAJ&hl=en&oi=sra), [Weidong Zhang](https://scholar.google.com/citations?user=K8y4I0AAAAAJ&hl=en&oi=sra), and [Rushi Lan](https://scholar.google.com/citations?user=8yXQp-0AAAAJ&hl=en&oi=ao)<br />
	Computer Science and Information Security, Guilin University of Electronic Technology

---

![network](https://github.com/PolarisFTL/MASFNet/blob/main/figs/network.png)
_An overview of the proposed MASFNet. MASFNet consists of four parts: 1) FAENet, 2) Backbone, 3) MSFNet, and 4) DH. Among them, the FAENet utilizes the Laplacian pyramid decomposition to split the input image into two different components, a low-frequency component (LF) and a high-frequency component (HF). The feature information of the input image is then adaptively enhanced through modular processing. Then, the output of FAENet is fed into the backbone for feature extraction. The backbone eventually outputs two different scale feature maps into MSFNet for multi-scale fusion. Finally, the DH detects targets and calculates the loss to optimize the model._

#### 😶‍🌫️ Experiments

![](https://github.com/PolarisFTL/MASFNet/blob/main/figs/mist.png)
![](https://github.com/PolarisFTL/MASFNet/blob/main/figs/mid-foggy.png)
![](https://github.com/PolarisFTL/MASFNet/blob/main/figs/high-foggy.png)
![](https://github.com/PolarisFTL/MASFNet/blob/main/figs/low-light.png)
![](https://github.com/PolarisFTL/MASFNet/blob/main/figs/exdark.png)

#### 📢 News

<ul>
<li>[2024-05-09] The paper is submitted.
<li>[2025-04-01] The paper is accepted.
<li>[2025-04-07] The paper is already available for viewing at Early Access.
<li>[2025-04-29] The code has been uploaded.
</ul>

#### 🔧 Requirements and Installation

> - Python 3.6.2
> - PyTorch 1.8.0
> - Cudatoolkit 11.1.1
> - Numpy 1.17.0
> - Opencv-python 4.1.2.30

#### 👽 Installation

```
# Clone the MASFNet
git clone https://github.com/PolarisFTL/MASFNet.git
# Install dependent packages
cd MASFNet
```

#### 🚗 Datasets

| Dataset Name | Total Images | Train Set | Test Set | Google Drive                                                                                  | BaiduYun                                                           |
| ------------ | ------------ | --------- | -------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| RTTS         | 4,322        | 3,889     | 433      | [Link](https://drive.google.com/file/d/1BhU8NnNIQP0mhzB3F-sh7S8xdu-wTO3D/view?usp=drive_link) | [Link (key:1234)](https://pan.baidu.com/s/1TiRYXcDEwnGst5QBZo2twg) |
| ExDark       | 7,363        | 6,626     | 737      | [Link](https://drive.google.com/file/d/1Q1oHGJys7KsO_n0JBtHLOns2as8-0p1M/view?usp=drive_link) | [Link (key:1234)](https://pan.baidu.com/s/1Fi9AUdB1HPBbktt6-8SKDQ) |
| VOC-Rain     | 10,653       | 9,482     | 1,171    | [Link](https://drive.google.com/file/d/1I64t88Oc4yHf8J_U9WghKLYC6pmkM_vz/view?usp=drive_link) | [Link (key:1234)](https://pan.baidu.com/s/1q2iq-cDS0vVm0ZYzLVHAPA) |
| VOC-Snow     | 10,653       | 9,482     | 1,171    | [Link](https://drive.google.com/file/d/1I64t88Oc4yHf8J_U9WghKLYC6pmkM_vz/view?usp=drive_link) | [Link (key:1234)](https://pan.baidu.com/s/1q2iq-cDS0vVm0ZYzLVHAPA) |

#### 🎈 Training and Testing

Run the following commands for training & testing:\
🐻 You need to download the pre-training weights and datasets firstly.

| Name           | Google                                                                                                                | BaiduYun                                                                                   |
| -------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| VOC07+12+COCO  | [yolov4_tiny_weights_voc.pth](https://drive.google.com/file/d/1DGszoaiVAACPGZBHL-8qg8or15of153y/view?usp=drive_link)  | [yolov4_tiny_weights_voc.pth (key:1234)](https://pan.baidu.com/s/1sJW8wYbzIprWvFWQotsLFQ)  |
| COCO-Train2017 | [yolov4_tiny_weights_coco.pth](https://drive.google.com/file/d/1Y2M-nUEL_cHnQeLzgJO_sFBKTegUUAPq/view?usp=drive_link) | [yolov4_tiny_weights_coco.pth (key:1234)](https://pan.baidu.com/s/10Oo5EwQuh2WHwjRt4MBQ6w) |

```python
# train MASFNet for RTTS dataset
1python tools/voc_annotations.py
# VOCdevkit_path='the path of RTTS dataset', data_name='rtts'
modify the config.py
# data_name='rtts'
python train.py
# during training, the result will be saved in the logs-rtts
```

```python
# eval MASFNet for RTTS dataset
python tools/get_map.py
# data_name='rtts,
# vocdevkit_path='the path of RTTS datase'
# model_path = 'los-rtts/best_epoch_weights.pth'
python tools/fps.py
# compute the speed of model
python tools/predict.py
# try to predict the image in adverse weather
```

The steps are the same if training other datasets.

#### 🔥Model Performance

| Method       | Dataset  | Params | FLOPs | FPS | mAP (%) | Google Drive                                                                                  | BaiduYun                                                           |
| ------------ | -------- | ------ | ----- | --- | ------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| MASFNet-Fog  | RTTS     | 6.0M   | 10.7G | 152 | 73.68   | [Link](https://drive.google.com/file/d/13tMYePzn9yRMNpl7j6gg6057gAZTzUeZ/view?usp=drive_link) | [Link (key:1234)](https://pan.baidu.com/s/1yIZHZBx9yjmm4bRgCnUDmg) |
| MASFNet-Dark | ExDark   | 6.0M   | 10.7G | 125 | 63.80   | [Link](https://drive.google.com/file/d/1cr4mUwMeppQaGVf9tZLKmDZarbDuyuRA/view?usp=drive_link) | [Link (key:1234)](https://pan.baidu.com/s/1ZZrQYtvgC91yDglnOMdKMg) |
| MASFNet-Rain | VOC-Rain | 6.0M   | 10.7G | 213 | 60.13   | [Link](https://drive.google.com/file/d/1xerYUIv30YTKdhMcHclgmFDHTJNaoxFm/view?usp=drive_link) | [Link (key:1234)](https://pan.baidu.com/s/1EF2BAAMx04_9RJCqqfOlvQ) |
| MASFNet-Snow | VOC-Snow | 6.0M   | 10.7G | 214 | 59.52   | [Link](https://drive.google.com/file/d/1x3N6OOSOsP4IE8leVx-02W2qQfBjSJcD/view?usp=drive_link) | [Link (key:1234)](https://pan.baidu.com/s/1Ui0GpmqAwfi7A-F6k0ML3Q) |

#### 🔗Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@ARTICLE{10955257,
  author={Liu, Zhenbing and Fang, Tianle and Lu, Haoxiang and Zhang, Weidong and Lan, Rushi},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={MASFNet: Multiscale Adaptive Sampling Fusion Network for Object Detection in Adverse Weather},
  year={2025},
  volume={63},
  number={},
  pages={1-15}
}
```

#### 📨 Contact

If you have any questions, please feel free to reach me out at polarisftl123@gmail.com

#### 🌻 Acknowledgement

This code is based on [YOLOv4-Tiny](https://github.com/bubbliiiing/yolov4-tiny-pytorch.git) & [DENet](https://github.com/NIvykk/DENet.git). Thanks for the awesome work.
