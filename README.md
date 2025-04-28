### 📖 MASFNet: Multi-scale Adaptive Sampling Fusion Network for Object Detection in Adverse Weather
<a href="https://ieeexplore.ieee.org/document/10955257" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%93%9A Paper-IEEE-blue"></a>&ensp;
<a href="https://huggingface.co/spaces/PolarisFTL/MASFNet" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demos-blue"></a>&ensp;
![visitors](https://visitor-badge.laobi.icu/badge?page_id=PolarisFTL.MASFNet) <br />

---

![network](https://github.com/user-attachments/assets/1deeb63b-003e-4163-8b77-5f8cfc42194d)
*An overview of the proposed MASFNet. MASFNet consists of four parts: 1) FAENet, 2) Backbone, 3) MSFNet, and 4) DH. Among them, the FAENet utilizes the Laplacian pyramid decomposition to split the input image into two different components, a low-frequency component (LF) and a high-frequency component (HF). The feature information of the input image is then adaptively enhanced through modular processing. Then, the output of FAENet is fed into the backbone for feature extraction. The backbone eventually outputs two different scale feature maps into MSFNet for multi-scale fusion. Finally, the DH detects targets and calculates the loss to optimize the model.*

#### 📢 News
<ul>
<li>[2025-04-07] The paper is already available for viewing at Early Access.
<li> The code will be released soon.
</ul>

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
  pages={1-15},
}
```

#### 📨 Contact
If you have any questions, please feel free to reach me out at polarisftl123@gmail.com
