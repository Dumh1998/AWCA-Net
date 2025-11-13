## Overview

**[2025-11-07]**  We are delighted to share that our paper, **"High-precision Flood Change Detection with Lightweight SAR Transformer Network and Context-aware Attention for Enriched-diverse and Complex Flooding Scenarios"**, has been successfully accepted by the **ISPRS Journal of Photogrammetry and Remote Sensing (ISPRS 2026)!** 🎉🎉🎉

**[2025-11-10]**  **We have officially released our code and dataset for public access**.🔥🔥🔥


## AWCA-Net
![Figure3](https://github.com/user-attachments/assets/b72b81f6-6960-43fd-a072-164f0cf4c628)


## Dataset
We constructed the first enriched-diverse benchmark dataset for flood change detection (VarFloods).
You can download it from the following links:- **(https://pan.baidu.com/s/1NwpvAfhpxpkRgKcf8x0PRw?pwd=idhi)**

## Data Structure

Please prepare the dataset in the following structure:

```
├── Train
│   ├── A
│   │   └── tif/png (input image of T1)
│   ├── B
│   │   └── tif/png (input image of T2)
│   └── label
│       └── tif/png (binary change mask)
├── Val
│   ├── A
│   │   └── tif/png (input image of T1)
│   ├── B
│   │   └── tif/png (input image of T2)
│   └── label
│       └── tif/png (binary change mask)
├── Test
│   ├── A
│   │   └── tif/png (input image of T1)
│   ├── B
│   │   └── tif/png (input image of T2)
│   └── label
│       └── tif/png (binary change mask)
```


## Citation
**If you find our work useful, please consider citing our paper:**

```bibtex
@article{DU2026507,
    author = {Menghao Du and Zhenfeng Shao and Xiongwu Xiao and Jindou Zhang and Duowang Zhu and Jinyang Wang and Timo Balz and Deren Li},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    title = {High-precision flood change detection with lightweight SAR transformer network and context-aware attention for enriched-diverse and complex flooding scenarios},
    year = {2026},
    volume = {231},
    pages = {507-531},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2025.11.011},
}
