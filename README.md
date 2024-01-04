## SCAN

[paper](https://arxiv.org/abs/2311.14925)



<img src=".\fig\Fig1.png" style="zoom:300%;" />

### Abstract

Fourier phase retrieval is essential for high-definition imaging of nanoscale structures across diverse fields, notably coherent diffraction imaging. This study presents the Single impliCit neurAl Network (SCAN), a tool built upon coordinate neural networks meticulously designed for enhanced phase retrieval performance. Bypassing the pitfalls of conventional iterative methods, which frequently face high computational loads and are prone to noise interference, SCAN adeptly connects object coordinates to their amplitude and phase within a unified network in an unsupervised manner. While many existing methods primarily use Fourier magnitude in their loss function, our approach incorporates both the predicted magnitude and phase, enhancing retrieval accuracy. Comprehensive tests validate SCAN’s superiority over traditional and other deep learning models regarding accuracy and noise robustness. We also demonstrate that SCAN excels in the ptychography setting.

### Repository Overview and Implementation

The current version is for simulation and reconstruction with a known probe. You can start by cloning the repo and revising the demos following the comments inside.

- SCAN_CDI_recon: for CDI reconstruction
- - image: simple cases of ground truth 
  - demo: simulation and reconstruction
- SCAN_Pty_recon: for Ptychography reconstruction
- - 2d_probe: Assume a known probe
  - SCANptycho: demo of ptychography simulation and reconstruction by SCAN

### Results

#### CDI Reconstruction

Fig. 1: Best performance of different methods under noise-free condition

<img src=".\fig\comnoisefree.png"/>

#### Ptychography

Fig. 2: Ptychographic reconstruction comparisons between ePIE and SCAN at overlap rates of 30%, 50%, and 70%.

<img src=".\fig\ptychogra.svg"/>

More results could be found in the paper

### Acknowledgement

We make use of the code published in the following repositories:

- SIREN: Implementation of INR (including torchmeta) https://github.com/vsitzmann/siren

### Citation

```
@inproceedings{coordinatebased,
      title={Coordinate-based Neural Network for Fourier Phase Retrieval}, 
      author={Tingyou Li, Zixin Xu, Yong S. Chu, Xiaojing Huang, Jizhou Li},
      booktitle={ICASSP},
      year={2024}
}
```

