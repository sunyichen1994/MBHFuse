# MBHFuse: A multi- branch heterogeneous global and local infrared and visible image fusion with differential convolutional amplification features (JOLT 2024)


## Introduction

This is official implementation of ["MBHFuse: A Multi- branch Heterogeneous Global and Local Infrared and Visible Image Fusion with Differential Convolutional Amplification Features"](https://www.sciencedirect.com/science/article/pii/S0030399224011241) with Pytorch.


## Tips

The Trained Model is [here](https://pan.baidu.com/s/1t675r_PeK2qPCddDCgKnEg).


## Recommended Environment
 * CUDA 11.1
 * conda 4.11.0
 * Python 3.7.16
 * PyTorch 1.9.0
 * timm 0.9.7
 * tqdm 4.65.0
 * pandas 1.3.5


## Framework
![Image text](https://github.com/sunyichen1994/MBHFuse/blob/main/The%20network%20architecture/The%20network%20architecture.jpg)


## Structure of GFE
![Image text](https://github.com/sunyichen1994/MBHFuse/blob/main/The%20network%20architecture/GFE.jpg)


## Structure of LFE
![Image text](https://github.com/sunyichen1994/MBHFuse/blob/main/The%20network%20architecture/LFE.jpg)


## Comparison with SOTA methods
### Fusion results on TNO dataset
![Image text](https://github.com/sunyichen1994/MBHFuse/blob/main/Qualitative%20evaluation/TNO.jpg)

### Fusion results on RoadScene dataset
![Image text](https://github.com/sunyichen1994/MBHFuse/blob/main/Qualitative%20evaluation/RoadScene.jpg)

### Fusion results on MSRS dataset
![Image text](https://github.com/sunyichen1994/MBHFuse/blob/main/Qualitative%20evaluation/MSRS.jpg)

### Fusion results on M3FD dataset
![Image text](https://github.com/sunyichen1994/MBHFuse/blob/main/Qualitative%20evaluation/M3FD.jpg)

### Fusion results on LLVIP dataset
![Image text](https://github.com/sunyichen1994/MBHFuse/blob/main/Qualitative%20evaluation/LLVIP.jpg)

### Quantitative fusion results
![Image text](https://github.com/sunyichen1994/MBHFuse/blob/main/Qualitative%20evaluation/Quantitative%20fusion%20results.jpg
)


## Citation

If you find this repository useful, please consider citing the following paper:

```
@article{sun2024JOLT,
  title={MBHFuse: A multi- branch heterogeneous global and local infrared and visible image fusion with differential convolutional amplification features},
  author={Sun, Yichen and Dong, Mingli and Yu, Mingxin and Zhu, Lianqing},
  journal={Optics and Laser Technology},
  year={2025},
  volume={181},
  part={A},
  pages={111666},
  doi={10.1016/j.optlastec.2024.111666}}
  }  
```


If you have any questions, feel free to contact me (sunyichen@emails.bjut.edu.cn)


## Acknowledgements

Parts of this code repository is based on the following works:

 * https://github.com/linklist2/PIAFusion_pytorch
 * https://github.com/Zhaozixiang1228/MMIF-CDDFuse
