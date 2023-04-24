# Virus-Instance-Segmentation-Transformer-Network

This repository contains virus dataset and code (Virus-Instance-Segmentation-Transformer-Network) , provided by Hainan university, Chinese Center for Disease Control and Prevention and Huazhong Agricultural University.

**Dataset**

The virus dataset consists of 318 images with a size of 900 * 820 of three similar viruses (FLUAV, RSV and SARS-CoV-2), with a total number of 1891 virus particles, which were labelled by virologists.

**Environment**

The code is developed and tested under the following configurations.

CUDA>=11.0, Python>=3.7, torch>=1.11.0

The weights of swin transformer can be download in：
https://pan.baidu.com/s/105tIfA1IXsUtcBWCkeZYrQ?pwd=5smq,
code：5smq

**How to use**

Adjust the weight and data path in "config.py" and run the code "train_Yolact_swin.py" and "eval.py". 
