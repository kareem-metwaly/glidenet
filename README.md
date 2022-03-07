# GlideNet: Global, Local and Intrinsic based Dense Embedding NETwork for Multi-category Attributes Prediction

**Accepted at The IEEE/CVF Computer Vision and Pattern Recognition - CVPR 2022** 

This repo contains the implementation of GlideNet and some useful files to reuse its building blocks.


## Structure of the code 

[models](models) contains different architectures for guidance to reimplement or reuse some building blocks of GlideNet.
Most importantly, [glidenet.py](models/glidenet.py) contains the implemented complete GlideNet structure.
In addition, [informedconv2d.py](models/informedconv2d.py) contains a PyTorch implementation of the novel Informed Convolution.
You can also find some examples of modules that are based on Informed Convolution at [informed_resnet.py](models/informed_resnet.py)


[configs](configs) contains examples of configuration files that can be used to defing the parameters of GlideNet's architecture. 


[dataset](dataset) contains 


**Important Note**:
The code has previously used some proprietary packages during my internship at [Scale AI](https://scale.com/).
Therefore, these packages are missing here, which inhibits the ability to run the code. 
However, in `models/glidenet.py`, you can find the complete implemented structure of GlideNet.
You can use `configs/models/car/glidenet.yaml` for example to play with the configuration of the architecture. 




## Setting up the environment


All required packages are found in [requirements.txt](requirements.txt).
There are some missing proprietary packages, but they are not essential for building GlideNet and its components. 


```shell
conda create -n glidenet python=3.8.5
pip install -r requirements.txt
```



## Citation

Please cite our CVPR 2022 paper if you use any of the building blocks of GlideNet in your work.

```
@InProceedings{metwaly_cvpr_2022_glidenet,
    author    = {Metwaly, Kareem and Kim, Aerin and Branson, Elliot and Monga, Vishal},
    title     = {GlideNet: Global, Local and Intrinsic based Dense Embedding NETwork for Multi-category Attributes Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
}
```
