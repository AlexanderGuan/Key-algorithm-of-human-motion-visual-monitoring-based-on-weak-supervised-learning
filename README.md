# 基于弱监督学习的人体运动视觉监控关键算法  

## 一、简介：  
>针对图像语义分割中训练样本数量不足、标注精度不足时，全监督学习算法在语义分割中存在的分割质量较低的问题，本文提出一种将含噪声样本作为训练集的容噪条件生成式对抗网络模型（Noise-tolerance Conditional Generative Adversarial Networks, NCGAN），只需要少量或粗糙的训练样本即可得到较好的分割结果。该方法采用条件生成式对抗网络的基本思想，同时利用U-Net++分割网络将输入数据进行分割，继而利用马尔科夫辨别器对分割结果进行评价，在不断的对抗训练后模型将捕获图像中的高低频信息。本文使用3个公开数据集进行验证，结果表明，本算法在保证检测准确率不低于80%的前提下，将需要人工标注的训练数据减少至有监督算法的30%，且在不同场景下具有较强的通用性。与U-Net模型相比，本文所提出的模型在pixel acc评价标准上提升2.02%,在MIoU标准上提升0.77。  

## 二、步骤：  
### 2.1 环境：  
- Linux  
- NVIDIA GPU + CUDA CUDNN  
- Python3 (3.7+) + Pytorch(0.4+)  

### 2.2 开始：  
#### 安装  
- clone这个仓库:  
>git clone https://github.com/AlexanderGuan/Key-algorithm-of-human-motion-visual-monitoring-based-on-weak-supervised-learning  
>cd Key-algorithm-of-human-motion-visual-monitoring-based-on-weak-supervised-learning  

- 安装依赖库：  
> pip3 install -r requirements.txt  

- 训练:  
>python3 train.py --dataroot [图像的位置(绝对路径)] --name [项目的名称] --direction BtoA  

- 推理：  
>python3 train.py --dataroot [待处理的图像的位置(绝对路径)] --name [项目的名称] --direction BtoA  

- 结果将保存至: ./results/[项目的名称]/latest_test/  

### 使用本项目已经训练好的与训练模型  
- 从百度网盘下载预训练模型:

- 推理:  
python3 test.py --dataroot [待处理的图像的位置(绝对路径)] --direction BtoA --name photo2label_pretrained  


## 引用  
>@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


>@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
