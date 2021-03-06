# 基于弱监督学习的人体运动视觉监控关键算法  

## 一、简介  
>针对图像语义分割中训练样本数量不足、标注精度不足时，全监督学习算法在语义分割中存在的分割质量较低的问题，本文提出一种将含噪声样本作为训练集的容噪条件生成式对抗网络模型（Noise-tolerance Conditional Generative Adversarial Networks, NCGAN），只需要少量或粗糙的训练样本即可得到较好的分割结果。该方法采用条件生成式对抗网络的基本思想，同时利用U-Net++分割网络将输入数据进行分割，继而利用马尔科夫辨别器对分割结果进行评价，在不断的对抗训练后模型将捕获图像中的高低频信息。本文使用3个公开数据集进行验证，结果表明，本算法在保证检测准确率不低于80%的前提下，将需要人工标注的训练数据减少至有监督算法的30%，且在不同场景下具有较强的通用性。与U-Net模型相比，本文所提出的模型在pixel acc评价标准上提升2.02%,在MIoU标准上提升0.77。  

## 二、实验与结果  
- **CityScapes数据集**  
分割的类别为人体、汽车、房屋、植被、道路（分别为CityScapes Class 1,Class 2）。设置学习率为0.001，批大小为8，按照6:1:1的比例划分训练集、验证集和测试集。迭代7000次，每次迭代时间越200秒，训练时间52小时。训练完成后，使用网络进行测试得到结果如图4-1所示，其正确地将每个属类进行语义分割。  
![cityscapes](https://github.com/AlexanderGuan/Key-algorithm-of-human-motion-visual-monitoring-based-on-weak-supervised-learning/blob/main/cityscapes.JPG)  
经测试，统计结果在CityScapes数据集上的多目标语义分割平均像素精度为81.34%，基本与全监督语义分割算法持平，符合实际工程需求。  

- **GoogleMap数据集**  
本节在谷歌地图数据集上进行了实验，该数据集共包含建筑与背景两个类。原始图像为16张高分辨率的卫星图像，本文将图像裁剪为256x256分辨率的共3800张图像。设置学习率为0.001，批大小为8，按照6:1:1的比例划分训练集、验证集和测试集。Epoch为200次，共迭代95000次，平均每个epoch需要200秒，共用时11.5个小时。  
![google,ap](https://github.com/AlexanderGuan/Key-algorithm-of-human-motion-visual-monitoring-based-on-weak-supervised-learning/blob/main/google%20map.JPG)  
经测试，统计结果在谷歌地图数据集上的多目标语义分割平均像素精度为83.25%。
 
## 三、步骤  
### 3.1 环境  
- Linux  
- NVIDIA GPU + CUDA CUDNN  
- Python3 (3.7+) + Pytorch(0.4+)  

### 3.2 开始  
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

### 3.3 使用本项目已经训练好的与训练模型  
- 从百度网盘下载预训练模型:

- 推理:  
python3 test.py --dataroot [待处理的图像的位置(绝对路径)] --direction BtoA --name photo2label_pretrained  

## 四、模型设计  
### 4.1 问题描述  
本项目以实际生产中的予以分割需求为研究导向，研究语义分割这一问题。目前，语义分割多为监督学习，且限定于某个特定场景，如街道、室内等，该方法设计模型时需要设计与数据集相适宜的模型，导致泛化性较低。因此本项目着重研究一种泛化性强，可以减少人工标记数量的人体语义分割方法。  

### 4.2 条件生成式对抗网络  
#### 4.2.1 生成式对抗网络
- **基本思想**  
文献[3]提出条件生成式对抗网络(Generative Adversarial Network, GAN)，其主要包括两个网络，一个是生成器Generater，另一个是辨别器Discriminator，一个是生成器Generater将随机输入的噪声映射为伪图像，另一个是辨别器Discriminator判断输入数据是真实图像还是一个是Generater产生的伪图像，图为GAN网络的结构。  
![GAN网络结构](https://github.com/AlexanderGuan/Key-algorithm-of-human-motion-visual-monitoring-based-on-weak-supervised-learning/blob/main/GAN.JPG)  

- **目标函数**  
GAN的目标函数如下式所示：
L_GAN (G,D)=E_y [logD(y)]+E_x [log⁡(1-D(G(x))]
其中，x表示随机产生的噪声，通常是高斯噪声，y表示来自数据集的真实样本，E_y [logD(y)]表示输入一批真实样本后Discriminator对其的判定情况，D(y)越趋近1表明Discriminator对于真实图片较敏感。D(G(x))表示生成器所生成图像的真实程度，其值越小则1-D(G(x))越趋近于1，取对数则越趋近于0，这样就实现了零和博弈。  

- **训练步骤**  
首先固定Generater，当输入一个噪声x便产生一个样本，输入真实数据y就产生一个D值，max(D)表示辨别器对真实样本的判定越准确，min(G)表示辨别器对生成样本的判定越准确，在这样的相互对抗中，模型将被训练到最优。  

- **缺陷**  
GAN这种不需要预先建模的方法的缺点是太过自由，对于像素较多的图像，GAN会变得不太可控，于是研究者们通过给生成器和辨别器增加一些条件性的约束，来解决训练过于自由的问题。  

#### 4.2.2条件生成式对抗网络  
- **基本思想**  
CGAN是带有约束条件的生成对抗模型，他在生成器和辨别器的建模中加入了条件变量，条件变量可以视标签也可以是图像，使用这个额外的条件变量，对生成器进行数据的生成具有指导作用。  
- **网络结构**  
![CGAN网络结构](https://github.com/AlexanderGuan/Key-algorithm-of-human-motion-visual-monitoring-based-on-weak-supervised-learning/blob/main/CGAN.JPG)  
CGAN的网络结构如图所示，其中上方的线框为辨别器，下方线框为生成器，生成器根据高斯噪声和原始图像生成伪标签，生成的伪标签作为输入交付给辨别器，辨别器结合原始图像来判断输入图像的真伪。
在生成器模型中，条件变量y实际上是作为一个额外的输入层（additional input layer），它与生成器的噪声输入z组成了一个联合的隐层表达。在辨别器模型中，条件变量y与真实数据x同样同时作为输入数据，再进行训练。
和原始的GAN相比，CGAN在生成器的输入和判别器的输入中都加入了条件变量y，目的是有条件地监督生成器生成的数据，将原本的无监督GAN变为了半监督，从而提高模型预测的准确率。

-**目标函数**  
L_CGAN (G,D)=E_(x,y) [logD(x,y)]+E_(y,z) [log⁡(1-D(G(y,z))]  

### 4.3 UNet++ Generater网络  
- **网络结构**  
![UNet网络结构](https://github.com/AlexanderGuan/Key-algorithm-of-human-motion-visual-monitoring-based-on-weak-supervised-learning/blob/main/UNet.JPG)  

- **损失函数**  
![Unet function](https://github.com/AlexanderGuan/Key-algorithm-of-human-motion-visual-monitoring-based-on-weak-supervised-learning/blob/main/UNet%20function.JPG)

## 五、引用  
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
