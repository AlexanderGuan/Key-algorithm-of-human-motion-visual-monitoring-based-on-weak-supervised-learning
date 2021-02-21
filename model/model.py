import torch
from .base_model import BaseModel
from . import networks
import sys
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    该类实现了pix2pix模型，用于学习给定成对数据的从输入图像到输出图像的映射

    The model training requires '--dataset_mode aligned' dataset.
    模型训练需要'--dataset_mode aligned'的数据集

    By default, it uses a '--netG unet256' U-Net generator,
    默认情况下，它使用'--netG unet256'U-Net 生成器，
    a '--netD basic' discriminator (PatchGAN),
    和'--netD basic'辨别器(PatchGAN)，
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    和'--gan_mode' ？ GAN损失（原始GAN论文中使用的交叉熵目标）

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # 定义网络（同时定义生成器和辨别器）define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # 初始化优化器；调度器将自动通过函数<BaseModel.setup>创建
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        将input data从dataloader中解包出来并执行必要的预处理操作
        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        选项 'direction' 可以用来交换domain A与 domain B 之间的图像
        """

        """将一张已配对图像分为两张，一张为原始图像，一张为真值图像"""
        AtoB = self.opt.direction == 'AtoB'
        # （推测） real_A是标签，real_B是图像
        self.real_A = input['A' if AtoB else 'B'].to(
            self.device)  # 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。

        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        """进行向前传播；通过<optimize_parameters> and <test>函数均可调用"""

        self.fake_B = self.netG(self.real_A)  # G(A)  # (推测) 根据真值图像（标注图像）预测假图像, ? 为什么传入netG?netG没有相应的形参
        # print(self.fake_B.shape)#torch.Size([1, 3, 256, 256])
        # sys.exit(0)

        # print(self.fake_B)  # （推测）self.fake_B为根据真值标签生成的假图像，因为本条语句每次生成的张量不同
        # sys.exit(0)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        """为辨别器计算GAN损失"""
        # print(type(self.real_A))#<class 'torch.Tensor'>
        # print(len(self.real_A))#1
        # sys.exit(0)

        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),  # (推测) 把真值标签和G(x)结合到一起
                            1)  # 我们使用conditional GAN;我们需要将输入和输入都投入到辨别器中  （本句推测）将真值标签和产生的虚假图像结合到一个tensor中 # we use conditional GANs; we need to feed both input and output to the discriminator
        # 作者解释此处detach()的原因‘When updating D, we don't need to compute gradients for G. That's why we used detach in backward_D_basic.’
        # detach()相关讲解:https://blog.csdn.net/qq_39709535/article/details/80804003
        # https://blog.csdn.net/superjunenaruto/article/details/99942241     https://www.cnblogs.com/jiangkejie/p/9981707.html
        pred_fake = self.netD(
            fake_AB.detach())  # detach:重新声明一个变量,指向原变量的存放位置,但是requires_grad为false 详见https://www.jianshu.com/p/f1bd4ff84926
        # print(fake_AB.shape)                  #torch.Size([1, 6, 256, 256])
        # print(fake_AB.detach().shape)         #torch.Size([1, 6, 256, 256])
        # print(pred_fake.shape)                #torch.Size([1, 1, 30, 30])

        # print(self.real_A.shape)#torch.Size([1, 3, 256, 256]) #（推测）表示fake_AB是将self.real_A,与self.fake_B从深度方向(颜色三通道)拼接而成
        # print(self.fake_B.shape)#torch.Size([1, 3, 256, 256])
        # sys.exit(0)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)  # 计算损失。自动调用了GANLoss中的__call__函数，返回计算出的损失值（张量）
        # print(self.loss_D_fake)                       #tensor(0.5643, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
        # print(type(self.loss_D_fake))                 #<class 'torch.Tensor'>
        # sys.exit(0)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        # print(pred_real.shape)#torch.Size([1, 1, 30, 30])
        # sys.exit(0)

        # （推测）real_A是真值图像，real_B是原始图像，real_AB是将A和B从深度方向结合
        # print('self.real_A.shape: {}'.format(self.real_A.shape))  # self.real_A.shape: torch.Size([1, 3, 256, 256])
        # print('self.real_B.shape: {}'.format(self.real_B.shape))  # self.real_B.shape: torch.Size([1, 3, 256, 256])
        # print('real_AB.shape: {}'.format(real_AB.shape))  # real_AB.shape: torch.Size([1, 6, 256, 256])
        # print('pred_real.shape: {}'.format(pred_real.shape))  # pred_real.shape: torch.Size([1, 1, 30, 30])
        # sys.exit(0)
        self.loss_D_real = self.criterionGAN(pred_real, True)  # 计算真实图片的损失值
        # combine loss and calculate gradients
        # 结合损失值并计算梯度
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()  # 向后传播

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        '''为生成器计算GAN和L1损失'''
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        # print(fake_AB.shape)#torch.Size([1, 6, 256, 256])
        # print(type(pred_fake))#<class 'torch.Tensor'>
        # print(pred_fake.shape)  # torch.Size([1, 1, 30, 30])
        # sys.exit(0)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # 生成假图像G(A) (保存在self.fake_B中)# compute fake images: G(A)
        # 更新D update D
        # print(self.netD)  # 保存至pix2pix/relative_txt/netD.txt
        # print(type(self.netD)) # <class 'torch.nn.parallel.data_parallel.DataParallel'>
        # sys.exit(0)
        self.set_requires_grad(self.netD, True)  # 为D启动反向传播 # enable backprop for D
        self.optimizer_D.zero_grad()  # 将D的梯度设置为0 # set D's gradients to zero
        self.backward_D()  # 计算D的梯度 # calculate gradients for D
        self.optimizer_D.step()  # 更新D的权重 # update D's weights
        # 更新 update G
        self.set_requires_grad(self.netD, False)  # D在优化G时不需要梯度 # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # 将G的梯度设置为0 # set G's gradients to zero
        self.backward_G()  # 计算G的梯度 #calculate graidents for G
        self.optimizer_G.step()  # 更新G的权重 # udpate G's weights
