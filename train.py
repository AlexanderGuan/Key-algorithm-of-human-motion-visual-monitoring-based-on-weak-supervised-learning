"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import sys
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options获得训练选项(opt是入参的解析器)
    # 调用base_options类中的parse方法, 返回有所的add_argument()
    # ? 怎样调用了Trainoptions方法

    dataset = create_dataset(
        opt)  # create a dataset given opt.dataset_mode and other options根据opt.dataset_mode和其它选项创建数据集

    dataset_size = len(dataset)  # get the number of images in the dataset.获取数据集中的图像数目
    print('The number of training images = %d' % dataset_size)
    #    print(dataset)#<data.CustomDatasetDataLoader object at 0x7f9ea2deff70>
    #    sys.exit(0)

    model = create_model(opt)  # 根据给定的opt.model和其他选项创建模型# create a model given opt.model and other options
    # print(model)  # <models.pix2pix_model.Pix2PixModel object at 0x7f5458730f10>
    # print(type(model))  # <class 'models.pix2pix_model.Pix2PixModel'>
    # sys.exit(0)
    model.setup(opt)  # 装载并打印网络;创建调度器regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots创建可以显示/保存图像和损失图的可视化工具
    total_iters = 0  # 训练中迭代总次数 # the total number of training iterations

    # args = opt.parse_args()
    # print(opt)#保存至pix2pix/relative_txt/opt.txt中
    # print(type(opt))#<class 'argparse.Namespace'>
    # print(vars(opt))#保存至pix2pix/relative_txt/vars_opt.txt中
    # sys.exit(0)

    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>不同epoch的外循环，通过<epoch_count>, <epoch_count>+<save_latest_freq>保存模型
        epoch_start_time = time.time()  # 针对全部epoch的计时器# timer for entire epoch
        iter_data_time = time.time()  # 针对每个迭代中读取数据的计时器 # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch当前epoch中的迭代总数，每次epoch设置为0
        visualizer.reset()  # 重置可视化工具，确保其将结果保存至HTML（每个epoch中至少一次）# reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()  # 每个epoch开始时更新学习率 # update learning rates in the beginning of every epoch.

        '''每个data是一个字典，代表训练集的两张图片及其文件路径，包含4个key与4个value;
        4个key分别是 'A','B','A_paths'和'B_paths';
        'A' 和'B'代表图片，其value是维度为[1, 3, 256, 256]的tensor;'A_paths'和'B_paths'代表文件路径，其value为文件路径（字符串）
        tensor的维度是torch.Size([1, 3, 256, 256])，代表图片是三通道，像素值为256×256'''

        '''for i, data in enumerate(dataset):
            print(len(data))  # 4
            print(data['A'].shape) # torch.Size([1, 3, 256, 256])
            sys.exit(0)'''
        # print(len(dataset))#2000
        # print(type(dataset))#<class 'data.CustomDatasetDataLoader'>

        # print(data)#保存至pix2pix/relative_txt/data.txt(一个data, 同时也是dataset[0])
        # print(type(data))  # <class 'dict'>
        # print(dataset)  # <data.CustomDatasetDataLoader object at 0x7f848cddaf70>
        # print(type(dataset))#<class 'data.CustomDatasetDataLoader'>
        # sys.exit(0)

        for i, data in enumerate(dataset):  # 每个epoch中的内循环；从dataset中取两行图片（图像和其真值标签） # inner loop within one epoch
            iter_start_time = time.time()  # 计时器，记录每次迭代的计算时间 # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # print(type(data))#<class 'dict'>
            # print(data)  # 保存至pix2pix/relative_txt/data.txt
            # sys.exit(0)

            model.set_input(data)  # 将数据从数据集中解包出来并进行处理 # unpack data from dataset and apply preprocessing
            #            print(type(model))#<class 'models.pix2pix_model.Pix2PixModel'>
            #            print(model)#<models.pix2pix_model.Pix2PixModel object at 0x7f1a2f8feee0>
            #            sys.exit(0)

            # print(model)  # <models.pix2pix_model.Pix2PixModel object at 0x7f6fb05d5f40>
            # print(type(model))  # <class 'models.pix2pix_model.Pix2PixModel'>
            # sys.exit(0)
            model.optimize_parameters()  # 计算损失函数，获取梯度值， 更新网络权重# calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # 将图像展示到visdom并保存到HTML文件夹 # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # 打印每个epoch耗费的时间
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
