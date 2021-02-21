"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
import sys
from models.base_model import BaseModel


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".
    导入模块"models/[model_name]_model.py"
    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    在这个文件中，类DatasetNameModel()将被实例化。它必须成为BaseModel的一个子类，
    并且不需要区分大小写
    """
    model_filename = "models." + model_name + "_model"  # model_filename=models.pix2pix_model
    # print(model_filename)#models.pix2pix_model
    # sys.exit(0)
    modellib = importlib.import_module(
        model_filename)  # importlib.import_module可以通过字符串来导入模块，同一文件夹下字符串为模块名。相当于import models.pix2pix_model。详见https://www.cnblogs.com/wioponsen/p/13672012.html
    # print(modellib)  # <module 'models.pix2pix_model' from '/home/guan/Desktop/github/pix2pix/models/pix2pix_model.py'>
    # sys.exit(0)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'  # pix2pixmodel
    #    print(target_model_name)  # pix2pixmodel
    #    sys.exit(0)

    # print(modellib.__dict__)
    # print(type(modellib.__dict__))  # <class 'dict'>
    # print(type(modellib.__dict__.items()))# <class 'dict_items'>
    # sys.exit(0)

    for name, cls in modellib.__dict__.items():  # __dict__返回字典, items()以元组的方式返回字典, name是字典中的每个键， cls是字典的每个值
        # print('name: {}'.format(name)) #name: Pix2PixModel
        # print(type(name))# <class 'str'>
        # print('cls: {}'.format(cls))# cls: <class 'models.pix2pix_model.Pix2PixModel'>
        # print(type(cls))# 基本是<class 'str'>，但也有一些其它类型
        if name.lower() == target_model_name.lower() \
                and issubclass(cls, BaseModel):  # 如果name的小写为pix2pixmodel, 且如果cls为BaseModel的子类(返回True或False)
            model = cls  # model是pix2pix_model.py中的Pix2PixModel类型
            # print(model)          #<class 'models.pix2pix_model.Pix2PixModel'>
            # print(type(model))    #<class 'abc.ABCMeta'>
    # sys.exit(0)

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.
    根据给定选项创建模型
    This function warps the class CustomDatasetDataLoader.
    本函数将类CustomDatasetDataLoader封装
    This is the main interface between this package and 'train.py'/'test.py'
    这是本package与train.py/test.py之间的主接口

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)  # ? 参数是什么意思
    print("model [%s] was created" % type(instance).__name__)
    return instance
