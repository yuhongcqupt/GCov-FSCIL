# 这个文件中的函数和类提供了一些基本的工具和功能，可以在项目中的其他模块中被调用和使用。
import random
import torch
import os
import time
import numpy as np
import pprint as pprint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import matplotlib
_utils_pp = pprint.PrettyPrinter()

def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    """
    Args:
        logits: tensor([[0.1, 0.9, 0.2],
                        [0.3, 0.5, 0.7]])
        label=true label:tensor([1, 2])

    Returns: 标量-准确率

    pred_label 预测标签 形状同label
    torch.argmax 找到张量上最大值的索引
    (pred == label) 生成布尔张量[true, false]
    type(torch.cuda.FloatTensor) 转为浮点数张量 [1.0, 0.0]
    mean() 将得到的均值张量[0.5]
    item() 转为标量0.5
    """
    pred_label = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred_label == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred_label == label).type(torch.FloatTensor).mean().item()



def count_acc_inc(pred_label, label):
    """
    Args:
        logits: tensor([[0.1, 0.9, 0.2],
                        [0.3, 0.5, 0.7]])
        label=true label:tensor([1, 2])

    Returns: 标量-准确率

    pred_label 预测标签 形状同label
    torch.argmax 找到张量上最大值的索引
    (pred == label) 生成布尔张量[true, false]
    type(torch.cuda.FloatTensor) 转为浮点数张量 [1.0, 0.0]
    mean() 将得到的均值张量[0.5]
    item() 转为标量0.5
    """
    if torch.cuda.is_available():
        return (pred_label == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred_label == label).type(torch.FloatTensor).mean().item()
#---------------------------#
#新写的准确率计算方法用到了类mynet的属性adj_matrix 需要写到该类下
#----------------------------#

def count_acc_all(pre_cov,pre_teen,label):
    if torch.cuda.is_available():
        correct = ((pre_cov == label) | (pre_teen == label)).type(torch.cuda.FloatTensor).mean().item()
    else:
        correct = ((pre_cov == label) | (pre_teen == label)).type(torch.FloatTensor).mean().item()
    return correct



def count_acc_topk(x,y,k=2):
    """
    计算top2的准确率
    Args:
        x: logits=tensor([0.0780,  0.0629,  0.3039],
                        [-0.4277, -0.0775,  0.0474])
        y:label =tensor([[4],
                        [5],
                        [5]])

    """
    _,max_k = torch.topk(x,k,dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1)  # 把一行label张量换成竖排
    #top1=(test_labels == maxk[:,0:1]).sum().item()
    topk=(test_labels == max_k).sum().item()
    return float(topk/total)


def count_base_top2ac(args, logits, labels, k=2):
    """
    计算基类测试样本的top2准确率

    """
    total = 0
    topk = 0
    for i in range(logits.size(0)):

        label = labels[i]  # 获取第i个样本的真实标签
        if label < args.base_class:  # 60
            total = total + 1
            _, maxk = torch.topk(logits[i], k, dim=-1)  # 第i个样本前k个类别索引
            if torch.eq(maxk, label).any().item():  # 若真实样本在k个其中
                topk += 1
    return topk, total

def count_base_inc_ac(args, labels, pre_labels):
    """
    计算基类测试样本的top1准确率

    """
    topk_base = 0
    topk_inc = 0

    for i in range(labels.size(0)):

        true_label = labels[i]  # 获取第i个样本的真实标签
        if true_label < args.base_class:  # 60
            pre_label = pre_labels[i]  # 第i个样本前预测类别索引
            if pre_label == true_label:  # 若真实样本在k个其中
                topk_base += 1
        else:
            pre_label = pre_labels[i]
            if pre_label == true_label:  # 若真实样本在k个其中
                topk_inc += 1
    return topk_base, topk_inc

def count_base_inc(args, labels, pre_labels):
    """
    计算基类测试样本的top1准确率

    """
    topk_base = 0
    topk_inc = 0

    for i in range(labels.size(0)):

        true_label = labels[i]  # 获取第i个样本的真实标签
        if true_label < args.base_class:  # 60
            pre_label = pre_labels[i]  # 第i个样本前预测类别索引
            if pre_label == true_label:  # 若真实样本在k个其中
                topk_base += 1
        else:
            pre_label = pre_labels[i]
            if pre_label == true_label:  # 若真实样本在k个其中
                topk_inc += 1
    return topk_base, topk_inc

def count_inc_top2ac(args, logits, labels, session, k=2):
    """
    计算增量类测试样本的top2准确率

    """
    total = 0
    topk = 0
    for i in range(logits.size(0)):  # 对于单个样本
        label = labels[i]  # 获取样本的真实标签
        if (args.base_class - 1) < label < (args.base_class + (session * args.way)):
            total = total + 1
            _, maxk = torch.topk(logits[i], k, dim=-1)  # 样本前k个类别索引
            if torch.eq(maxk, label).any().item():  # 若label与maxk中的一个值相等，则加1
                topk += 1
    return topk, total




def count_acc_taskIL(logits, label,args):
    basenum=args.base_class
    incrementnum=(args.num_classes-args.base_class)/args.way
    for i in range(len(label)):
        currentlabel=label[i]
        if currentlabel<basenum:
            logits[i,basenum:]=-1e9
        else:
            space=int((currentlabel-basenum)/args.way)
            low=basenum+space*args.way
            high=low+args.way
            logits[i,:low]=-1e9
            logits[i,high:]=-1e9

    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def confmatrix(logits,label,filename):
    
    font={'family':'FreeSerif','size':18}
    matplotlib.rc('font',**font)
    matplotlib.rcParams.update({'font.family':'FreeSerif','font.size':18})
    plt.rcParams["font.family"]="FreeSerif"

    pred = torch.argmax(logits, dim=1)
    cm=confusion_matrix(label, pred,normalize='true')
    #print(cm)
    clss=len(cm)
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    cax = ax.imshow(cm,cmap=plt.cm.jet) 
    if clss<=100:
        plt.yticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=16)
        plt.xticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=16)
    elif clss<=200:
        plt.yticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=16)
        plt.xticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=16)
    else:
        plt.yticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=16)
        plt.xticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=16)

    plt.xlabel('Predicted Label',fontsize=20)
    plt.ylabel('True Label',fontsize=20)
    plt.tight_layout()
    plt.savefig(filename+'.pdf',bbox_inches='tight')
    plt.close()

    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    cax = ax.imshow(cm,cmap=plt.cm.jet) 
    cbar = plt.colorbar(cax) # This line includes the color bar
    cbar.ax.tick_params(labelsize=16)
    if clss<=100:
        plt.yticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=16)
        plt.xticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=16)
    elif clss<=200:
        plt.yticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=16)
        plt.xticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=16)
    else:
        plt.yticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=16)
        plt.xticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=16)
    plt.xlabel('Predicted Label',fontsize=20)
    plt.ylabel('True Label',fontsize=20)
    plt.tight_layout()
    plt.savefig(filename+'_cbar.pdf',bbox_inches='tight')
    plt.close()

    return cm





def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count