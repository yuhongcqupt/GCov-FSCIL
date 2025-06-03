import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000', 'mini_imagenet_withpath']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['cub200','manyshotcub']:
            # pretrained=True follow TOPIC, models for cub is imagenet pre-trained.
            # https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            # 在处理 CUB-200 数据集时使用了 ImageNet 预训练的权重
            self.encoder = resnet18(True, args)
            self.num_features = 512
        # 通过 nn.AdaptiveAvgPool2d 添加了一个自适应平均池化层，将输入的任意大小的特征图池化成大小为 (1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 定义了一个全连接层 self.fc，该层的输入尺寸为 self.num_features，输出尺寸为 self.args.num_classes，并且没有偏置项
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

    def forward_metric(self, x):  # 进行度量学习的前向计算
        x = self.encode(x)  # 使用模型的 encode 方法对输入 x 进行特征提取
        if 'cos' in self.mode:  # 使用余弦相似度进行度量学习
            # F.normalize(x, p=2, dim=-1)对x在最后一个维度上（ dim=-1）进行L2范数标准化
            # F.linear(...): 这一步实现了线性变换，具体地说，它计算了标准化后的输入 x 与标准化后的全连接层权重之间的点积
            # 这相当于计算了余弦相似度
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            # 将余弦相似度乘以一个温度参数，，超参
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def encode(self, x):
        # 输入 x 通过预定义的神经网络模型 encoder 进行特征提取
        # 在这里，encoder 是一个 ResNet 模型
        x = self.encoder(x)
        # 将每个通道的特征图变为一个标量值
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        # 如果 mode 不是 'encoder'，则调用 forward_metric 方法进行度量或分类操作
        # 如果 mode 是 'encoder'，则调用 encode 方法提取特征表示
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)
            breakpoint()
            print('yes！！update the fc with proto to incremental session!!\n')

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            #print(class_index)
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)  # 用原型更新全连接层
            self.fc.weight.data[class_index]=proto
            #print(proto)
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))


    # 对全连接层进行进一步微调（fine-tuning）的过程
    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        # 新全连接层参数就被复制到了模型中相应位置
        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

