# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import torch

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)  # 次迭代 trainloader 中的一个批次时，你会在终端看到一个动态更新的进度条
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)  # 通过模型前向传播获取预测 logits
        logits = logits[:, :args.base_class]
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()  # 清零模型参数的梯度，以便进行下一次反向传播
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度更新模型参数
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()  # 设置模型为评估模式

    # shuffle=False: 如果设置为True，则在每个时代（epoch）开始时打乱数据。在评估或测试时，通常不需要打乱数据。
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform  # transform通常是一个数据预处理或增强的操作
    embedding_list = []
    label_list = []
    # data_list=[]

    # 提取训练集中每个类别的类别原型
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            #  将模型的模式设置为'encoder'
            #  在深度学习中，一个模型可能有不同的模式，比如在训练期间用作分类器，而在测试期间用作特征提取器。
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())

    # 沿指定维度拼接张量列表
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []   # 原型

    for class_index in range(args.base_class):
        # 在 label_list 中找到值等于 class_index 的元素的索引
        # label_list == class_index：创建一个布尔张量，其元素为 True（1）或 False（0）
        # nonzero()：找到布尔张量中非零元素的索引
        data_index = (label_list == class_index).nonzero()

        # 对 embedding_list 中属于同一类别的嵌入向量进行平均，得到原型
        embedding_this = embedding_list[data_index.squeeze(-1)]

        #----------------------mean-------------------------------#
        # embedding_this = embedding_this.mean(0)

        #----------------------k-means------------------------------#

        # 将 GPU 上的张量移动到 CPU 上
        embedding_cpu = embedding_this.cpu().numpy()

        # 使用 NumPy 进行聚类
        # 在这里，你可以使用你原来的聚类代码，比如 KMeans 等
        # 注意：此处的代码应该接收 NumPy 数组，而不是 PyTorch 张量
        # 例如，如果你使用 scikit-learn 的 KMeans：

        kmeans = KMeans(n_clusters=1, random_state=42)
        kmeans.fit(embedding_cpu)

        # 获取聚类中心
        center_point_numpy = kmeans.cluster_centers_

        # 将 NumPy 数组转换为 PyTorch 张量
        embedding_this = torch.tensor(center_point_numpy[0])

        # 将 PyTorch 张量移动到 GPU
        embedding_this = embedding_this.cuda()

        #-----------------------------------------------------#


        proto_list.append(embedding_this)

    # proto_list 中的每个元素代表一个类别的原型向量，通过堆叠这些向量，可以得到一个矩阵，其中每行代表一个类别的平均嵌入向量
    proto_list = torch.stack(proto_list, dim=0)

    #  proto_list 中的平均嵌入向量赋值给模型中全连接层（fc）的权重
    # [:args.base_class]：选择全连接层权重的前 args.base_class 行改变，其余不变
    model.module.fc.weight.data[:args.base_class] = proto_list

    # 将 fc 的权重赋值给 fc_old
    model.module.fc_old.weight.data.copy_(model.module.fc.weight.data)

    return model




def test(model, testloader, epoch, args, session, validation=True):
    breakpoint()
    test_class = args.base_class + session * args.way  # 计算当前测试的类别数
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5= Averager()
    lgt=torch.tensor([])  # 模型输出
    lbs=torch.tensor([])  # 真实标签
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)  # 使用模型对数据进行前向传播，得到预测的逻辑输出
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)  # 计算交叉熵损失
            acc = count_acc(logits, test_label)
            top5acc=count_acc_topk(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            # 每个测试批次结束时，lgt 和 lbs 中都包含了之前所有测试批次中的所有模型输出和真实标签
            # 这对于后续计算混淆矩阵等评估指标非常有用
            lgt=torch.cat([lgt,logits.cpu()])
            lbs=torch.cat([lbs,test_label.cpu()])

        vl = vl.item()  # 转换为标量值
        va = va.item()
        va5= va5.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va,va5))

        # lgt.view(-1, test_class) 将 lgt 转换为一个二维张量，其中每行对应一个样本，每列对应一个类别的logit值
        # lbs.view(-1) 则将 lbs 转换为一维张量
        lgt=lgt.view(-1,test_class)
        lbs=lbs.view(-1)
        breakpoint()
        print('validation=',validation)
        if validation is not True:
            print("enter")
            # 在模型进行测试时选择是否计算混淆矩阵和打印额外的评估指标
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm=confmatrix(lgt,lbs,save_model_dir)
            perclassacc=cm.diagonal()
            base_acc=np.mean(perclassacc[:args.base_class])
            inc_acc=np.mean(perclassacc[args.base_class:])
            print('Base Acc:',base_acc, 'Inc ACC:', inc_acc)
    return vl, va
