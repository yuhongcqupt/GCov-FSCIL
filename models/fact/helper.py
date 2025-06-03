# import new Network name here and add in model_class args
from copy import deepcopy

from sklearn.cluster import KMeans
import errno

from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats


def base_train(model, trainloader, optimizer, scheduler, epoch, args, mask):
    tl = Averager()
    ta = Averager()
    model = model.train()
    model.module.mode = 'cos'
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        beta = torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
        data, train_label = [_.cuda() for _ in batch]
        logits = model(data)
        logits_ = logits[:, :args.base_class]
        loss = F.cross_entropy(logits_, train_label)
        acc = count_acc(logits_, train_label)
        if epoch >= args.loss_iter:
            logits = torch.cat([logits[:, :args.base_class], logits[:, -100:]], dim=1)
            logits_MaskTrueMaxValue = logits.masked_fill(
                F.one_hot(train_label, num_classes=model.module.pre_allocate) == 1, -1e9)


            logits_PostClass = logits_MaskTrueMaxValue * mask[train_label]
            pseudo_label = torch.argmax(logits_PostClass[:, args.base_class:], dim=-1) + args.base_class

            loss2 = F.cross_entropy(logits_MaskTrueMaxValue, pseudo_label)

            RandOrder = torch.randperm(data.size(0)).cuda()
            train_label_NewOrder = train_label[RandOrder]

            emb1 = model.module.pre_encode(data)

            mixed_emb = beta * emb1 + (1 - beta) * emb1[RandOrder]
            mixed_logits = model.module.post_encode(mixed_emb)
            mixed_logits = torch.cat([mixed_logits[:, :args.base_class], mixed_logits[:, -100:]], dim=1)

            diff_id = train_label_NewOrder != train_label
            mixed_logits = mixed_logits[diff_id]

            pseudo_label_new = torch.argmax(mixed_logits[:, args.base_class:],
                                            dim=-1) + args.base_class

            loss3 = F.cross_entropy(mixed_logits, pseudo_label_new)

            pseudo_label_old = torch.argmax(mixed_logits[:, :args.base_class], dim=-1)
            mixed_logits_MaskPseudoMaxValue = mixed_logits.masked_fill(
                F.one_hot(pseudo_label_new, num_classes=model.module.pre_allocate) == 1, -1e9)
            loss4 = F.cross_entropy(mixed_logits_MaskPseudoMaxValue, pseudo_label_old)

            total_loss = loss + args.balance * (loss2 + loss3 + loss4)
        else:
            total_loss = loss

        lrc = optimizer.param_groups[0]['lr']
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(self, trainset, transform, model, args):
    model = model.eval()
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    proto_list = []
    for class_base in range(args.base_class):
        data_index = (label_list == class_base).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        radius_temp = []
        cov = torch.cov(embedding_this.T)
        radius_temp.append(torch.trace(cov) / embedding_this.size(1))

        embedding_proto = embedding_this.mean(0)
        proto_list.append(embedding_proto)
        model.module.cov_mats.append(cov)
        model.module.base_cov_mats.append(cov)

    proto_list = torch.stack(proto_list, dim=0)
    radius = np.sqrt(np.mean(radius_temp))
    print('radius')
    print(radius)
    model.module.fc.weight.data[:args.base_class] = proto_list
    model.module.fc_old.weight.data[:args.base_class] = proto_list
    bridge = 100
    for class_index in range(args.base_class):
        model.module.cov_mats[class_index] = torch.corrcoef(
            model.module.shrink_cov(model.module.cov_mats[class_index], bridge))
    return model, radius


def test(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    model.module.mode = 'cos'

    vl = Averager()
    va = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            print('Seen Acc:', seenac, 'Unseen ACC:', unseenac)
    return vl, va


def test_withfc(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model.module.forpass_fc(data, multi=True)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            print('Seen Acc:', seenac, 'Unseen ACC:', unseenac)
    return vl, va


def gen_train(gen_support_input, gen_train_loader, model_E, model_G, criterion, optimizer):
    x_dim = 224
    n_support = 90
    n_cls = 10
    cls_num = 20
    losses = AverageMeter()
    top1 = AverageMeter()
    model_G.train()
    model_E = model_E.eval()
    model_E.module.mode = 'encoder'

    gt = np.tile(range(n_cls), 15)
    gt.sort()
    gt = torch.cuda.LongTensor(gt)

    for inter in tqdm(range(5)):
        input, target = gen_train_loader.__iter__().next()
        input = input.cuda()
        train_input = input.view(n_cls, cls_num, 3, x_dim, x_dim)[:, -5:, :, :, :].contiguous().view(-1, 3, x_dim, x_dim)
        train_tmp = replace_to_rotate_(train_input, 5)
        train_tmp = model_E(train_tmp)
        train_tmp = train_tmp.view(n_cls, 5, -1)

        query_input = input.view(n_cls, cls_num, 3, x_dim, x_dim)[:, :-5, :, :, :].contiguous().view(-1, 3, x_dim, x_dim)
        query_tmp = replace_to_rotate_(query_input, 15)
        query_tmp = model_E(query_tmp)

        gen_support = gen_support_input.cuda()
        gen_support = model_E(gen_support)
        gen_support = gen_support.view(n_support, 2, -1)

        weight, diversity_loss, support_all, generate_all = model_G(train_tmp, gen_support)

        predict = F.linear(F.normalize(query_tmp, p=2, dim=-1), F.normalize(weight, p=2, dim=-1)) * model_G.s

        acc = (predict.topk(1)[1].view(-1) == gt).float().sum(0) / gt.shape[0] * 100.
        loss = criterion(predict, gt) + diversity_loss
        losses.update(loss.item(), predict.size(0))
        top1.update(acc.item(), predict.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg, top1.avg


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def validate(gen_support_input, gen_test_loader, model_E, model_G, criterion):
    x_dim = 224
    n_cls = 10
    cls_num = 20
    n_support = 90
    losses = AverageMeter()
    top1 = AverageMeter()
    model_G.eval()
    model_E = model_E.eval()
    model_E.module.mode = 'encoder'
    """标签"""
    gt = np.tile(range(n_cls), 15)
    gt.sort()
    gt = torch.cuda.LongTensor(gt)
    with torch.no_grad():
        accuracies = []
        for inter in tqdm(range(5)):
            input, target = gen_test_loader.__iter__().next()
            input = input.cuda()

            train_input = input.view(n_cls, cls_num, 3, x_dim, x_dim)[:, -5:, :, :, :].contiguous().view(-1, 3, x_dim, x_dim)
            train_tmp = replace_to_rotate_(train_input, 5)
            train_tmp = model_E(train_tmp)
            train_tmp = train_tmp.view(n_cls, 5, -1)

            query_input = input.view(n_cls, cls_num, 3, x_dim, x_dim)[:, :-5, :, :, :].contiguous().view(-1, 3, x_dim, x_dim)
            query_tmp = replace_to_rotate_(query_input, 15)
            query_tmp = model_E(query_tmp)

            gen_support = gen_support_input.cuda()
            gen_support = model_E(gen_support)
            gen_support = gen_support.view(n_support, 2, -1)

            weight, diversity_loss, support_all, generate_all = model_G(train_tmp, gen_support)

            predict = F.linear(F.normalize(query_tmp, p=2, dim=-1), F.normalize(weight, p=2, dim=-1)) * model_G.s

            acc = (predict.topk(1)[1].view(-1) == gt).float().sum(0) / gt.shape[0] * 100.
            accuracies.append(acc.item())
            loss = criterion(predict, gt) + diversity_loss
            losses.update(loss.item(), predict.size(0))
            top1.update(acc.item(), predict.size(0))

    mean, h = mean_confidence_interval(accuracies)
    return losses.avg, top1.avg, h


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(state, epoch, checkpoint='checkpoint_gen', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    torch.save(state, filepath)
    print('save checkpoint success', epoch)


def replace_to_rotate(proto_tmp, query_tmp):

    low_way = int(proto_tmp.shape[0] / 5)
    for i in range(low_way):
        rot_list = [90, 180, 270]
        sel_rot = [90, 180, 270, 90, 180, 270, 90, 180, 270, 90, 180, 270][i]
        if sel_rot == 90:
            for j in range(5 * i, 5 * (i + 1)):
                proto_tmp[j] = proto_tmp[j].transpose(1, 2).flip(1)
            for j in range(15 * i, 15 * (i + 1)):
                query_tmp[j] = query_tmp[j].transpose(1, 2).flip(1)
        elif sel_rot == 180:
            for j in range(5 * i, 5 * (i + 1)):
                proto_tmp[j] = proto_tmp[j].flip(1).flip(2)
            for j in range(15 * i, 15 * (i + 1)):
                query_tmp[j] = query_tmp[j].flip(1).flip(2)
        elif sel_rot == 270:
            for j in range(5 * i, 5 * (i + 1)):
                proto_tmp[j] = proto_tmp[j].transpose(1, 2).flip(2)
            for j in range(15 * i, 15 * (i + 1)):
                query_tmp[j] = query_tmp[j].transpose(1, 2).flip(2)
    return proto_tmp, query_tmp


def replace_to_rotate_(x, cls_num):
    low_way = int(x.shape[0] / cls_num)
    for i in range(low_way):
        sel_rot = [90, 180, 270, 90, 180, 270, 90, 180, 270, 90, 180, 270, 0, 180, 270, 90, 180, 270, 90, 180, 270, 90, 180, 270][i]
        if sel_rot == 90:
            for j in range(cls_num * i, cls_num * (i + 1)):
                x[j] = x[j].transpose(1, 2).flip(1)
        elif sel_rot == 180:
            for j in range(cls_num * i, cls_num * (i + 1)):
                x[j] = x[j].flip(1).flip(2)
        elif sel_rot == 270:
            for j in range(cls_num * i, cls_num * (i + 1)):
                x[j] = x[j].transpose(1, 2).flip(2)
    return x
