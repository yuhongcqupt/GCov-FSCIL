import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from sklearn.cluster import KMeans
import numpy as np


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100', 'manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
            self.rotate_classes = 0
        if self.args.dataset in ['mini_imagenet', 'manyshotmini', 'imagenet100', 'imagenet1000']:
            self.encoder = resnet18(False, args)
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
        self.fc_old = nn.Linear(self.num_features, self.pre_allocate, bias=False)

        nn.init.orthogonal_(self.fc_old.weight)
        nn.init.orthogonal_(self.fc.weight)

        self.dummy_orthogonal_classifier = nn.Linear(self.num_features, self.pre_allocate - self.args.base_class,
                                                     bias=False)
        self.dummy_orthogonal_classifier.weight.requires_grad = False
        self.dummy_orthogonal_classifier.weight.data = self.fc.weight.data[self.args.base_class:, :]
        self.pre_allocate = self.args.num_classes
        self.adj_matrix = torch.zeros((self.pre_allocate, self.pre_allocate))

        self.proto_box = torch.Tensor().cuda()
        self.aux_proto_box = torch.tensor([]).cuda()
        self.proto_weights = torch.tensor([]).cuda()
        self.distinct_proto = torch.tensor([]).cuda()
        self.sim_matrix = torch.zeros(self.pre_allocate - self.args.base_class, 10).cuda()
        self.sim_proto = torch.zeros(self.pre_allocate - self.args.base_class, 64).cuda()
        print('aux_proto_box initialized over.')

        print(self.dummy_orthogonal_classifier.weight.data.size())
        print('self.dummy_orthogonal_classifier.weight initialized over.')
        self.radius = 0
        self.cov_mats, self.base_cov_mats = [], []
        self.cov = True
        self.generator = True

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x2 = F.linear(F.normalize(x, p=2, dim=-1),
                          F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))

            x = torch.cat([x1[:, :self.args.base_class], x2], dim=1)

            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def forpass_fc(self, x, multi):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def forpass_fc_old(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:

            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc_old.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc_old(x)
            x = self.args.temperature * x
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def pre_encode(self, x):

        if self.args.dataset in ['cifar100', 'manyshotcifar']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)

        elif self.args.dataset in ['mini_imagenet', 'manyshotmini', 'cub200']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)

        return x

    def post_encode(self, x):
        if self.args.dataset in ['cifar100', 'manyshotcifar']:

            x = self.encoder.layer3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        elif self.args.dataset in ['mini_imagenet', 'manyshotmini', 'cub200']:

            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x

        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, dataloader, class_list, session, radius, model_g, gen_support_input):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]

            data = self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list, radius, session, model_g, gen_support_input)
            print('yes！！update the fc with proto to incremental session!!\n')

        if 'ft' in self.args.new_mode:
            self.update_fc_ft(new_fc, data, label, session)
        return new_fc

    def update_fc_avg(self, data, label, class_list, radius, session, model_g, gen_support_input):
        """
        Args:
            data: 该阶段全部样本特征张量
            label: 真实标签张量
            class_list: 类别数组
        Returns: new_fc 增量原型张量
        """
        new_fc = []
        emb = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]

            if self.generator:
                model_g.eval()
                gen_support = gen_support_input.cuda()
                gen_support = self.encode(gen_support).detach()
                gen_support = gen_support.view(90, 2, -1)
                proto, diversity_loss, support_all, generate_all = model_g(torch.unsqueeze(embedding, 0), gen_support)
                embedding = support_all[0]
                self.fc.weight.data[class_index] = proto
                print('生成样本成功，已加权到原型中')
            else:
                proto = embedding.mean(0)
                self.fc.weight.data[class_index] = proto

            if self.cov:
                cov = torch.cov(embedding.T)
                self.cov_mats.append(cov)
            proto = embedding.mean(0)
            emb.append(embedding)
            new_fc.append(proto)

        if self.cov:
            ridge = 100
            for idd, class_index in enumerate(class_list):
                self.cov_mats[class_index] = torch.corrcoef(self.shrink_cov(self.cov_mats[class_index], ridge))

        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self, new_fc, data, label, session):
        new_fc = new_fc.clone().detach()
        new_fc.requires_grad = True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, momentum=0.9, dampening=0.9,
                                    weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data, fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[
        self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(
            new_fc.data)

    def soft_calibration(self, args, session):
        base_protos = self.fc.weight.data[:self.args.base_class].detach().cpu()
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        cur_protos = self.fc.weight.data[self.args.base_class + (session - 1) * self.args.way: self.args.base_class +session * self.args.way].detach().cpu()
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)

        weights = torch.mm(cur_protos, base_protos.T) * self.args.temperature
        norm_weights = torch.softmax(weights, dim=1)

        delta_protos = torch.matmul(norm_weights, base_protos)
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)

        updated_protos = (1 - args.shift_weight) * cur_protos + args.shift_weight * delta_protos

        self.fc_old.weight.data = torch.cat([self.fc.weight.data[:, :self.args.base_class], self.fc.weight.data[:, -100:]], dim=1)
        self.fc.weight.data[
        self.args.base_class+ (session - 1) * self.args.way: self.args.base_class +session * args.way] = updated_protos

    def get_distinct_proto(self, emb, session, way):
        base_protos = self.fc.weight.data[:self.args.base_class]
        cur_protos = self.fc.weight.data[self.args.base_class + (session - 1) * self.args.way + way]

        cosin_sim = F.cosine_similarity(cur_protos.unsqueeze(0), base_protos, dim=1)
        _, top10_base = torch.topk(cosin_sim, k=10, dim=-1)
        self.sim_matrix[(session - 1) * self.args.way + way] = top10_base
        cur_protos = F.normalize(cur_protos.detach().cpu(), p=2, dim=-1)
        sim_base_proto = F.normalize(base_protos[top10_base].detach().cpu(), p=2, dim=-1)
        weights = torch.matmul(cur_protos, sim_base_proto.T) * self.args.temperature
        norm_weights = torch.softmax(weights, dim=0)
        delta_protos = torch.matmul(norm_weights, sim_base_proto)
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        updated_protos = (1 + 1) * cur_protos - 1 * delta_protos
        self.sim_proto[(session - 1) * self.args.way + way] = updated_protos

    def update_adj(self, new_features, session):
        self.proto_box = self.proto_box.to(new_features.device)
        self.proto_box = torch.cat((self.proto_box, new_features), dim=0)
        return self.adj_matrix

    def get_pre_lables_logits(self, logits, emb, j, test_label, type1, type2, type3):
        index_old = torch.argmax(logits[:, :60], dim=1)
        index_new = torch.argmax(logits[:, 60:], dim=1) + 60
        proto_1 = self.aux_proto_box[:, 0, :]
        proto_2 = self.aux_proto_box[:, 1, :]
        for i in range(emb.size(0)):
            if logits[i][index_old[i]] > logits[i][index_new[i]] and torch.any(
                    self.sim_matrix[index_new[i] - 60] == index_old[i]).item():
                proto_1_select = proto_1[index_old[i]]
                proto_2_select = proto_2[index_old[i]]
                x_1 = F.linear(F.normalize(proto_1_select, p=2, dim=-1),
                               F.normalize(self.sim_proto[index_new[i] - 60], p=2, dim=-1))
                x_2 = F.linear(F.normalize(proto_2_select, p=2, dim=-1),
                               F.normalize(self.sim_proto[index_new[i] - 60], p=2, dim=-1))
                x_new = F.linear(F.normalize(emb[i], p=2, dim=-1),
                                 F.normalize(self.sim_proto[index_new[i] - 60], p=2, dim=-1))
                x_new_proto = F.linear(F.normalize(self.fc.weight.data[index_new[i]], p=2, dim=-1),
                                       F.normalize(self.sim_proto[index_new[i] - 60], p=2, dim=-1))
 
                x = torch.stack([x_1, x_2, x_new, x_new_proto]) * self.args.temperature
                mean_old = torch.min(abs(x[0] - x[2]), abs(x[1] - x[2]))
                mean_new = abs(x[2] - x[3])

                if mean_old < mean_new:
                    type1 = type1 + 1
                    max_value = torch.max(logits[i][60:])
                    logits[i][:60][logits[i][:60] >= max_value] = 0
                    if test_label[i] == index_new[i]:
                        type2 = type2 + 1
                    if test_label[i] == index_old[i]:
                        type3 = type3 + 1

        return logits, type1, type2, type3

    def shrink_cov(self, cov, bridge):
        iden = torch.eye(len(cov)).cuda()
        cov_ = cov.cuda() + (bridge * iden).cuda()
        return cov_

    def eval_task(self, vectors, y_true):
        dists = self._maha_dist(vectors)
        scores = dists.T
        y_pred = np.argsort(scores, axis=1)[:, :1]
        nme_accy = None
        return y_pred, nme_accy

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = self.accuracy(y_pred.T[0], y_true, self._known_classes, self.args["init_cls"], self.args["increment"])
        ret["hm"] = self.harm_mean(grouped["old"], grouped["new"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        return ret

    def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
        assert len(y_pred) == len(y_true), "Data length error."
        all_acc = {}
        all_acc["total"] = np.around(
            (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
        )

        idxes = np.where(
            np.logical_and(y_true >= 0, y_true < init_cls)
        )[0]
        label = "{}-{}".format(
            str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
        # for incremental classes
        for class_id in range(init_cls, np.max(y_true), increment):
            idxes = np.where(
                np.logical_and(y_true >= class_id, y_true < class_id + increment)
            )[0]
            label = "{}-{}".format(
                str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
            )
            all_acc[label] = np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )

        # Old accuracy
        idxes = np.where(y_true < nb_old)[0]

        all_acc["old"] = (
            0
            if len(idxes) == 0
            else np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )
        )

        # New accuracy
        idxes = np.where(y_true >= nb_old)[0]
        all_acc["new"] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

        return all_acc

    def harm_mean(seen, unseen):
        harm_means = []
        for _seen, _unseen in zip([seen], [unseen]):
            _hmean = (2 * _seen * _unseen) / (_seen + _unseen + 1e-12)
            _hmean = float('%.3f' % (_hmean))
            harm_means.append(_hmean)
        return harm_means

    def _maha_dist(self, vector):
        vectors = vector.clone().detach().cuda()
        maha_dist = []
        for class_index in range(len(self.cov_mats)):
            dist = self._mahalanobis(vectors, self.fc.weight.data[class_index], self.cov_mats[class_index])
            maha_dist.append(dist)
        maha_dist = np.array(maha_dist)
        return maha_dist

    def _mahalanobis(self, vectors, class_means, cov):
        x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(class_means, p=2, dim=-1)
        if cov is None:
            cov = torch.eye(self._network.feature_dim)  # identity covariance matrix for euclidean distance
        inv_covmat = torch.linalg.pinv(cov).float().cuda()
        left_term = torch.matmul(x_minus_mu, inv_covmat)
        mahal = torch.matmul(left_term, x_minus_mu.T)
        return torch.diagonal(mahal, 0).cpu().numpy()

    def count_acc_inc_detach(self, pre_teen, pre_cov, emb, test_label):
        pre = pre_teen
        diff_indices = torch.nonzero(pre_cov != pre_teen).flatten()
        for idx in diff_indices:
            idx = idx.item()
            if pre_cov[idx] > 99 :
                pre[idx] = pre_cov[idx]
        return pre

    def count_detach(self, pre_teen, pre_cov, emb, test_label):
        cov_right = 0
        cov_right_base = 0
        teen_right = 0
        teen_right_base = 0
        array = []
        diff_indices = torch.nonzero(pre_cov != pre_teen).flatten()
        for idx in diff_indices:
            array.append([pre_cov[idx].item(), pre_teen[idx].item(), test_label[idx].item()])
            if pre_cov[idx] == test_label[idx]:
                cov_right += 1
                if test_label[idx] < 60:
                    cov_right_base += 1
            if pre_teen[idx] == test_label[idx]:
                teen_right += 1
                if test_label[idx] < 60:
                    teen_right_base += 1
        return cov_right, cov_right_base, teen_right, teen_right_base, array

    """======================================================================================="""

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output