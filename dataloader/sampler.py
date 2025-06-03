import torch
import numpy as np
import copy


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class BasePreserverCategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.arange(len(self.m_ind))
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class NewCategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        self.classlist = np.arange(np.min(label), np.max(label) + 1)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for c in self.classlist:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch





class CategoriesSupportSampler():
    def __init__(self, label, n_batch, n_cls, n_per, all_cls):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.all_cls = all_cls
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            class_list = [x + (self.all_cls-self.n_cls) for x in range(self.n_cls)]
            for j in class_list:
                feature_idx = torch.tensor(
                    np.random.choice(self.m_ind[j], self.n_per, replace=False))
                batch.append(feature_idx)
            batch = torch.stack(batch).reshape(-1)
            yield batch


class CategoriesSampler_train():
    def __init__(self, label, n_batch, n_cls, n_per, all_cls):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.all_cls = all_cls

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            class_list = range(self.n_cls)
            for j in class_list:
                feature_idx = torch.tensor(
                    np.random.choice(self.m_ind[j], self.n_per, replace=False))
                batch.append(feature_idx)
            batch = torch.stack(batch).reshape(-1)

            yield batch

class CategoriesSampler_test():
    def __init__(self, label, n_batch, n_cls, n_per, all_cls):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.all_cls = all_cls

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            class_list = [x for x in range(self.n_cls)]
            for j in class_list:
                feature_idx = torch.tensor(
                    np.random.choice(self.m_ind[j], self.n_per, replace=True))
                batch.append(feature_idx)
            batch = torch.stack(batch).reshape(-1)
            yield batch

class CategoriesSampler_train_preudo():
    def __init__(self, label, n_batch, n_cls, n_per, all_cls):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.all_cls = all_cls

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            class_list = [x + 10 for x in range(self.n_cls)]
            for j in class_list:
                feature_idx = torch.tensor(
                    np.random.choice(self.m_ind[j], self.n_per, replace=False))
                batch.append(feature_idx)
            batch = torch.stack(batch).reshape(-1)

            yield batch


class CategoriesSampler_test_preudo():
    def __init__(self, label, n_batch, n_cls, n_per, all_cls):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.all_cls = all_cls

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            class_list = [x + 10 for x in range(self.n_cls)]
            for j in class_list:
                feature_idx = torch.tensor(
                    np.random.choice(self.m_ind[j], self.n_per, replace=True))
                batch.append(feature_idx)
            batch = torch.stack(batch).reshape(-1)

            yield batch


class CategoriesSupportSampler_test():

    def __init__(self, label, n_batch, n_cls, n_per, all_cls):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.all_cls = all_cls
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            class_list = range(self.all_cls)
            class_list = np.random.choice(class_list, self.n_cls, replace=False)
            for j in class_list:
                feature_idx = torch.tensor(
                    np.random.choice(self.m_ind[j], self.n_per, replace=False))
                batch.append(feature_idx)
            batch = torch.stack(batch).reshape(-1)
            yield batch










if __name__ == '__main__':
    q = np.arange(5, 10)
    print(q)
    y = torch.tensor([5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 5, 5, 55, ])
    label = np.array(y)
    m_ind = []
    for i in range(max(label) + 1):
        ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
        ind = torch.from_numpy(ind)
        m_ind.append(ind)
    print(m_ind, len(m_ind))
