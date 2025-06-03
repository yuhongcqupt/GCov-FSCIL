import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

SEED = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def latent_loss(text_encoder):
    c_dim = list(text_encoder.size())[-1]  # 512 32
    # split the context into mean and variance predicted by task context encoder
    z_dim = c_dim // 2
    c_mu = text_encoder[:, :z_dim]
    c_log_var = text_encoder[:, z_dim:]
    z_mean = c_mu
    z_stddev = torch.exp(c_log_var / 2)
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def Task_concate(reference):
    r_way, r_shot, _ = reference.size()
    r_pair = []
    for i in range(0, r_way):  # 每个类
        current_class = []
        for j in range(0, r_shot):
            for k in range(j + 1, r_shot):
                pair_tempt = torch.cat((reference[i][j], reference[i][k]), 0)
                current_class.append(pair_tempt)
        current_class = torch.stack(current_class, 0)
        r_pair.append(current_class)
    r_pair = torch.stack(r_pair, 0)

    return r_pair


class GeneratorNet(nn.Module):
    # N_generate=64 生成64个
    def __init__(self, N_generate):
        super(GeneratorNet, self).__init__()
        self.N_generate = N_generate
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.encoder1 = Encoder1()
        self.encoder2 = Encoder2()
        self.sampler1 = Sampler1(self.N_generate)
        self.sampler2 = Sampler2()
        self.decoder = Decoder(self.N_generate)

    def forward(self, support, reference):
        delta = self.encoder1(reference)
        diversity = self.sampler1(delta)
        support_set = support.mean(1)
        way = support_set.shape[0]
        support_set = torch.unsqueeze(support_set, 1)  #(way,64) 种子
        prototype = torch.zeros(way, 512).cuda()
        support_all = torch.zeros(way, 1+support.shape[1], 512).cuda()  # 对生成样本平均为1个样本
        generate_all = torch.zeros(way, self.N_generate, 512).cuda()
        for i in range(way):
            current_support = support_set[i]
            current_generate_set = diversity * current_support + current_support  # (64,1024)*(1024)+(1024)=(64,64)
            generate_all[i] = current_generate_set
            current_mean = current_generate_set.mean(0).unsqueeze(0)
            support_all[i] = torch.cat((support[i], current_mean), dim=0)
            prototype[i] = torch.mean(support_all[i], dim=0)

        inter_class_diversity_loss = 0
        intra_class_diversity_loss = 0
        generate_all = torch.unsqueeze(generate_all, 2)  # (5, 65, 1, 64)
        prototype = torch.unsqueeze(prototype, 1)  # (5,1,64)
        for i in range(way):
            for j in range(i + 1, way):
                inter_class_diversity_loss += F.pairwise_distance(prototype[i], prototype[j], 2)
            for k in range(self.N_generate):
                intra_class_diversity_loss += F.pairwise_distance(generate_all[i][k], prototype[i], 2)  # 类内
        prototype = torch.squeeze(prototype)  # (5,64)
        loss_diversity = intra_class_diversity_loss / inter_class_diversity_loss
        return prototype, loss_diversity, support_all, torch.squeeze(generate_all, dim=2)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        norm = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(norm)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output


class Encoder1(nn.Module):
    def __init__(self):
        """cub200"""
        super(Encoder1, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.leakyRelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.leakyRelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout2 = nn.Dropout(0.5)


    def forward(self, reference):
        r_pair = Task_concate(reference)
        r_way, r_shot, _ = r_pair.size()
        r_pair = r_pair.view(r_way * r_shot, -1)

        # 对reference编码
        x = self.fc1(r_pair)
        x = self.leakyRelu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.leakyRelu2(x)
        x = self.dropout2(x)
        x = x.view(r_way, r_shot, -1)

        x = torch.mean(x, [0, 1])  # (128)

        return x



class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.fc = nn.Linear(512, 256)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.fc(x)
        x = self.leakyRelu(x)
        x = self.dropout(x)

        return x


class Decoder(nn.Module):
    def __init__(self, N_generate):
        super(Decoder, self).__init__()
        self.N_generate = N_generate
        self.fc = nn.Linear(128,
                            self.N_generate + 1)  # reweighting系数是通过decoder得到的，也就意味着生成样本的个数训练和测试要一致。也就是说测试的时候不能任意个数采样了。
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc(x)
        x = self.leakyRelu(x)
        x = self.dropout(x)
        x = F.softmax(x, dim=1)  # (5,64)

        return x



class Sampler1(nn.Module):
    def __init__(self, N_generate):
        super(Sampler1, self).__init__()
        self.N_generate = N_generate

    def forward(self, delta):
        z_dim = 512

        c_mu = delta[:z_dim]
        c_log_var = delta[z_dim:]
        z_signal = torch.randn(self.N_generate, z_dim).cuda()
        z_c = c_mu + torch.exp(c_log_var / 2) * z_signal

        return z_c

class Sampler2(nn.Module):
    def __init__(self):
        super(Sampler2, self).__init__()

    def forward(self, x):
        z_dim = 128
        c_mu = x[:, :z_dim]
        c_log_var = x[:, z_dim:]
        z_signal = torch.randn(5, z_dim).cuda()
        z_c = c_mu + torch.exp(c_log_var / 2) * z_signal

        return z_c


if __name__ == '__main__':
    support_set = torch.rand(5, 20, 512).cuda()
    reference = torch.rand(90, 2, 512).cuda()
    generatorNet = GeneratorNet(16).cuda()
    proto, loss, support_all, generate_all = generatorNet(support_set, reference)










