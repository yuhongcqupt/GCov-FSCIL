from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable
from .helper import *
from utils import *
from dataloader.data_utils import *
from .Network import MYNET
import os
import generator as model_g

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.model.shrink_cov = self.model.module.shrink_cov
        self.model.cov_mats = self.model.module.cov_mats
        self.model.base_cov_mats = self.model.module.base_cov_mats
        self.model.eval_task = self.model.module.eval_task
        self.model.count_acc_inc_detach = self.model.module.count_acc_inc_detach

        self.model_G = model_g.GeneratorNet(N_generate=args.N_generate).cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']

        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        result_list = [args]
        masknum = 3
        mask = np.zeros((args.base_class, args.num_classes))
        for i in range(args.num_classes - args.base_class):
            picked_dummy = np.random.choice(args.base_class, masknum, replace=False)
        mask = torch.tensor(mask).cuda()

        new_key = 'module.fc_old.weight'
        new_value = self.best_model_dict['module.fc.weight']
        self.best_model_dict[new_key] = new_value

        dummy_orthogonal_classifier = nn.Linear(512, 100, bias=False)
        new_key = 'module.dummy_orthogonal_classifier.weight'
        new_value = dummy_orthogonal_classifier.weight
        self.best_model_dict[new_key] = new_value

        gen_trainloader, gen_testloader= get_base_dataloader_gen(self.args)

        gen_support_set, gen_support_loader = get_support_dataloader_gen(self.args)
        gen_support_input, gen_support_target = gen_support_loader.__iter__().next()

        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.load_state_dict(self.best_model_dict)


            if session == 0:
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                optimizer_gen = torch.optim.SGD(self.model_G.parameters(), args.lr_gen,
                                                momentum=0.9, weight_decay=5e-4, nesterov=True)
                scheduler_gen = torch.optim.lr_scheduler.MultiStepLR(optimizer_gen,
                                                                     milestones=[10, 20, 40 , 50 , 60 , 80,
                                                                                 100, 120, 140], gamma=0.4)
                criterion = nn.CrossEntropyLoss().cuda()
                gen_epoch = 1
                TRAIN_PHASE = ['a'] * args.epochs_base + ['m'] * gen_epoch

                best_acc = 0
                best_epoch = 0
                for epoch in range(args.epochs_base + gen_epoch):
                    train_phase = TRAIN_PHASE[epoch]
                    start_time = time.time()
                    if train_phase != 'm':
                        print('phase: base_train...')
                        tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args, mask)
                    else:
                        print('phase: meta_train...')
                        lr_gen = scheduler_gen.get_last_lr()[0]
                        print('\nEpoch: [%d | %d] LR: %f' % (epoch, gen_epoch, lr_gen))
                        train_loss, train_acc = gen_train(gen_support_input, gen_trainloader, self.model, self.model_G, criterion, optimizer_gen)
                        print('current epoch gen_train acc: {:.3f}, loss={:.3f}'.format(train_acc, train_loss))
                        test_loss, test_acc, _ = validate(gen_support_input, gen_testloader, self.model, self.model_G, criterion)
                        print('current epoch gen_test acc: {:.3f}, loss={:.3f}'.format(test_acc, test_loss))
                        scheduler_gen.step()
                        if test_acc > best_acc:
                            best_acc = test_acc
                            best_epoch = epoch
                            save_checkpoint({
                                'epoch': epoch,
                                'state_dict_G': self.model_G.state_dict(),
                                'optimizer': optimizer_gen.state_dict(),
                                'scheduler': scheduler_gen.state_dict()
                            }, epoch, checkpoint='checkpoint_gen')
                        print('best_test_acc_gen:', best_acc)
                        print('best_test_epoch_gen:', best_epoch)

                    # save better model
                    if train_phase != 'm':
                        tsl, tsa = test(self.model, testloader, epoch, args, session)
                        # save better model
                        if (tsa * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            print('********A better model is found!!**********')
                            print('Saving model to :%s' % save_model_dir)

                        print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                           self.trlog['max_acc'][session]))

                        self.trlog['train_loss'].append(tl)
                        self.trlog['train_acc'].append(ta)
                        self.trlog['test_loss'].append(tsl)
                        self.trlog['test_acc'].append(tsa)
                        lrc = scheduler.get_last_lr()[0]
                        result_list.append(
                            'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                        print('This epoch takes %d seconds' % (time.time() - start_time),
                              '\nstill need around %.2f mins to finish this session' % (
                                      (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                        scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model, radius = replace_base_fc(self, train_set, testloader.dataset.transform, self.model,
                                                         args)
                    print('yes！！update the fc with proto to base session!!\n')

                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)
                    self.model.module.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                self.dummy_classifiers = deepcopy(
                    self.model.module.fc.weight.detach())

                self.dummy_classifiers = F.normalize(self.dummy_classifiers[self.args.base_class:, :], p=2,
                                                     dim=-1)
                self.old_classifiers = self.dummy_classifiers[:self.args.base_class, :]  # ([40, 64])
            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                if args.soft_mode == 'soft_proto':
                    new_protos = self.model.module.update_fc(trainloader, np.unique(train_set.targets), session, radius,
                                                             self.model_G, gen_support_input)
                else:
                    raise NotImplementedError
                tsl, (seenac, unseenac, avgac) \
                    = self.test_intergrate(self.model, testloader, 0, args, session, result_list=result_list)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (avgac * 100))
                self.trlog['seen_acc'].append(float('%.3f' % (seenac * 100)))
                self.trlog['unseen_acc'].append(float('%.3f' % (unseenac * 100)))

                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                self.best_model_dict = deepcopy(self.model.state_dict())

                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                print('Session :[%d]' % session, ' ==> Base Acc:', self.trlog['seen_acc'][-1],
                      'Inc Acc:', self.trlog['unseen_acc'][-1],
                      'Avg Acc:', self.trlog['max_acc'][session]
                      )
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def test_intergrate(self, model, testloader, epoch, args, session, result_list=None):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        va2 = Averager()
        Base_total = 0
        Base_top2 = 0
        Base_t1 = 0
        Inc_total = 0
        Inc_top2 = 0
        Inc_t1 = 0

        Base_cov = 0
        Inc_cov = 0

        va_cov = Averager()
        va_teen = Averager()

        lgt = torch.tensor([])
        lbs = torch.tensor([])

        proj_matrix = torch.mm(self.dummy_classifiers,
                               F.normalize(torch.transpose(model.module.fc.weight[:test_class, :], 1, 0), p=2, dim=-1))

        eta = args.eta

        softmaxed_proj_matrix = F.softmax(proj_matrix, dim=1)
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                emb = model.module.encode(data)
                proj = torch.mm(F.normalize(emb, p=2, dim=-1), torch.transpose(self.dummy_classifiers, 1, 0))
                topk, indices = torch.topk(proj, 40)
                res = (torch.zeros_like(proj))
                res_logit = res.scatter(1, indices, topk)
                logits1 = torch.mm(res_logit, proj_matrix)
                logits2_teen = model.module.forpass_fc(data, multi=False)[:, :test_class]
                logits = eta * F.softmax(logits1, dim=1) + (1 - eta) * F.softmax(logits2_teen, dim=1)

                pre_teen = torch.argmax(logits, dim=1)
                acc_teen = count_acc_inc(pre_teen, test_label)
                loss_teen = F.cross_entropy(logits, test_label)
                top2acc_teen = count_acc_topk(logits, test_label)

                """协方差矩阵利用马氏距离"""
                pre_1, nme_accy = model.module.eval_task(emb, test_label)
                pre_cov = torch.tensor(pre_1.reshape(-1)).clone().detach().cuda()
                acc_cov = count_acc_inc(pre_cov, test_label)
                """--------------------------------------"""

                vl.add(loss_teen.item())
                va.add(acc_teen)

                va2.add(top2acc_teen)

                va_cov.add(acc_cov)
                va_teen.add(acc_teen)

                lgt = torch.cat([lgt, logits.cpu()])
                lbs = torch.cat([lbs, test_label.cpu()])

                base_top2, base_total = count_base_top2ac(args, logits, test_label)
                Base_total += base_total
                Base_top2 += base_top2

                inc_top2, inc_total = count_inc_top2ac(args, logits, test_label, session)
                Inc_total += inc_total
                Inc_top2 += inc_top2

                base_t1, inc_t1 = count_base_inc_ac(args, test_label, pre_teen)
                Base_t1 += base_t1
                Inc_t1 += inc_t1

                base_cov, inc_cov = count_base_inc(args, test_label, pre_cov)
                Base_cov += base_cov
                Inc_cov += inc_cov

            vl = vl.item()
            va = va.item()
            va_cov = va_cov.item()
            va_teen = va_teen.item()
            va2 = va2.item()
            print(
                'epo {}, test, loss_proto+gen={:.4f} acc_proto={:.4f}, acc@2_proto={:.4f}'.format(
                    epoch, vl, va, va2))

            print('acc_teen={:.4f}'.format(va_teen * 100))

        if session > 0:
            base_teen = float(Base_t1 / Base_total)
            inc_teen = float(Inc_t1 / Inc_total)
            base_cov = float(Base_cov / Base_total)
            inc_cov = float(Inc_cov / Inc_total)
            print('acc_cov={:.4f}, base_cov={:.4f}, inc_cov={:.4f}'.format(va_cov * 100, base_cov * 100, inc_cov * 100))
            print('acc_teen={:.4f}, base_teen={:.4f}, inc_teen={:.4f}'.format(va_teen * 100, base_teen * 100, inc_teen * 100))

            result_list.append(
                f"Base Acc:{base_teen}  Inc Acc:{inc_teen}")
            return vl, (base_cov, inc_cov, va)

        else:
            return vl, va

    def set_save_path(self):

        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
            self.args.save_path = self.args.save_path + 'Bal%.2f-LossIter%d' % (
                self.args.balance, self.args.loss_iter)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)
            self.args.save_path = self.args.save_path + 'Bal%.2f-LossIter%d' % (
                self.args.balance, self.args.loss_iter)

        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None





