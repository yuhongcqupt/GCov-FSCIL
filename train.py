import argparse
import importlib

import sys

from utils import *


MODEL_DIR=''
DATA_DIR = ''

PROJECT='GCov-FSLIC'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[20,60])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.4)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    #for gCov-fscil
    parser.add_argument('-balance', type=float, default=0.01)
    parser.add_argument('-loss_iter', type=int, default=0)
    parser.add_argument('-alpha', type=float, default=2.0)
    parser.add_argument('-eta', type=float, default=0.1)

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='5')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')

    # gen
    parser.add_argument('--gen_train_epoch', default=10, type=int, metavar='N')
    parser.add_argument('--lr_gen', '--learning-rate', default=0.05, type=float,metavar='LR')
    parser.add_argument('--momentum_gen', default=0.9, type=float, metavar='M')
    parser.add_argument('--weight_decay_gen', '--wd', default=5e-4, type=float,metavar='W')
    parser.add_argument('--N_generate', default=16, type=int, metavar='M')
    parser.add_argument('--N_reference_per_class', default=2, type=int,metavar='R')

    return parser


def add_commond_line_parser(params):
    project = params[2]
    # base parser
    parser = get_command_line_parser()

    if project == 'base':
        args = parser.parse_args(params[2:])
        return args

    elif project == 'GCov-FSCIL':
        parser.add_argument('-softmax_t', type=float, default=16)
        parser.add_argument('-shift_weight', type=float, default=0.5, help='weights of delta prototypes')
        parser.add_argument('-soft_mode', type=str, default='soft_proto',
                            choices=['soft_proto', 'soft_embed', 'hard_proto'])
        args = parser.parse_args(params[1:])
        return args
    else:
        raise NotImplementedError


# python脚本入口
if __name__ == '__main__':
    args = add_commond_line_parser(sys.argv)

    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()