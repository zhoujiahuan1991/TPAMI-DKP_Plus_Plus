from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys

from torch.backends import cudnn
import torch.nn as nn
import random
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.trainer_aux import RehearserTrainer
import numpy as np
import torch

from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res
from reid.models.rehearser import KernelLearning
import datetime
def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content
def main():
    args = parser.parse_args()

    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
  
    main_worker(args)


def main_worker(args):
    timestamp = cur_timestamp_str()
    log_name = f'log_{timestamp}.txt'
    if True:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    
    print("==========\nArgs:{}\n==========".format(args))

    training_set = ['market1501', 'cuhk_sysu', 'dukemtmc', 'msmt17', 'cuhk03']    
    # all the revelent datasets
    all_set = ['market1501', 'dukemtmc', 'msmt17', 'cuhk_sysu', 'cuhk03']  # 'sense','prid'
    # the datsets only used for testing
    testing_only_set = [x for x in all_set if x not in training_set]
    # get the loders of different datasets
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    
    print("data prepared!!")

    # train on the datasets squentially
    for set_index in range(0, len(training_set)):                                         
        print("creating model!!")
                
        print("using mobilenet-v3!!!!")
        rehearser=KernelLearning(n_kernel=args.n_kernel, groups=args.groups, model='mobile-v3').cuda()
                       
        print("generating trainer!!")
        rehearser_trainer = RehearserTrainer(args=args,rehearser=rehearser)
        
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[
                set_index]  # status of current dataset  
        print("start training on ", name) 

        for epoch in range(50):
            rehearser_trainer.train(epoch, train_loader, train_iters=400,dataset_name=name)
            # if (epoch+1)%10 ==0:
            #     save_checkpoint({
            #     'state_dict': rehearser.state_dict(),
            #     'epoch': epoch + 1
            # }, True, fpath=osp.join(args.logs_dir, '{}_transfer_{}.pth.tar'.format(name,epoch)))  

        save_checkpoint({
                'state_dict': rehearser.state_dict(),
                'epoch': epoch + 1
            }, True, fpath=osp.join(args.logs_dir, '{}_transfer.pth.tar'.format(name)))    






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=200)
    
    # path   
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/xukunlun/DATA/PRID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('../logs/try'))
   
 
    parser.add_argument('--color_style', type=str,default='rgb', help="data augmentation strategy", choices=['lab','rgb'])

    parser.add_argument('--n_kernel', default=1, type=int, help="number of Distribution Transfer kernel")   
    parser.add_argument('--groups', default=1, type=int, help="convolution group number of each Distribution Transfer kernel")  

    main()

