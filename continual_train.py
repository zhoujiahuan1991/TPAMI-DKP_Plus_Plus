from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import os

from torch.backends import cudnn
import torch.nn as nn
import random
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet_uncertainty import ResNetSimCLR
from reid.trainer import lreidTrainer
from torch.utils.tensorboard import SummaryWriter
from reid.models.rehearser import KernelLearning


from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res
from reid.evaluation.fast_test import fast_test_p_s
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

    torch.set_num_threads(1)

    
    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)

    main_worker(args)


def main_worker(args):
    # log_name = 'log.txt'
    timestamp = cur_timestamp_str()
    log_name = f'log_{timestamp}.txt'
    sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name=f'log_res_{timestamp}.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))    # record the test results
    

    """
    loading the datasets:
    """
    if args.setting == 1:
        training_set = ['market1501', 'cuhk_sysu', 'dukemtmc', 'msmt17', 'cuhk03']
    elif args.setting == 2:
        training_set = ['dukemtmc', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03']
    elif args.setting == 51:
        training_set = ['msmt17', 'cuhk_sysu', 'dukemtmc', 'market1501', 'cuhk03']
    elif args.setting == 52:
        training_set = ['dukemtmc', 'market1501', 'cuhk03', 'msmt17', 'cuhk_sysu']
    elif args.setting == 53:
        training_set = ['cuhk_sysu', 'dukemtmc', 'cuhk03', 'msmt17', 'market1501']
    elif args.setting == 54:
        training_set = ['cuhk03', 'msmt17', 'dukemtmc', 'market1501', 'cuhk_sysu']
    elif args.setting == 55:
        training_set = ['market1501', 'msmt17', 'dukemtmc', 'cuhk_sysu', 'cuhk03']
    # all the revelent datasets
    all_set = ['market1501', 'dukemtmc', 'msmt17', 'cuhk_sysu', 'cuhk03',
               'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']  # 'sense','prid'
    # the datsets only used for testing
    testing_only_set = [x for x in all_set if x not in training_set]
    # get the loders of different datasets
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    
    first_train_set = all_train_sets[0]
    
    model=ResNetSimCLR(num_classes=first_train_set[1], uncertainty=args.uncertainty, n_sampling=args.n_sampling)
    model.load_from_pretrain(args) 
    
    model.cuda()
    model = DataParallel(model)    

    writer = SummaryWriter(log_dir=args.logs_dir)
    # Load from checkpoint
    '''test the models under a folder'''

    #------------------------------------
    #
    if args.test_folder:
        ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]   # obatin pretrained model name
        checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[0]))  # load the first model
        copy_state_dict(checkpoint['state_dict'], model)     #    
        for step in range(len(ckpt_name) - 1):
            features_all_old, labels_all_old,features_mean, labels_named,vars_mean,vars_all=obtain_old_types( all_train_sets,step+1,model)
            proto_type={}
            proto_type[step]={
                    "features_all_old":features_all_old,
                    "labels_all_old":labels_all_old,
                    'mean_features':features_mean,
                    'labels':labels_named,
                    'mean_vars':vars_mean,
                    "vars_all":vars_all
                }
            torch.save(proto_type,args.test_folder+'/proto-stage{}.pt'.format(step+1))


            model_old = copy.deepcopy(model)    # backup the old model            
            checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[step + 1]))
            copy_state_dict(checkpoint['state_dict'], model)                         
           
            model = linear_combination(args, model, model_old, args.fix_EMA)

            save_name = '{}_checkpoint_adaptive_ema.pth.tar'.format(training_set[step+1])
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
            }, True, fpath=osp.join(args.logs_dir, save_name))        
        
        fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=len(all_train_sets)-1, logger=logger_res,
                      args=args,writer=writer)
        exit(0)
    #------------------------------------

    # resume from a model
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        if args.evaluate:
            fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=0, logger=logger_res,
                      args=args,writer=writer)
            exit(0)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
   
    # Evaluator
    if args.MODEL in ['50x']:
        out_channel = 2048
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")

       
    # train on the datasets squentially
    for set_index in range(0, len(training_set)):       
        model_old = copy.deepcopy(model)
        if args.trans==True:

            if set_index == 0:
                if args.resume is not None:
                    continue
                else:
                    model = train_dataset(args, all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                                    writer,logger_res=logger_res, prompter = None, model_old=None)    
                    
            elif args.resume_folder and set_index<= args.resume_id:
                fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(all_train_sets[set_index][-1]))
                checkpoint = load_checkpoint(fpath)
                class_num = get_add_num(all_train_sets, set_index+1)
                model.module.classifier = nn.Linear(out_channel, class_num, bias=False)
                model=model.cuda()
                copy_state_dict(checkpoint['state_dict'], model)
            else:                
                prompter=KernelLearning(n_kernel=1).cuda()
                checkpoint = load_checkpoint('transfer_model/{}_transfer.pth.tar'.format(all_train_sets[set_index-1][-1])) 
                                                                                     
                copy_state_dict(checkpoint['state_dict'], prompter)
                prompter = DataParallel(prompter) 

                print("{} {} prompter is starting----------------".format(all_train_sets[set_index-1][-1],model.__class__.__name__))
                model = train_dataset(args, all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                                    writer, logger_res=logger_res, prompter = prompter, model_old=model_old)                
        else:
            if args.resume is not None and set_index == 0:
                continue
            else:
                model = train_dataset(args, all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                            writer, logger_res=logger_res, prompter = None)
        if set_index>0:
            model = linear_combination(args, model, model_old, args.fix_EMA)
              
            fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res,
                      args=args,writer=writer)
    print('finished')

    
def obtain_old_types( all_train_sets, set_index, model):
    dataset_old, num_classes_old, train_loader_old, _, init_loader_old, name_old = all_train_sets[
        set_index - 1]  # trainloader of old dataset
    features_all_old, labels_all_old, fnames_all, camids_all, features_mean, labels_named, vars_mean,vars_all = extract_features_uncertain(model,
                                                                                                                  init_loader_old,
                                                                                                                  get_mean_feature=True)  # init_loader is original designed for classifer init
    features_all_old = torch.stack(features_all_old)
    labels_all_old = torch.tensor(labels_all_old).to(features_all_old.device)
    features_all_old.requires_grad = False
    return features_all_old, labels_all_old, features_mean, labels_named,vars_mean,vars_all

def get_add_num(all_train_sets, set_index):
    if set_index<1:
        add_num = 0
    else:
        add_num = sum([all_train_sets[i][1] for i in range(set_index)])  
    return add_num

def train_dataset(args, all_train_sets, all_test_only_sets, set_index, model, out_channel, writer,logger_res=None, prompter=None, model_old=None):
    if set_index>0:
        features_all_old, labels_all_old,features_mean, labels_named,vars_mean,vars_all=obtain_old_types( all_train_sets,set_index,model)
        proto_type={}
        proto_type[set_index-1]={
                "features_all_old":features_all_old,
                "labels_all_old":labels_all_old,
                'mean_features':features_mean,
                'labels':labels_named,
                'mean_vars':vars_mean,
                "vars_all":vars_all
            }
        torch.save(proto_type,args.logs_dir+'/proto-stage{}.pt'.format(set_index))
        del features_all_old
        del vars_all
    else:
        proto_type=None
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[set_index]  # status of current dataset    

    Epochs= args.epochs0 if 0==set_index else args.epochs          


    
    add_num = get_add_num(all_train_sets, set_index)

    if set_index>0:
        '''store the old model'''
        model_old = model_old.cuda()

        # after sampling rehearsal, recalculate the addnum(historical ID number)
        add_num = sum([all_train_sets[i][1] for i in range(set_index)])  # get model out_dim
        # Expand the dimension of classifier
        org_classifier_params = model.module.classifier.weight.data
        model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)
        model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)
        model.cuda()    
        # Initialize classifer with class centers    
        class_centers = initial_classifier(model, init_loader)
        model.module.classifier.weight.data[add_num:].copy_(class_centers)
        model.cuda()

    # Re-initialize optimizer
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            print('not requires_grad:', key)
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)    
    # Stones=args.milestones
    Stones = [20, 30] if name == 'msmt17' else args.milestones
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
  
    trainer = lreidTrainer(args, model,  writer=writer)

    print('####### LReID model is starting training on {} #######'.format(name))
    for epoch in range(0, Epochs):
        train_loader.new_epoch()
        trainer.train(epoch, train_loader,  optimizer, training_phase=set_index + 1,
                      train_iters=len(train_loader), add_num=add_num,proto_type=proto_type, prompter=prompter, dataset_name= name, model_old=model_old
                      )
 
        lr_scheduler.step()       

   

        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))

            logger_res.append('epoch: {}'.format(epoch + 1))
            
            mAP=0.
            if args.middle_test:              
                fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res,
                      args=args,writer=writer)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))    

    return model 


def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
            # print(k,'+++')
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]
    model_new.load_state_dict(model_new_state_dict)
    return model_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model    
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x'])
    parser.add_argument('--pre_train', type=str, default='imagenet',
                        choices=['imagenet', 'pass','lup','lupws'])
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    
    # path   
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/xukunlun/DATA/PRID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('../logs/try'))

      
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
   
    parser.add_argument('--setting', type=int, default=1, help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.5, type=float, help="anti-forgetting weight")   
    parser.add_argument('--n_sampling', default=6, type=int, help="number of sampling by Gaussian distribution")
    parser.add_argument('--lambda_1', default=0.1, type=float, help="temperature")
    parser.add_argument('--lambda_2', default=0.1, type=float, help="temperature")  

    parser.add_argument('--trans', default=True, type=bool)  
    parser.add_argument('--uncertainty', default=True, type=bool, help="whether to start the DKP")  
       
    parser.add_argument('--fix_EMA', default=0.5, type=float) 
    parser.add_argument('--triplet_loss', default=True, type=bool) 
    parser.add_argument('--resume_folder', type=str, default=None, help="resume the models in a file")
    parser.add_argument('--resume_id', type=int, default=-1, help="resume the models in a file")

    parser.add_argument('--s_r_tri_weight', default=1.5, type=float) 
    parser.add_argument('--merge_tri_weight', default=0.75, type=float) 
    parser.add_argument('--distill_weight', default=0.75, type=float) 

    parser.add_argument('--save_evaluation', action='store_true', help="save ranking results")
    
    main()
