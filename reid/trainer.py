from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn

from reid.loss.loss_uncertrainty import TripletLoss_set, ContrastiveLoss
from .utils.meters import AverageMeter
from reid.metric_learning.distance import cosine_similarity
from reid.utils.data import transforms as T
from reid.utils.color_transformer import ColorTransformer

import random
import os
import cv2

def remap(inputs_r,imgs_origin, training_phase, save_dir,dataset_name):
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    x=inputs_r.detach()
    x=x.cpu()
    # print(x.shape)
    x=(x)*std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    # print(x.min(),x.max())

    x=(x*255).clamp(min=0,max=255)
    x=x.permute(0,2,3,1)
    x=x.numpy().astype('uint8')
    vis_dir=save_dir+f'/vis/{dataset_name}/'+str(training_phase)+'/'
    os.makedirs(vis_dir, exist_ok=True)


    imgs_origin=imgs_origin.permute(0,2,3,1)
    imgs_origin=(imgs_origin*255).cpu().numpy().astype('uint8')
    

    for i in range(len(x)):
        cv2.imwrite(vis_dir+f'{i}_reconstruct.png',x[i][:,:,::-1])
        cv2.imwrite(vis_dir+f'{i}_origin.png',imgs_origin[i][:,:,::-1])
    print("saved images in ", vis_dir)

class lreidTrainer(object):
    def __init__(self,args, model, writer=None):
        super(lreidTrainer, self).__init__()
        self.args = args
        self.model = model
        self.writer = writer
        self.uncertainty=True
        self.trans = args.trans

        if self.uncertainty:
            self.criterion_ce=nn.CrossEntropyLoss()
            self.criterion_triple=TripletLoss_set()

        if self.trans:
            self.color_transformer = ColorTransformer()
            self.train_transformer = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        self.AF_weight=args.AF_weight   # anti-forgetting loss
             
        self.n_sampling=args.n_sampling        
       
    def train(self, epoch, data_loader_train,  optimizer, training_phase,
              train_iters=200, add_num=0, proto_type=None, prompter=None, dataset_name=None, model_old=None        
              ):       
        self.model.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()

            
        if proto_type is not None:
            proto_type_merge={}
            steps=list(proto_type.keys())
            steps.sort()
            stages=1
            if stages<len(steps):
                steps=steps[-stages:]

            proto_type_merge['mean_features']=torch.cat([proto_type[k]['mean_features'] for k in steps])
            proto_type_merge['labels'] = torch.tensor([proto_type[k]['labels'] for k in steps]).to(proto_type_merge['mean_features'].device)
            proto_type_merge['mean_vars'] = torch.cat([proto_type[k]['mean_vars'] for k in steps])
           
            features_mean=proto_type_merge['mean_features']

        batch_time = AverageMeter()
        data_time = AverageMeter()
        s_losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        r_losses_ce = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs_o , s_inputs, targets, cids, domains = self._parse_data(train_inputs)
            targets += add_num
            # BS*2048, BS*(1+n_sampling)*2048,BS*(1+n_sampling)*num_classes, BS*2048
            s_features, merge_feat, cls_outputs, out_var,feat_final_layer = self.model(s_inputs)

            s_loss_ce, loss_tp = 0, 0
            r_loss_ce =  0

            if prompter is not None:

                prompter.train()
                with torch.no_grad():
                    kernel = prompter(s_inputs_o)   
                
                r_inputs=self.decode_transfer_img(s_inputs_o,kernel)
                
                r_features, r_merge_feat, r_cls_outputs, r_out_var, r_feat_final_layer = self.model(r_inputs)
                if self.n_sampling<1:
                    r_cls_outputs=torch.cat((r_cls_outputs,r_cls_outputs),dim=1)
                    r_merge_feat=torch.cat((r_merge_feat,r_merge_feat),dim=1)
                    cls_outputs=torch.cat((cls_outputs,cls_outputs),dim=1)
                    merge_feat=torch.cat((merge_feat,merge_feat),dim=1)

                for r_id in range(1 + self.n_sampling):
                    r_loss_ce += self.criterion_ce(r_cls_outputs[:, r_id], targets)
                r_loss_ce = r_loss_ce / (1 + self.n_sampling)
                r_loss_ce=r_loss_ce*1

                for s_id in range(1 + self.n_sampling):
                    s_loss_ce += self.criterion_ce(cls_outputs[:, s_id], targets)
                s_loss_ce = s_loss_ce / (1 + self.n_sampling)
                s_loss_ce = s_loss_ce*1
                loss = s_loss_ce + r_loss_ce 
                # print(self.args.triplet_loss)
                if self.args.triplet_loss:
                    loss_tp = self.criterion_triple(torch.cat((r_merge_feat,merge_feat), dim=1), targets)[0]

                    loss_s_tp = self.criterion_triple(merge_feat, targets)[0]
                    loss_r_tp = self.criterion_triple(r_merge_feat, targets)[0]

                    loss = loss + loss_tp * self.args.merge_tri_weight + (loss_s_tp + loss_r_tp) * self.args.s_r_tri_weight

            else:
                if self.n_sampling<1:
                    cls_outputs=torch.cat((cls_outputs,cls_outputs),dim=1)
                    merge_feat=torch.cat((merge_feat,merge_feat),dim=1)
                ###ID loss###
                for s_id in range(1 + self.n_sampling):
                    s_loss_ce += self.criterion_ce(cls_outputs[:, s_id], targets)
                s_loss_ce = s_loss_ce / (1 + self.n_sampling)
                s_loss_ce = s_loss_ce*1
                loss = s_loss_ce 
                ###set triplet-loss##
                if self.args.triplet_loss:
                    loss_tp = self.criterion_triple(merge_feat, targets)[0]
                    loss_tp = loss_tp*1.5    
                    loss = loss + loss_tp   


            s_losses_ce.update(s_loss_ce.item())
            losses_tr.update(loss_tp.item())
            r_losses_ce.update(r_loss_ce)

            if model_old is not None:
                s_divergence, r_divergence = 0, 0
                model_old.eval()
                with torch.no_grad():
                    # old_s_features = model_old(s_inputs)
                    old_r_features = model_old(r_inputs)
                
                # s_Affinity_matrix_old = self.get_normal_affinity(old_s_features)
                # s_Affinity_matrix_new = self.get_normal_affinity(s_features)
                r_Affinity_matrix_old = self.get_normal_affinity(old_r_features)
                r_Affinity_matrix_new = self.get_normal_affinity(r_features)

                # s_divergence = self.cal_KL(s_Affinity_matrix_old, s_Affinity_matrix_new)
                r_divergence = self.cal_KL(r_Affinity_matrix_old, r_Affinity_matrix_new)
                loss = loss + r_divergence * self.args.distill_weight


            if training_phase>1:
                if self.n_sampling>0:
                    features_var = proto_type_merge['mean_vars']    # obtain the prototype strandard variances
                    noises = torch.randn(features_mean.size()).to(features_mean.device) # generate gaussian noise
                    samples = noises * features_var + features_mean # obtain noised sample
                else:
                    samples=features_mean
                samples = F.normalize(samples, dim=1)   # normalize the sample
               
                s_proto_divergence, r_proto_divergence = 0, 0         

                Affinity_matrix_new_s = self.get_normal_affinity(s_features, self.args.lambda_2) # obtain the affinity matrix

                s_features_old = cosine_similarity(s_features, samples)    # obtain the new-old relation
                s_features_old = F.softmax(s_features_old /self.args.lambda_1, dim=1)  # normalize the relation

                Affinity_matrix_old_s = self.get_normal_affinity(s_features_old, self.args.lambda_2)  # obtain the affinity matrix under the prototype view
                Affinity_matrix_new_log_s = torch.log(Affinity_matrix_new_s)
                s_proto_divergence = self.KLDivLoss(Affinity_matrix_new_log_s, Affinity_matrix_old_s)
        

                loss = loss + s_proto_divergence * self.AF_weight #+ r_proto_divergence * self.AF_weight                             
            
            optimizer.zero_grad()
            loss.backward()       
                     
            optimizer.step()          
           
            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/s_Loss_ce_{}".format(training_phase), scalar_value=s_losses_ce.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val,
                          global_step=epoch * train_iters + i)

                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                          global_step=epoch * train_iters + i)
            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      's_Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tp {:.3f} ({:.3f})\t'   
                      'r_Loss_ce {:.3f} ({:.3f})\t'          
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              s_losses_ce.val, s_losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              r_losses_ce.val, r_losses_ce.avg,
                              ))
                # if prompter is not None:
                #     remap(r_inputs,s_inputs_o, epoch, self.args.logs_dir, dataset_name)

    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix

    def _parse_data(self, inputs):
        imgs_o, imgs, _, pids, cids, domains = inputs
        inputs_o = imgs_o.cuda()
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs_o, inputs, targets, cids, domains

    def decode_transfer_img(self, imgs,kernels,n_kernel=1,groups=1):
        # print(imgs.size(), kernels.size())
        BS=imgs.size(0)
        for i in range(n_kernel):
            if groups==1:
                offset=3*3*3*3+3
                k_w=kernels[:,offset*i:offset*(i+1)-3]
                k_w=k_w.reshape(BS,3,3,3,3)
                k_b=kernels[:,offset*(i+1)-3:offset*(i+1)]
                imgs=torch.cat([F.conv2d(img.unsqueeze(0), weight=w, bias=b, stride=1, padding=1) for w,b,img in zip(k_w,k_b,imgs)])
            else:
                offset=3*3*3+3
                k_w=kernels[:,offset*i:offset*(i+1)-3]
                k_w=k_w.reshape(BS,3,1,3,3)
                k_b=kernels[:,offset*(i+1)-3:offset*(i+1)]
                imgs=torch.cat([F.conv2d(img.unsqueeze(0), weight=w, bias=b, stride=1, padding=1,groups=args.groups) for w,b,img in zip(k_w,k_b,imgs)])

        return imgs

    def cal_KL(self,Affinity_matrix_new, Affinity_matrix_old):
        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Affinity_matrix_old) 
        return divergence 


