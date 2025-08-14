import torch
import torch.nn.functional as F

from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from .data.sampler import RandomIdentitySampler, MultiDomainRandomIdentitySampler

import collections
import numpy as np
import copy

def extract_features(model, data_loader):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    model.eval()
    with torch.no_grad():
        for i, (imgs_o, imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            features = model(imgs)
            for fname, feature, pid, cid in zip(fnames, features, pids, cids):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
    model.train()
    return features_all, labels_all, fnames_all, camids_all


def initial_classifier(model, data_loader):
    pid2features = collections.defaultdict(list)
    features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)
    for feature, pid in zip(features_all, labels_all):
        pid2features[pid].append(feature)
    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = torch.stack(class_centers)
    return F.normalize(class_centers, dim=1).float().cuda()

def obtain_voronoi_loader(dataset,new_labels, add_num=0, batch_size = 32,num_instance=4,workers=8):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    voronoi_set=[]

    if not isinstance(new_labels, list):
        new_labels = new_labels.tolist()

    for instance, lablel in zip(dataset.train, new_labels):    
        a=(instance[0], lablel, instance[2], instance[3])
        voronoi_set.append(a
            
        )     

    voronoi_loader = DataLoader(Preprocessor(voronoi_set, dataset.images_dir, train_transformer),
                             batch_size=batch_size,num_workers=workers, sampler=RandomIdentitySampler(voronoi_set, num_instance),
                             pin_memory=True, drop_last=True)
    # new_labels.sort()

    return voronoi_loader, voronoi_set
def extract_features_uncertain(model, data_loader, get_mean_feature=False):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    var_all=[]
    model.train()
    with torch.no_grad():
        for i, (o_imgs, imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            mean_feat, merge_feat, cls_outputs, out_var,_ = model(imgs)
            for fname, feature, pid, cid, var in zip(fnames, mean_feat, pids, cids,out_var):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
                var_all.append(var)

    # features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)

    if get_mean_feature:
        features_collect = {}
        var_collect = {}

        for feature, label, var in zip(features_all, labels_all,var_all):
            if label in features_collect:
                features_collect[label].append(feature)
                var_collect[label].append(var)
            else:
                features_collect[label] = [feature]
                var_collect[label]=[var]
        labels_named = list(set(labels_all))  # obtain valid features
        labels_named.sort()
        features_mean=[]
        vars_mean=[]
        for x in labels_named:
            if x in features_collect.keys():
                feats=torch.stack(features_collect[x])
                feat_mean=feats.mean(dim=0)
                features_mean.append(feat_mean)

                vars_2=(torch.stack(var_collect[x])**2).mean(dim=0)+(feats**2).mean(dim=0)-feat_mean**2
                vars_mean.append(torch.sqrt(vars_2))
            else:
                features_mean.append(torch.zeros_like(features_all[0]))
                vars_mean.append(torch.zeros_like(var_all[0]))
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named,torch.stack(vars_mean),var_all
    else:
        return features_all, labels_all, fnames_all, camids_all
    
def extract_features_prompter(args, model, data_loader, get_mean_feature=False, prompter=None):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    var_all=[]
    model.train()
   
    with torch.no_grad():
        for i, (o_imgs, imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            mean_feat, merge_feat, cls_outputs, out_var,_ = model(imgs)

            for fname, feature, pid, cid, var in zip(fnames, mean_feat, pids, cids,out_var):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
                var_all.append(var)

            if prompter is not None:
                prompter.train()

                with torch.no_grad():
                    kernel = prompter(o_imgs)
                r_imgs = decode_transfer_img(args, o_imgs, kernel)
                r_mean_feat, r_merge_feat, r_cls_outputs, r_out_var,_ = model(r_imgs)
                
                for fname, r_feature, pid, cid, r_var in zip(fnames, r_mean_feat, pids, cids, r_out_var):
                    features_all.append(r_feature)
                    labels_all.append(int(pid))
                    fnames_all.append(fname)
                    camids_all.append(cid)
                    var_all.append(r_var)
            


    # features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)

    if get_mean_feature:
        features_collect = {}
        var_collect = {}

        for feature, label, var in zip(features_all, labels_all,var_all):
            if label in features_collect:
                features_collect[label].append(feature)
                var_collect[label].append(var)
            else:
                features_collect[label] = [feature]
                var_collect[label]=[var]
        labels_named = list(set(labels_all))  # obtain valid features
        labels_named.sort()
        features_mean=[]
        vars_mean=[]
        for x in labels_named:
            if x in features_collect.keys():
                feats=torch.stack(features_collect[x])
                feat_mean=feats.mean(dim=0)
                features_mean.append(feat_mean)

                vars_2=(torch.stack(var_collect[x])**2).mean(dim=0)+(feats**2).mean(dim=0)-feat_mean**2
                vars_mean.append(torch.sqrt(vars_2))
            else:
                features_mean.append(torch.zeros_like(features_all[0]))
                vars_mean.append(torch.zeros_like(var_all[0]))
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named,torch.stack(vars_mean),var_all
    else:
        return features_all, labels_all, fnames_all, camids_all

def decode_transfer_img(args,imgs,kernels):
    BS=imgs.size(0)
    for i in range(args.n_kernel):
        if args.groups==1:
            offset=3*3*3*3+3
            k_w=kernels[:,offset*i:offset*(i+1)-3]
            k_w=k_w.reshape(BS,3,3,3,3)
            k_b=kernels[:,offset*(i+1)-3:offset*(i+1)]
            imgs=torch.cat([F.conv2d(img.unsqueeze(0), weight=w.cpu(), bias=b.cpu(), stride=1, padding=1) for w,b,img in zip(k_w,k_b,imgs)])
        else:
            offset=3*3*3+3
            k_w=kernels[:,offset*i:offset*(i+1)-3]
            k_w=k_w.reshape(BS,3,1,3,3)
            k_b=kernels[:,offset*(i+1)-3:offset*(i+1)]
            imgs=torch.cat([F.conv2d(img.unsqueeze(0), weight=w.cpu(), bias=b.cpu(), stride=1, padding=1,groups=args.groups) for w,b,img in zip(k_w,k_b,imgs)])

    return imgs


def extract_features_voro(model, data_loader, get_mean_feature=False):    

    features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)

    if get_mean_feature:
        features_collect = {}        

        for feature, label in zip(features_all, labels_all):
            if label in features_collect:
                features_collect[label].append(feature)               
            else:
                features_collect[label] = [feature]                
        labels_named = list(set(labels_all))  # obtain valid features
        labels_named.sort()
        features_mean=[]        
        for x in labels_named:
            if x in features_collect.keys():
                features_mean.append(torch.stack(features_collect[x]).mean(dim=0))                
            else:
                features_mean.append(torch.zeros_like(features_all[0]))
                
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named
    else:
        return features_all, labels_all, fnames_all, camids_all



def select_replay_samples(model, dataset, training_phase=0, add_num=0, old_datas=None, select_samples=2,batch_size = 32,workers=8):
    replay_data = []
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_loader = DataLoader(Preprocessor(dataset.train, root=dataset.images_dir,transform=transformer),
                              batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True, drop_last=False)

    features_all, labels_all, fnames_all, camids_all = extract_features(model, train_loader)

    pid2features = collections.defaultdict(list)
    pid2fnames = collections.defaultdict(list)
    pid2cids = collections.defaultdict(list)

    for feature, pid, fname, cid in zip(features_all, labels_all, fnames_all, camids_all):
        pid2features[pid].append(feature)
        pid2fnames[pid].append(fname)
        pid2cids[pid].append(cid)

    labels_all = list(set(labels_all))

    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = F.normalize(torch.stack(class_centers), dim=1)
    select_pids = np.random.choice(labels_all, 250, replace=False)
    for pid in select_pids:
        feautures_single_pid = F.normalize(torch.stack(pid2features[pid]), dim=1, p=2)
        center_single_pid = class_centers[pid]
        simi = torch.mm(feautures_single_pid, center_single_pid.unsqueeze(0).t())
        simi_sort_inx = torch.sort(simi, dim=0)[1][:2]
        for id in simi_sort_inx:
            replay_data.append((pid2fnames[pid][id], pid+add_num, pid2cids[pid][id], training_phase-1))

    if old_datas is None:
        data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
                             batch_size=batch_size,num_workers=workers, sampler=RandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True)
    else:
        replay_data.extend(old_datas)
        data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
                             batch_size=training_phase*batch_size,num_workers=workers,
                             sampler=MultiDomainRandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True)

    return data_loader_replay, replay_data


def get_pseudo_features(data_specific_batch_norm, training_phase, x, domain, unchange=False):
    fake_feat_list = []
    if unchange is False:
        for i in range(training_phase):
            if int(domain[0]) == i:
                data_specific_batch_norm[i].train()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
            else:
                data_specific_batch_norm[i].eval()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
                data_specific_batch_norm[i].train()
    else:
        for i in range(training_phase):
            data_specific_batch_norm[i].eval()
            fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])

    return fake_feat_list
