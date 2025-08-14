# import torchvision.transforms as T
import copy
import os.path
import os
from reid.utils.feature_tools import *
import lreid_dataset.datasets as datasets
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data import IterLoader
import numpy as np

def worker_init_fn(worker_id, seed=1):
    np.random.seed(seed + worker_id)


def get_data(name, data_dir, height, width, batch_size, workers, num_instances, select_num=0):
    # root = osp.join(data_dir, name)
    root = data_dir
    
    dataset = datasets.create(name, root)
    '''select some persons for training'''
    if select_num > 0:
        train = []
        for instance in dataset.train:
            if instance[1] < select_num:
                # new_id=id_2_id[instance[1]]
                train.append((instance[0], instance[1], instance[2], instance[3]))

        dataset.train = train
        dataset.num_train_pids = select_num
        dataset.num_train_imgs = len(train)

   

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    # just reize the image
    basic_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
    ])


    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None


    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer, transform_basic=basic_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer, transform_basic=basic_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)
    
    basic_transformer_init = T.Compose([
        T.Resize((height, width), interpolation=3),
        # T.RandomHorizontalFlip(p=0.5),
        # T.Pad(10),
        # T.RandomCrop((height, width)),
        T.ToTensor(),
    ])

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer, transform_basic=basic_transformer_init),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)


    return [dataset, num_classes, train_loader, test_loader, init_loader, name]


def build_data_loaders(cfg, training_set, testing_only_set, toy_num=0):
    # Create data loaders
    data_dir = cfg.data_dir
    height, width = (256, 128)
    training_loaders = [get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                 cfg.num_instances, select_num=500) for name in training_set]

  
    testing_loaders = [get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                cfg.num_instances) for name in testing_only_set]
    return training_loaders, testing_loaders


