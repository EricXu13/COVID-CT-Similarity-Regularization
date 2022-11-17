import numpy as np

import torch.nn as nn
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

from .covid_dataset import COVIDxDataset, ExternalDataset
from .samplers import SubsetRandomSampler
from .utils import TwoCropTransform


def build_loader(config):
    # Build datasets for train, val, test
    train_set = build_dataset(split='train', config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    val_set = build_dataset(split='val', config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")
    test_set = build_dataset(split='test', config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build test dataset")
    
    # Build samplers for DDP
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    train_sampler = DistributedSampler(
            train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    indices = np.arange(dist.get_rank(), len(val_set), dist.get_world_size())
    val_sampler = SubsetRandomSampler(indices)
    
    indices = np.arange(dist.get_rank(), len(test_set), dist.get_world_size())
    test_sampler = SubsetRandomSampler(indices)
    
    # build DataLoaders for three sets
    train_loader = DataLoader(
        train_set, sampler=train_sampler, batch_size=config.DATA.BATCH_SIZE, \
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(
        val_set, sampler=val_sampler, batch_size=config.DATA.BATCH_SIZE, shuffle=False, \
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=False)
    test_loader = DataLoader(
        test_set, sampler=test_sampler, batch_size=config.DATA.BATCH_SIZE, shuffle=False, \
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=False)
    
    mixup_fn = build_mixup_fn(config)

    return train_loader, val_loader, test_loader, mixup_fn
    

def build_dataset(split, config):
    split_file = split + '_COVIDx_CT-2A.txt'
    transform = build_transform(split == 'train', config=config)
    dataset = COVIDxDataset(config.DATA.DIR, split_file, transform)
    return dataset

def build_external_dataset(config):
    transform = build_transform(train=False, config=config)
    dataset = ExternalDataset(config.DATA.DIR, transform)
    return dataset


def build_transform(train, config):
    if train:
        return build_train_transform_by_level(config)
           
    t = []
    if config.TEST.CROP:
        size = int((256 / 224) * config.DATA.IMG_SIZE)
        t.append(transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)))
        t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
    else:
        t.append(transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                              interpolation=_pil_interp(config.DATA.INTERPOLATION)))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_train_transform_by_level(config):
#     Augmentation Level:
#     0: No augmentation
#     1: + Random Resized Crop, scale=(0.08, 1.0), ratio=(3./4., 4./3.)
#     2: + Horizontal Flipping, p=0.5
#     3: + Randn Augmentation, "rand-m9-mstd0.5-inc1"
#     4: + Random Erasing, p=0.25, mode=pixel, recount=1
#     5: + Cutmix, alpha=1.0
#     6. + Mixup, cutmix switch to mixup with prob=0.5, mixup alpha = 0.8
    level = config.AUG.LEVEL
    if level == 0:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            no_aug=True,
            auto_augment=None,
            interpolation=config.DATA.INTERPOLATION,)
    elif level == 1:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            hflip=0.,
            color_jitter=None,
            interpolation=config.DATA.INTERPOLATION,)
    elif level == 2:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            hflip=0.5,
            color_jitter=None,
            interpolation=config.DATA.INTERPOLATION,)
    elif level == 3:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            auto_augment=config.AUG.AUTO_AUGMENT,
            interpolation=config.DATA.INTERPOLATION,)
    else:
        # level >= 4
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            auto_augment=config.AUG.AUTO_AUGMENT,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,)
        
    if level > 6 or level < 0:
        raise NotImplementedError("Unknown data augmentation level, valid in range [0, 6]")
        
    if config.TRAIN.TWOCROP:
        transform = TwoCropTransform(transform)
        
    return transform


def build_mixup_fn(config):
    # setup mixup / cutmix
    mixup_fn = None
    if config.AUG.LEVEL > 4: 
        cutmix_enable = config.AUG.LEVEL == 6
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, \
            cutmix_minmax=config.AUG.CUTMIX_MINMAX, prob=config.AUG.MIXUP_PROB, \
            switch_prob=config.AUG.MIXUP_SWITCH_PROB if cutmix_enable else 0., \
            mode=config.AUG.MIXUP_MODE, \
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    return mixup_fn
        
    

    

