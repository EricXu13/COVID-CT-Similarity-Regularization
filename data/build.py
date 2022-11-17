import numpy as np

import torch.nn as nn
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp

from .covid_dataset import COVIDxDataset
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
    
    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = (config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. 
                    or config.AUG.CUTMIX_MINMAX is not None) and (config.AUG.TYPE == 'auto')
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, \
            cutmix_minmax=config.AUG.CUTMIX_MINMAX, prob=config.AUG.MIXUP_PROB, \
            switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE, \
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return train_loader, val_loader, test_loader, mixup_fn
    

def build_dataset(split, config):
    split_file = split + '_COVIDx_CT-2A.txt'
    transform = build_transform(split == 'train', config=config)
    dataset = COVIDxDataset(config.DATA.DIR, split_file, transform)
    return dataset


def build_transform(train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if train:
        return build_train_transform(config, resize_im)
           
    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(transforms.Resize(size, interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_train_transform(config, resize_im):
    if config.AUG.TYPE == 'auto':
        transform = create_transform(
            input_size = config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,)
    elif config.AUG.TYPE == 'none':
        transform = create_transform(
            input_size = config.DATA.IMG_SIZE,
            is_training=True,
            no_aug=True,
            auto_augment=None,
            interpolation=config.DATA.INTERPOLATION,)
    elif config.AUG.TYPE == 'simple':
        transform = create_transform(
            input_size = config.DATA.IMG_SIZE,
            is_training=True,
            hflip=0.5,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=None,
            interpolation=config.DATA.INTERPOLATION,)
    elif config.AUG.TYPE == 'simple-no-color':
        transform = create_transform(
            input_size = config.DATA.IMG_SIZE,
            is_training=True,
            hflip=0.5,
            color_jitter=None,
            auto_augment=None,
            interpolation=config.DATA.INTERPOLATION,)
    else:
        raise NotImplementedError("Unknown arg for data augmentation")
    
    if not resize_im:
        transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
    if config.TRAIN.TWOCROP:
        transform = TwoCropTransform(transform)
        
    return transform
        
    

    

