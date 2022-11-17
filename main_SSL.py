import os
import time
import datetime
import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import load_cfg
from data import build_loader
from models import build_model
from BYOL import BYOL
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor


try:
    from apex import amp
except ImportError:
    amp = None

    
def parse_args():
    parser = argparse.ArgumentParser('Script for training Self-supervised learning models', add_help=False)
    parser.add_argument('--general-cfg', type=str, default='configs/general_config_SSL.yaml', help='path to general config file')
    parser.add_argument('--cfg', type=str, help='path to specific config file')
    
    parser.add_argument('--model-arch', type=str, help="select model architecture")
    parser.add_argument('--aug-level', type=int, choices=range(7),
                        help="Augmentation level, (default: 4)")
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--base-lr', type=float, help="base learning rate for optimization")

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output/SSL', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    
    parser.add_argument('--seed', type=int, help='random seed, default:0')
    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    
    args, unparsed = parser.parse_known_args()
    
    config = load_cfg(args)
    return config

    
def main(config):
    train_loader, _, _, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.ARCH}")
    model = build_model(config).cuda()
    logger.info(str(model))
    learner = BYOL(model, config)
    
    optimizer = build_optimizer(config, learner)
    if config.AMP_OPT_LEVEL != "O0":
        learner, optimizer = amp.initialize(learner, optimizer, opt_level=config.AMP_OPT_LEVEL)
    
    learner = torch.nn.parallel.DistributedDataParallel(learner, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
    model_without_ddp = learner.module.net
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        print('resume_file: ',resume_file)
        print(config.MODEL.RESUME)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
    else:
        logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
        
    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

        
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_epoch(config, learner, train_loader, optimizer, epoch, mixup_fn, lr_scheduler)
        
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
        
        
def train_epoch(config, model, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    model.module.update_moving_average()
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    batch_time, loss_meter, norm_meter = AverageMeter(), AverageMeter(), AverageMeter()
    
    start = time.time()
    end = time.time()
    for idx, ((v1, v2), targets) in enumerate(data_loader):
        v1, v2 = v1.cuda(non_blocking=True), v2.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        if mixup_fn is not None:
            v1, _ = mixup_fn(v1, targets)
            v2, _ = mixup_fn(v2, targets)
        
        loss = model(v1, v2)
        
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != 'O0':
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(learner.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                model.module.update_moving_average()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(learner.parameters())
    
            optimizer.step()
            model.module.update_moving_average()
            lr_scheduler.step_update(epoch * num_steps + idx)
        
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")    
    
    
if __name__ == '__main__':
    config = parse_args()
    
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"
        
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
        
    torch.cuda.set_device(config.LOCAL_RANK)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()
    
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 128.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 128.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 128.0
    
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()
    
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        
    logger.info(config.dump())
    
    main(config)