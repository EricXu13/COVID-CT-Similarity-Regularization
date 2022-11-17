import os
from yacs.config import CfgNode as CN


def load_cfg(args):
    config = CN.load_cfg(open(args.general_cfg, 'r'))
    if args.cfg:
        config.merge_from_file(args.cfg)
    
    config.defrost()
    if hasattr(args, 'model_arch') and args.model_arch:
        config.MODEL.ARCH = args.model_arch
    if hasattr(args, 'aug_level') and args.aug_level >= 0:
        config.AUG.LEVEL = args.aug_level
    if hasattr(args, 'batch_size') and args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if hasattr(args, 'projection_size') and args.projection_size:
        config.SSL.PROJECTION_SIZE = args.projection_size
    if hasattr(args, 'decay_min_rate') and args.decay_min_rate:
        config.TRAIN.SCALING_SCHEDULER.MIN_RATE = args.decay_min_rate
    if hasattr(args, 'max_rate') and args.max_rate:
        config.TRAIN.SCALING_SCHEDULER.MAX_RATE = args.max_rate
    if hasattr(args, 'base_lr') and args.base_lr:
        config.TRAIN.BASE_LR = args.base_lr
    if hasattr(args, 'resume') and args.resume:
        config.MODEL.RESUME = args.resume
        config.TRAIN.AUTO_RESUME = False
    if hasattr(args, 'accumulation_steps') and args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if hasattr(args, 'amp_opt_level') and args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.seed:
        config.SEED = args.seed

    config.LOCAL_RANK = args.local_rank
    config.OUTPUT = os.path.join(config.OUTPUT, str(config.SEED), config.MODEL.ARCH, config.TAG)
    config.freeze()
    
    return config


def load_cfg_eval(args):
    config = CN.load_cfg(open(args.general_cfg, 'r'))
    if args.cfg:
        config.merge_from_file(args.cfg)
    config.defrost()
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.ARCH, config.TAG)
#     config.OUTPUT = os.path.join(config.OUTPUT, str(config.SEED), config.MODEL.ARCH, config.TAG)
    config.freeze()
    
    return config