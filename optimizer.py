import torch.optim as optim


def build_optimizer(config, model):
    opt_name = config.TRAIN.OPTIMIZER.NAME.lower()
    if hasattr(config, 'LINEAR_EVAL') and config.LINEAR_EVAL:
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias']:
                param.requires_grad = False
        
    lr = config.TRAIN.BASE_LR
    momentum = config.TRAIN.OPTIMIZER.MOMENTUM
    weight_decay = config.TRAIN.WEIGHT_DECAY
    eps = config.TRAIN.OPTIMIZER.EPS
    nesterov=config.TRAIN.OPTIMIZER.NESTEROV
    betas = config.TRAIN.OPTIMIZER.BETAS
    
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    if opt_name == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif opt_name == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        optimizer = optim.Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"The choosed optimizer is not defined: {opt_name}") 
    
    return optimizer