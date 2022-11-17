import copy
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
        
        
# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

        
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)        

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False
        
    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        
    def _hook(self, _, input, output):
        device = input[0].device
        if output.dim() > 2:
            output = F.adaptive_avg_pool2d(output, (1, 1))
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True
        
    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden
    
    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation
    
    
class BYOL_Constraint_Wrapper(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.net = net
        self.squeeze = self.net.__class__.__name__ == 'SqueezeNet'

        self.online_encoder = NetWrapper(net, args.SSL.PROJECTION_SIZE, args.SSL.PROJECTION_HIDDEN_SIZE, -2)

        self.target_encoder = None
        self.target_ema_updater = EMA(args.SSL.MOVING_AVERAGE_DECAY)

        self.online_predictor = MLP(args.SSL.PROJECTION_SIZE, args.SSL.PROJECTION_SIZE, 
                                    args.SSL.PROJECTION_HIDDEN_SIZE)
        
        self.fc = [*self.net.children()][-1]
        
        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward((torch.randn(2, 3, args.DATA.IMG_SIZE, args.DATA.IMG_SIZE, device=device),
                    torch.randn(2, 3, args.DATA.IMG_SIZE, args.DATA.IMG_SIZE, device=device)))
        
    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
        
    def compute_constraint(self, view1, view2):
        online_proj_one, representation_one = self.online_encoder(view1)
        online_proj_two, representation_two = self.online_encoder(view2)
                
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        
        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_proj_one, _ = target_encoder(view1)
            target_proj_two, _ = target_encoder(view2)
            target_proj_one.detach_()
            target_proj_two.detach_()
        
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return representation_one, representation_two, loss.mean()
    
    def forward(self, x):
        if isinstance(x, tuple):
            (view1, view2) = x
            repre1, repre2, constraint = self.compute_constraint(view1, view2)
            if self.squeeze:
                repre1, repre2 = repre1.unsqueeze(-1).unsqueeze(-1), repre2.unsqueeze(-1).unsqueeze(-1)
                logits1, logits2 = torch.flatten(self.fc(repre1), 1),  torch.flatten(self.fc(repre2), 1)
            else:
                logits1, logits2 = self.fc(repre1), self.fc(repre2)
            return logits1, logits2, constraint
        else:
            repre = self.online_encoder.get_representation(x)
            if self.squeeze:
                repre = repre.unsqueeze(-1).unsqueeze(-1)
                logits = torch.flatten(self.fc(repre), 1)
            else:
                logits = self.fc(repre)
            return logits
            
        

        