import torch
import torch.nn as nn
import torch.nn.functional as F


def init_func(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class PRPE(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(PRPE, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1),
            nn.Conv2d(in_ch // 2, int(3 * in_ch / 4), 1),
            nn.Conv2d(int(3 * in_ch / 4), int(3 * in_ch / 4), 3, groups=int(3 * in_ch / 4), padding=1, stride=stride),
            nn.Conv2d(int(3 * in_ch / 4), in_ch // 2, 1),
            nn.Conv2d(in_ch // 2, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )
        
        self.block.apply(init_func)
        
    def forward(self, x):
        return self.block(x)


class CovidNet(nn.Module):
    def __init__(self, num_classes):
        super(CovidNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(2, 2)
        
        self.PRPE_11 = PRPE(24, 84)
        self.PRPE_12 = PRPE(84, 84)
        self.PRPE_13 = PRPE(84, 84)
        
        self.PRPE_21 = PRPE(84, 152, stride=2)
        self.PRPE_22 = PRPE(152, 152)
        self.PRPE_23 = PRPE(152, 152)
        self.PRPE_24 = PRPE(152, 152)
        
        self.PRPE_31 = PRPE(152, 268, stride=2)
        self.PRPE_32 = PRPE(268, 268)
        self.PRPE_33 = PRPE(268, 268)
        self.PRPE_34 = PRPE(268, 268)
        self.PRPE_35 = PRPE(268, 268)
        self.PRPE_36 = PRPE(268, 268)
        
        self.PRPE_41 = PRPE(268, 412, stride=2)
        self.PRPE_42 = PRPE(412, 412)
        self.PRPE_43 = PRPE(412, 412)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dense = nn.Linear(412, num_classes)
        
        self.shortcut_conv1 = nn.Conv2d(24, 84, 1)
        self.shortcut_conv2 = nn.Conv2d(84, 152, 1, stride=2)
        self.shortcut_conv3 = nn.Conv2d(152, 268, 1, stride=2)
        self.shortcut_conv4 = nn.Conv2d(268, 412, 1, stride=2)
        
        
    def forward(self, x):
        x = self.max_pool(self.conv1(x))
        
        shortcut_1 = self.shortcut_conv1(x)
        h11 = self.PRPE_11(x)
        h12 = self.PRPE_12(h11 + shortcut_1)
        h13 = self.PRPE_13(h11 + h12 + shortcut_1)
        
        shortcut_2 = self.shortcut_conv2(h11 + h12 + h13 + shortcut_1)
        h21 = self.PRPE_21(h11 + h12 + h13 + shortcut_1)
        h22 = self.PRPE_22(h21 + shortcut_2)
        h23 = self.PRPE_23(h21 + h22 + shortcut_2)
        h24 = self.PRPE_24(h21 + h22 + h23 + shortcut_2)
        
        shortcut_3 = self.shortcut_conv3(h21 + h22 + h23 + h24 + shortcut_2)
        h31 = self.PRPE_31(h21 + h22 + h23 + h24 + shortcut_2)
        h32 = self.PRPE_32(h31 + shortcut_3)
        h33 = self.PRPE_33(h31 + h32 + shortcut_3)
        h34 = self.PRPE_34(h31 + h32 + h33 + shortcut_3)
        h35 = self.PRPE_35(h31 + h32 + h33 + h34 + shortcut_3)
        h36 = self.PRPE_36(h31 + h32 + h33 + h34 + h35 + shortcut_3)
        
        shortcut_4 = self.shortcut_conv4(h31 + h32 + h33 + h34 + h35 + h36 + shortcut_3)
        h41 = self.PRPE_41(h31 + h32 + h33 + h34 + h35 + h36 + shortcut_3)
        h42 = self.PRPE_42(h41 + shortcut_4)
        h43 = self.PRPE_43(h41 + h42 + shortcut_4)
        
        avg_pool = self.global_pool(h41 + h42 + h43 + shortcut_4)
        flat = torch.flatten(avg_pool, 1)
        
        out = self.dense(flat)
        return out