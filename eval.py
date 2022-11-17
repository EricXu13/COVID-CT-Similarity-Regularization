import os
import torch
import argparse
import numpy as np
import torch

from config import load_cfg


def parse_args():
    parser = argparse.ArgumentParser('Script for Evaluate deep learning models for COVID-CT Diagnosis', add_help=False)
    parser.add_argument('--general-cfg', type=str, default='configs/general_config.yaml', help='path to general config file')
    parser.add_argument('--cfg', type=str, help='path to specific config file')
    
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_arch>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--local_rank', default=0)
    args, unparsed = parser.parse_known_args()
    
    config = load_cfg(args)
    return config


if __name__ == '__main__':
    config = parse_args()