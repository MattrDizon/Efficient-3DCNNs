import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import *
from train import train_epoch
from validation import val_epoch
import test

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


if __name__ == '__main__':
    opt = parse_opts()
    opt.root_path = '~/'
    opt.video_path = '~/Thesis/FSL105_jpg_30'
    opt.annotation_path = '~/Thesis/FSL105_anno_30/ucf101_01.json'
    opt.result_path = 'Efficient-3DCNNs/result_ensemble'
    opt.dataset = 'ucf101'
    opt.n_classes = 30
    opt.width_mult = 0.5
    opt.train_crop = 'center'
    opt.learning_rate = 0.1
    opt.sample_duration = 16
    opt.downsample = 2
    opt.batch_size = 64
    opt.n_threads = 16
    opt.checkpoint = 1
    opt.n_val_samples = 1
    opt.groups = 3

    opt.test = True
    opt.no_train = True
    opt.no_val = True
    opt.crop_position_in_test = 'c'

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.width_mult) + 'x',
                               opt.modality, str(opt.sample_duration)])
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    
    opt_mobilenet = lambda: None
    opt_mobilenet.model = 'mobilenet'
    opt_mobilenet.n_classes = opt.n_classes
    opt_mobilenet.sample_size = opt.sample_size
    opt_mobilenet.width_mult = opt.width_mult

    
    model_mobilenet, parameters = generate_model(opt)
    print("***MobileNet***")
    print(model_mobilenet)

    opt_shufflenet = lambda: None
    opt_shufflenet.model = 'shufflenet'
    opt_shufflenet.n_classes = opt.n_classes
    opt_shufflenet.sample_size = opt.sample_size
    opt_shufflenet.width_mult = opt.width_mult
    opt_shufflenet.opt.groups = opt.groups

    model_shufflenet, parameters = generate_model(opt)
    print("***ShuffleNet***")
    print(model_shufflenet)


    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)


    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        # assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']
        model_mobilenet.load_state_dict(checkpoint['state_dict'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        # assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']
        model_shufflenet.load_state_dict(checkpoint['state_dict'])



    print('run')


    # Do Not Remove
    # if opt.test:
    #     spatial_transform = Compose([
    #         Scale(int(opt.sample_size / opt.scale_in_test)),
    #         CornerCrop(opt.sample_size, opt.crop_position_in_test),
    #         ToTensor(opt.norm_value), norm_method
    #     ])
    #     # temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
    #     # temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
    #     temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
    #     target_transform = VideoID()

    #     test_data = get_test_set(opt, spatial_transform, temporal_transform,
    #                              target_transform)
    #     test_loader = torch.utils.data.DataLoader(
    #         test_data,
    #         batch_size=16,
    #         shuffle=False,
    #         num_workers=opt.n_threads,
    #         pin_memory=True)
    #     test.test(test_loader, model, opt, test_data.class_names)




