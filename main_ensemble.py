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
from opt_sync import *
from ensemble import *
from thop import profile

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def flops(model):
    flops, prms = profile(model, input_size=(1, 3, 16, 112, 112))
    print("Total number of FLOPs: ", flops)
    print(prms)

if __name__ == '__main__':
    opt = parse_opts()
    # opt.result_path = 'Efficient-3DCNNs/result_ensemble_test'

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
    opt_shufflenet = lambda: None

    assign_attributes(opt, opt_mobilenet, opt_shufflenet)

    opt_mobilenet.model = 'mobilenet'
    opt_shufflenet.model = 'shufflenet'
    
    model_mobilenet, parameters_mobilenet = generate_model(opt_mobilenet)
    # print("***MobileNet***")
    # print(model_mobilenet)

    model_shufflenet, parameters_shufflenet = generate_model(opt_shufflenet)
    # print("***ShuffleNet***")
    # print(model_shufflenet)


    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    opt_mobilenet.resume_path = '/home/matthew/Efficient-3DCNNs_epoch100/result_mobilenet_bs16_lr0.1/ucf101_mobilenet_0.5x_RGB_16_best.pth'
    if opt_mobilenet.resume_path:
        print('loading checkpoint {}'.format(opt_mobilenet.resume_path))
        opt_mobilenet.checkpoint = torch.load(opt_mobilenet.resume_path)
        # assert opt.arch == checkpoint['arch']
        opt_mobilenet.begin_epoch = opt_mobilenet.checkpoint['epoch']
        model_mobilenet.load_state_dict(opt_mobilenet.checkpoint['state_dict'])

    model_shufflenet.resume_path = '/home/matthew/Efficient-3DCNNs_epoch100/results_shufflenet_bs16_lr0.1/ucf101_shufflenet_0.5x_RGB_16_best.pth'
    if model_shufflenet.resume_path:
        print('loading checkpoint {}'.format(model_shufflenet.resume_path))
        model_shufflenet.checkpoint = torch.load(model_shufflenet.resume_path)
        # assert opt.arch == checkpoint['arch']
        opt_shufflenet.begin_epoch = model_shufflenet.checkpoint['epoch']
        model_shufflenet.load_state_dict(model_shufflenet.checkpoint['state_dict'])

    model = EnsembleModel(model_mobilenet, model_shufflenet, opt.n_classes)
    
    print('run')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)
    print(model)

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        # temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
        # temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=16,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
    
    flops(model)
    # flops(model_mobilenet)
    # flops(model_shufflenet)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)
