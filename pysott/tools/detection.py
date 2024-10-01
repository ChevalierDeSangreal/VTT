from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True

class Detnet():
    """
    Needs to verify the format of picture and bbox
    """
    def __init__(self, init_pic, init_bbox):
        # load config
        cfg.merge_from_file('/home/wangzimo/VTT/VTT/pysott/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create model
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load('/home/wangzimo/VTT/VTT/pysott/experiments/siamrpn_r50_l234_dwxcorr/model.pth',
            map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        # build tracker
        self.tracker = build_tracker(model)

        self.tracker.init(init_pic, init_bbox)
    
    def __call__(self, pic):
        outputs = self.tracker.track(pic)
        bbox = outputs['bbox']
        return bbox



