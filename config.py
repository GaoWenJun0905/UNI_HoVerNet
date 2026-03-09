import importlib
import random

import cv2
import numpy as np

from dataset_GWJ import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        # model_name = "hovernet"
        # model_name = "UNI_hovernet"
        model_name = "ConVNeXt_Base_TripMTCA_hovernet"
        model_mode = "fast" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        # nr_type = 5 # number of nuclear types (including background)
        nr_type = 3 # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        # act_shape = [270, 270] # patch shape used as input to network - central crop performed after augmentation
        act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
        # out_shape = [80, 80] # patch shape at output of network
        out_shape = [164, 164] # patch shape at output of network

        if model_mode == "original":
            if act_shape != [270,270] or out_shape != [80,80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256,256] or out_shape != [164,164]:
                raise Exception("If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.dataset_name = "f1" # extracts dataset info from dataset.py
        # self.log_dir = "/media/server/data_2/24_GaoWenJun/UNI_HoVerNet/Logs/20260303-2" # where checkpoints will be saved
        self.log_dir = "/media/server/data_2/24_GaoWenJun/ConVNeXt_Base_TripMTCA_HoVerNet/Logs/20260309-1" # where checkpoints will be saved

        # paths to training and validation patches
        self.train_dir_list = [
            # "/home/wenjun_data/SMILE_Data/Data/TrainingData/test/HoVerNet/0.9:0.1_TTF-1_slide4/f1/f1/train/540x540_500x500",
            # "/home/wenjun_data/SMILE_Data/Data/TrainingData/test/SMILE/0.9:0.1_WT-1_slide4/f1/f1/train/540x540_500x500"
            # '/home/wenjun_data/SMILE_Data/Data/TrainingData/0.9:0.1_TTF-1_slide4/f1/f1/train/540x540_500x500',
            # '/home/wenjun_data/SMILE_Data/Data/TrainingData/test/HoVerNet/0.9:0.1_WT-1_slide4/f1/f1/train/540x540_500x500'
            '/home/wenjun_data/SMILE_Data/Data/TrainingData/20260208_test/0.9:0.1_TTF-1_slide4/f1/f1/train/540x540_500x500',
            '/home/wenjun_data/SMILE_Data/Data/TrainingData/20260208_test/0.9:0.1_WT-1_slide4/f1/f1/train/540x540_500x500'
        ]
        self.valid_dir_list = [
            # "/home/wenjun_data/SMILE_Data/Data/TrainingData/test/HoVerNet/0.9:0.1_TTF-1_slide4/f1/f1/valid/540x540_500x500",
            # "/home/wenjun_data/SMILE_Data/Data/TrainingData/test/SMILE/0.9:0.1_WT-1_slide4/f1/f1/valid/540x540_500x500"
            # '/home/wenjun_data/SMILE_Data/Data/TrainingData/0.9:0.1_TTF-1_slide4/f1/f1/valid/540x540_500x500',
            # '/home/wenjun_data/SMILE_Data/Data/TrainingData/test/HoVerNet/0.9:0.1_WT-1_slide4/f1/f1/valid/540x540_500x500'
            '/home/wenjun_data/SMILE_Data/Data/TrainingData/20260208_test/0.9:0.1_TTF-1_slide4/f1/f1/valid/540x540_500x500',
            '/home/wenjun_data/SMILE_Data/Data/TrainingData/20260208_test/0.9:0.1_WT-1_slide4/f1/f1/valid/540x540_500x500'
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models_GWJ.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
