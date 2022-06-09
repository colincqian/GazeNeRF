import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn


import os
import numpy as np

import shutil

from HeadNeRFOptions import BaseOptions
from NetWorks.HeadNeRFNet import HeadNeRFNet
from Utils.HeadNeRFLossUtils import HeadNeRFLossUtils
from Utils.RenderUtils import RenderUtils
from tqdm import tqdm

class Trainer(object):
    def __init__(self,config,data_loader):
        '''
        Training instance of headnerf
        '''
        self.config = config ## load configuration

        ####load configurations####################
        # data params 
        if config.is_train:
            self.train_loader = data_loader
            self.num_train = len(self.train_loader.dataset)
            print(f'Load {self.num_train} data samples')
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
            print(f'Load {self.num_test} data samples')
        self.batch_size = config.batch_size

        # training params
        self.epochs = config.epochs  # the total epoch to train
        self.start_epoch = 0
        self.lr = config.init_lr
        self.lr_patience = config.lr_patience
        self.lr_decay_factor = config.lr_decay_factor

        # misc params
        self.use_gpu = config.use_gpu
        self.ckpt_dir = config.ckpt_dir  # output dir
        self.print_freq = config.print_freq
        self.train_iter = 0
        self.pre_trained_model_path = config.pre_trained_model_path
        self.headnerf_options = config.headnerf_options

        # configure tensorboard logging
        log_dir = './logs/' + os.path.basename(os.getcwd())
        if os.path.exists(log_dir) and os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

        #build model
        if self.headnerf_options:
            check_dict = torch.load(self.headnerf_options, map_location=torch.device("cpu"))

            para_dict = check_dict["para"]
            self.opt = BaseOptions(para_dict)

            self.model = HeadNeRFNet(self.opt, include_vd=False, hier_sampling=False)        
            self.model.load_state_dict(check_dict["net"])
        else:
            self.opt = BaseOptions()
            self.model = HeadNeRFNet(self.opt, include_vd=False, hier_sampling=False)        
        
        ##device setting
        if self.use_gpu and torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            gpu_id = torch.cuda.current_device()
            self.device = torch.device("cuda:%d" % gpu_id)
            self.model.cuda()
            print(f'GPU name:{torch.cuda.get_device_name(gpu_id)}')
        else:
            self.device = torch.device("cpu")
            
        
        #initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(
            self.optimizer, step_size=self.lr_patience, gamma=self.lr_decay_factor)

        self._build_tool_funcs()

    def _build_tool_funcs(self):
        self.loss_utils = HeadNeRFLossUtils(device=self.device)
        self.render_utils = RenderUtils(view_num=45, device=self.device, opt=self.opt)
        
        self.xy = self.render_utils.ray_xy.to(self.device)
        self.uv = self.render_utils.ray_uv.to(self.device)
    
    def train(self):
        for epoch in range(self.start_epoch,self.epochs):
            print(
                '\nEpoch: {}/{} - base LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.lr)
            )

            #Training
            self.model.train()
            self.train_one_epoch(epoch,self.train_loader)

            add_file_name = 'epoch_' + str(epoch)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'scheule_state': self.scheduler.state_dict()
                 }, add=add_file_name
            )

            self.scheduler.step() 

        self.writer.close()
    
    def build_code_and_cam_info(self,mm3d_param):
        base_iden = mm3d_param['code_info']['base_iden'].squeeze(1)
        base_expr = mm3d_param['code_info']['base_expr'].squeeze(1)
        base_text = mm3d_param['code_info']['base_text'].squeeze(1)
        base_illu = mm3d_param['code_info']['base_illu'].squeeze(1)

        shape_code = torch.cat([base_iden, base_expr], dim=-1)
        appea_code = torch.cat([base_text, base_illu], dim=-1) 
        

        base_Rmats = mm3d_param['cam_info']["batch_Rmats"].squeeze(1)
        base_Tvecs = mm3d_param['cam_info']["batch_Tvecs"].squeeze(1)
        batch_inv_inmat = mm3d_param['cam_info']["batch_inv_inmats"].squeeze(1)

        cam_info = {
                "batch_Rmats": base_Rmats.to(self.device),
                "batch_Tvecs": base_Tvecs.to(self.device),
                "batch_inv_inmats": batch_inv_inmat.to(self.device)
            }
        code_info = {
            "bg_code": None, 
            "shape_code":shape_code.to(self.device), 
            "appea_code":appea_code.to(self.device), 
        }
        return code_info,cam_info

    def train_one_epoch(self, epoch, data_loader, is_train=True):
        loop_bar = tqdm(enumerate(data_loader), leave=True)
        for iter,data_info in loop_bar:
            with torch.set_grad_enabled(True):
                code_info,cam_info = self.build_code_and_cam_info(data_info['_3dmm'])
                pred_dict = self.model( "train", self.xy, self.uv,  **code_info, **cam_info)
                gt_img = data_info['img'].squeeze(1); mask_img = data_info['img_mask'].squeeze(1)
                batch_loss_dict = self.loss_utils.calc_total_loss(
                    delta_cam_info=None, opt_code_dict=None, pred_dict=pred_dict, 
                    gt_rgb=gt_img.to(self.device), mask_tensor=mask_img.to(self.device)
                )
            self.optimizer.zero_grad()
            batch_loss_dict["total_loss"].backward()
            self.optimizer.step()
            loop_bar.set_description("Opt, Loss: %.6f  " % batch_loss_dict["head_loss"].item())          
    
    

    def save_checkpoint(self, state, add=None):
        """
        Save a copy of the model
        """
        if add is not None:
            filename = add + '_ckpt.pth.tar'
        else:
            filename ='ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        print('save file to: ', filename)

    def load_checkpoint(self, input_file_path='./ckpt/ckpt.pth.tar', is_strict=True):
        """
        Load the copy of a model.
        """
        print('load the pre-trained model: ', input_file_path)
        ckpt = torch.load(input_file_path)

        # load variables from checkpoint
        self.model.load_state_dict(ckpt['model_state'], strict=is_strict)
        self.optimizer.load_state_dict(ckpt['optim_state'])
        self.scheduler.load_state_dict(ckpt['scheule_state'])
        self.start_epoch = ckpt['epoch'] - 1

        print(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                input_file_path, ckpt['epoch'])
        )

