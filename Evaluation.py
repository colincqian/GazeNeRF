from os.path import join
import os
from re import sub
from tracemalloc import start
import torch
import numpy as np
from NetWorks.HeadNeRFNet import HeadNeRFNet,HeadNeRFNet_Gaze
import cv2
from train_headnerf import BaseOptions
from Utils.HeadNeRFLossUtils import HeadNeRFLossUtils
from Utils.RenderUtils import RenderUtils
import pickle as pkl
import time
from glob import glob
from tqdm import tqdm
import imageio
import random
import argparse
from tool_funcs import put_text_alignmentcenter
import h5py
from Utils.D6_rotation import gaze_to_d6
from scipy import stats

import matplotlib
try:
	matplotlib.use("QtAgg")
except:
	pass
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shutil

from Utils.Eval_utils import calc_eval_metrics
import pickle
from prettytable import PrettyTable
from train_headnerf import load_config

#define eye gaze base
# UPPER_RIGHT = torch.tensor([0.25,-0.5])
# UPPER_LEFT = torch.tensor([0.25,0.6])
# LOWER_RIGHT = torch.tensor([-0.5,-0.5])
# LOWER_LEFT = torch.tensor([-0.5,0.6])
UPPER_RIGHT = torch.tensor([0.8,-0.8])
UPPER_LEFT = torch.tensor([0.8,0.8])
LOWER_RIGHT = torch.tensor([-0.8,-0.8])
LOWER_LEFT = torch.tensor([-0.8,0.8])

model_list = {
    'HeadNeRF':HeadNeRFNet,
    'HeadNeRF_Gaze':HeadNeRFNet_Gaze
}


def gaze_feat_tensor(gaze_dim,scale_factor,base_gaze_value):

    # base_gaze_np = base_gaze_value.cpu().detach().numpy()
    # base_gaze_d6 = gaze_to_d6(base_gaze_np)
    # res = torch.from_numpy(base_gaze_d6)
    # res = res.repeat(1,gaze_dim//res.size(0))
    
    return base_gaze_value.repeat(1,gaze_dim//base_gaze_value.size(0)) * scale_factor


class FittingImage(object):
    
    def __init__(self, model_path, save_root, gpu_id, 
                    config,
                    include_eye_gaze = True,
                    eye_gaze_dim = 16,
                    gaze_scale_factor = 1,
                    vis_vect = False,
                    D6_rotation = False,
                    model_name = 'HeadNeRF') -> None:
        super().__init__()
        self.model_path = model_path
        if gpu_id >= 0:
            self.device = torch.device("cuda:%d" % gpu_id)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.save_root = save_root
        self.opt_cam = True
        self.view_num = 45
        self.duration = 3.0 / self.view_num
        self.model_name = os.path.basename(model_path)[:-4]
        self.include_eye_gaze = include_eye_gaze
        self.eye_gaze_dim = eye_gaze_dim
        self.scale_factor = gaze_scale_factor
        self.vis_vect = vis_vect
        self.use_6D_rotation = D6_rotation
        self.model_name = model_name
        self.error_ave_dict = {'ave_e_h0':[],
                                'ave_e_v0':[],
                                'ave_e_h1':[],
                                'ave_e_v1':[],
                                'ave_e_h2':[],
                                'ave_e_v2':[],
        }
        self.config = config
        self.build_info()
        self.build_tool_funcs()


    def build_info(self):
        check_dict = torch.load(self.model_path, map_location=torch.device("cpu"))

        para_dict = check_dict["para"]
        base_opt = self.config.base_opt
        self.opt = BaseOptions(base_opt,para_dict) #just use the same feature size as the para_dict

        self.featmap_size = self.opt.featmap_size
        self.pred_img_size = self.opt.pred_img_size
        
        if not os.path.exists(self.save_root): os.mkdir(self.save_root)

        net = model_list[self.model_name](self.opt, include_vd=False, hier_sampling=False, eye_gaze_dim=self.eye_gaze_dim)        
        net.load_state_dict(check_dict["net"],strict=False)
        
        self.net = net.to(self.device)
        self.net.eval()


    def build_tool_funcs(self):
        self.loss_utils = HeadNeRFLossUtils(device=self.device)
        self.render_utils = RenderUtils(view_num=45, device=self.device, opt=self.opt)
        
        self.xy = self.render_utils.ray_xy
        self.uv = self.render_utils.ray_uv
    

    def load_data(self,hdf_path,img_index,cam_idx = 0):
        #process imgs
        '''
        if cam_idx >= 0 only load data wtih cam_idx
        if cam_idx < 0 load data given the img_index
        '''
        self.hdf = h5py.File(hdf_path, 'r', swmr=True)
        assert self.hdf.swmr_mode

        if cam_idx >= 0:
            while cam_idx != self.hdf['cam_index'][img_index]:
                img_index += 1
                if img_index == self.hdf['cam_index'].shape[0]:
                    print(f'Cannot find camera index {cam_idx} in {hdf_path}')
                    raise ValueError

        img_size = (self.pred_img_size, self.pred_img_size)

        img = self.hdf['face_patch'][img_index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0
        gt_img_size = img.shape[0]
        if gt_img_size != self.pred_img_size:
            img = cv2.resize(img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        
        self.uncropped_gt_image = (img * 255).astype(np.uint8)
        
        mask_img =  self.hdf['mask'][img_index]
        eye_mask_img = self.hdf['eye_mask'][img_index]
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        if eye_mask_img.shape[0] != self.pred_img_size:
            eye_mask_img = cv2.resize(eye_mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        self.head_mask_np = mask_img
        img[mask_img < 0.5] = 1.0
        
        self.img_tensor = (torch.from_numpy(img).permute(2, 0, 1)).unsqueeze(0).to(self.device)
        self.mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)
        self.eye_mask_tensor = torch.from_numpy(eye_mask_img[None, :, :]).unsqueeze(0).to(self.device)

        gaze_label = self.hdf['face_gaze'][img_index]
        gaze_label = gaze_label.astype('float')
        self.gaze_tensor = (torch.from_numpy(gaze_label)).to(self.device)
        self.base_gaze = self.gaze_tensor.repeat(1,self.eye_gaze_dim//self.gaze_tensor.size(0))
        

       # load init codes from the results generated by solving 3DMM rendering opt.
        nl3dmm_para_dict = self.hdf['nl3dmm']
        base_code = torch.from_numpy(nl3dmm_para_dict["code"][img_index]).float().detach().unsqueeze(0).to(self.device)
        
        self.base_iden = base_code[:, :self.opt.iden_code_dims]
        self.base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
        self.base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims 
                                                            + self.opt.expr_code_dims + self.opt.text_code_dims]
        self.base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]
        
        self.base_c2w_Rmat = torch.from_numpy(nl3dmm_para_dict["c2w_Rmat"][img_index]).float().detach().unsqueeze(0)
        self.base_c2w_Tvec = torch.from_numpy(nl3dmm_para_dict["c2w_Tvec"][img_index]).float().detach().unsqueeze(0).unsqueeze(-1)
        self.base_w2c_Rmat = torch.from_numpy(nl3dmm_para_dict["w2c_Rmat"][img_index]).float().detach().unsqueeze(0)
        self.base_w2c_Tvec = torch.from_numpy(nl3dmm_para_dict["w2c_Tvec"][img_index]).float().detach().unsqueeze(0).unsqueeze(-1)

        temp_inmat = torch.from_numpy(nl3dmm_para_dict["inmat"][img_index]).detach().unsqueeze(0)
        temp_inmat[:, :2, :] *= (self.featmap_size / gt_img_size)
        
        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0
        
        self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.to(self.device),
            "batch_Tvecs": self.base_c2w_Tvec.to(self.device),
            "batch_inv_inmats": self.temp_inv_inmat.to(self.device).float()
        }
        

    @staticmethod
    def eulurangle2Rmat(angles):
        batch_size = angles.size(0)
        
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXs = torch.eye(3, device=angles.device).view(1, 3, 3).repeat(batch_size, 1, 1)
        rotYs = rotXs.clone()
        rotZs = rotXs.clone()
        
        rotXs[:, 1, 1] = cosx
        rotXs[:, 1, 2] = -sinx
        rotXs[:, 2, 1] = sinx
        rotXs[:, 2, 2] = cosx
        
        rotYs[:, 0, 0] = cosy
        rotYs[:, 0, 2] = siny
        rotYs[:, 2, 0] = -siny
        rotYs[:, 2, 2] = cosy

        rotZs[:, 0, 0] = cosz
        rotZs[:, 0, 1] = -sinz
        rotZs[:, 1, 0] = sinz
        rotZs[:, 1, 1] = cosz
        
        res = rotZs.bmm(rotYs.bmm(rotXs))
        return res
    
    
    def build_code_and_cam(self):
        
        # code
        if self.include_eye_gaze and self.model_name == 'HeadNeRF':
            shape_code = torch.cat([self.base_iden + self.iden_offset, self.base_expr + self.expr_offset,self.base_gaze + self.gaze_offset], dim=-1)
            #appea_code = torch.cat([self.base_text, self.base_illu, self.base_gaze], dim=-1) + self.appea_offset
            appea_code = torch.cat([self.base_text, self.base_illu], dim=-1) + self.appea_offset
        else:
            shape_code = torch.cat([self.base_iden + self.iden_offset, self.base_expr + self.expr_offset], dim=-1)
            appea_code = torch.cat([self.base_text, self.base_illu], dim=-1) + self.appea_offset
        
        opt_code_dict = {
            "bg":None,
            "iden":self.iden_offset,
            "expr":self.expr_offset,
            "appea":self.appea_offset
        }

        if self.include_eye_gaze and self.model_name == 'HeadNeRF':
            opt_code_dict['gaze']=self.gaze_offset
        
        code_info = {
            "bg_code": None, 
            "shape_code":shape_code.float(), 
            "appea_code":appea_code.float(), 
        }

        #cam
        if self.opt_cam:
            delta_cam_info = {
                "delta_eulur": self.delta_EulurAngles, 
                "delta_tvec": self.delta_Tvecs
            }

            batch_delta_Rmats = self.eulurangle2Rmat(self.delta_EulurAngles)
            base_Rmats = self.cam_info["batch_Rmats"]
            base_Tvecs = self.cam_info["batch_Tvecs"]
            
            cur_Rmats = batch_delta_Rmats.bmm(base_Rmats)
            cur_Tvecs = batch_delta_Rmats.bmm(base_Tvecs) + self.delta_Tvecs
            
            batch_inv_inmat = self.cam_info["batch_inv_inmats"] #[N, 3, 3]    
            batch_cam_info = {
                "batch_Rmats": cur_Rmats,
                "batch_Tvecs": cur_Tvecs,
                "batch_inv_inmats": batch_inv_inmat
            }
            
        else:
            delta_cam_info = None
            batch_cam_info = self.cam_info

 
        if self.model_name == 'HeadNeRF_Gaze':
            code_info.update(
                {
                    "input_gaze": self.base_gaze.float().to(self.device),
                    "eye_mask" : self.eye_mask_tensor.float().to(self.device)
                }

            )
        return code_info, opt_code_dict, batch_cam_info, delta_cam_info
    
    
    @staticmethod
    def enable_gradient(tensor_list):
        for ele in tensor_list:
            ele.requires_grad = True


    def perform_fitting(self):
        self.delta_EulurAngles = torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        self.delta_Tvecs = torch.zeros((1, 3, 1), dtype=torch.float32).to(self.device)

        self.iden_offset = torch.zeros((1, 100), dtype=torch.float32).to(self.device)
        self.expr_offset = torch.zeros((1, 79), dtype=torch.float32).to(self.device)
        self.appea_offset = torch.zeros((1, 127), dtype=torch.float32).to(self.device)

        if self.include_eye_gaze and self.model_name == 'HeadNeRF':
            self.gaze_offset = torch.zeros((1, self.eye_gaze_dim), dtype=torch.float32).to(self.device)
            self.enable_gradient([self.gaze_offset])
            #self.appea_offset = torch.cat((self.appea_offset,torch.zeros(1,self.eye_gaze_dim,device = self.device)),dim=1)

        if self.opt_cam:
            self.enable_gradient(
                [self.iden_offset, self.expr_offset, self.appea_offset, self.delta_EulurAngles, self.delta_Tvecs]
            )
        else:
            self.enable_gradient(
                [self.iden_offset, self.expr_offset, self.appea_offset]
            )
        
        init_learn_rate = 0.01
        
        step_decay = 300
        iter_num = 1
        
        params_group = [
            {'params': [self.iden_offset], 'lr': init_learn_rate * 1.5},
            {'params': [self.expr_offset], 'lr': init_learn_rate * 1.5},
            {'params': [self.appea_offset], 'lr': init_learn_rate * 1.0},
        ]

        if self.include_eye_gaze and self.model_name == 'HeadNeRF':
            params_group.append({'params': [self.gaze_offset], 'lr': init_learn_rate * 1.0})
        
        if self.opt_cam:
            params_group += [
                {'params': [self.delta_EulurAngles], 'lr': init_learn_rate * 0.1},#0.1
                {'params': [self.delta_Tvecs], 'lr': init_learn_rate * 0.1},#0.1
            ]
            
        optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))
        lr_func = lambda epoch: 0.1 ** (epoch / step_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func) #adaptive learning rate
        
        gt_img = (self.img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        
        
        loop_bar = tqdm(range(iter_num),position=0)
        for iter_ in loop_bar:
            with torch.set_grad_enabled(True):
                code_info, opt_code_dict, cam_info, delta_cam_info = self.build_code_and_cam()
                pred_dict = self.net( "test", self.xy, self.uv,  **code_info, **cam_info)
                #input: xy: torch.Size([1, 2, 1024]),   uv:torch.Size([1, 1024, 2]) 
                #code info: appea: torch.Size([1, 127]), shape:torch.Size([1, 179])
                #cam info : batch_Rmats: torch.Size([1, 3, 3])  batch_Tvecs:torch.Size([1, 3, 1])   batch_inv_inmats:torch.Size([1, 3, 3])
                #pred_dict['coarse_dict'] -> dict_keys(['merge_img', 'bg_img']) -> torch.Size([1, 3, 512, 512])

                gt_label = {"gt_rgb":self.img_tensor}
                batch_loss_dict = self.loss_utils.calc_total_loss(
                    delta_cam_info=delta_cam_info, opt_code_dict=opt_code_dict, pred_dict=pred_dict, disp_pred_dict=None,
                    gt_rgb=gt_label, mask_tensor=self.mask_tensor,loss_weight=self.config.loss_config
                )

            optimizer.zero_grad()
            batch_loss_dict["total_loss"].backward()
            optimizer.step()
            scheduler.step()   
            loop_bar.set_description("Opt, Loss: %.6f  " % batch_loss_dict["head_loss"].item())          

            # coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
            # coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            # cv2.imwrite("./temp_res/opt_imgs/img_%04d.png" % iter_, coarse_fg_rgb[:, :, ::-1])

        coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
        coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        coarse_fg_rgb = cv2.cvtColor(coarse_fg_rgb, cv2.COLOR_BGR2RGB)
        res_img = np.concatenate([gt_img, coarse_fg_rgb], axis=1)
        
        self.res_img = res_img
        self.res_code_info = code_info
        self.res_cam_info = cam_info
        


    def save_res(self, base_name, save_root):
        
        # Generate Novel Views
        render_nv_res = self.render_utils.render_novel_views(self.net, self.res_code_info)
        NVRes_save_path = "%s/FittingResNovelView_%s.gif" % (save_root, base_name)
        imageio.mimsave(NVRes_save_path, render_nv_res, 'GIF', duration=self.duration)
        
        # Generate Rendered FittingRes
        img_save_path = "%s/FittingRes_%s.png" % (save_root, base_name)

        self.res_img = put_text_alignmentcenter(self.res_img, self.pred_img_size, "Input", (0,0,0), offset_x=0)
        self.res_img = put_text_alignmentcenter(self.res_img, self.pred_img_size, "Fitting", (0,0,0), offset_x=self.pred_img_size,)

        # self.res_img = cv2.putText(self.res_img, "Input", (110, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
        # self.res_img = cv2.putText(self.res_img, "Fitting", (360, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
        cv2.imwrite(img_save_path, self.res_img[:,:,::-1])
        
        self.tar_code_info ={}
        direction_list = [LOWER_RIGHT,UPPER_RIGHT,UPPER_LEFT,LOWER_LEFT,LOWER_RIGHT]
        morph_res_seq = []
        vec_results_seq = []
        for i in range(len(direction_list)-1):
            start = direction_list[i]
            end = direction_list[i+1]

            shape_code = self.res_code_info['shape_code'].clone().detach()
            appea_code = self.res_code_info['appea_code'].clone().detach()
            # shape_code[0,-self.eye_gaze_dim:] = -shape_code[0,-self.eye_gaze_dim:]
            # appea_code[0,-self.eye_gaze_dim:] = -appea_code[0,-self.eye_gaze_dim:]
            shape_code[0,-self.eye_gaze_dim:] = gaze_feat_tensor(self.eye_gaze_dim,self.scale_factor,end)
            #appea_code[0,-self.eye_gaze_dim:] = torch.ones(self.eye_gaze_dim)
            self.tar_code_info['shape_code'] = shape_code.clone().detach()
            self.tar_code_info['appea_code'] = appea_code.clone().detach()
            self.tar_code_info['bg_code'] = None
            shape_code[0,-self.eye_gaze_dim:] = gaze_feat_tensor(self.eye_gaze_dim,self.scale_factor,start)
            #appea_code[0,-self.eye_gaze_dim:] = -torch.ones(self.eye_gaze_dim)
            self.res_code_info['shape_code'] = shape_code.clone().detach()
            self.res_code_info['appea_code'] = appea_code.clone().detach()


            
            morph_res,vec_results = self.render_utils.render_gaze_redirect_res(self.net, self.res_code_info, self.tar_code_info, self.view_num,
                                                        self.scale_factor,self.eye_gaze_dim,vis_vect=self.vis_vect,D6_rotation=self.use_6D_rotation)
            #morph_res = self.render_utils.render_morphing_res(self.net, self.res_code_info, self.tar_code_info, self.view_num)

            morph_res_seq += morph_res
            vec_results_seq += vec_results


        morph_save_path = "%s/FittingResMorphing_%s.gif" % (save_root, base_name)
        imageio.mimsave(morph_save_path, morph_res_seq, 'GIF', duration=self.duration)

        image_vec_path = "%s/image_seq/" % (save_root)
        if not os.path.exists(image_vec_path):
            os.mkdir(image_vec_path)
        for ind,img_vec in enumerate(vec_results_seq):
            cv2.imwrite(os.path.join(image_vec_path,f"image{ind}.png"),img_vec)

        for k, v in self.res_code_info.items():
            if isinstance(v, torch.Tensor):
                self.res_code_info[k] = v.detach()
        
        temp_dict = {
            "code": self.res_code_info
        }

        torch.save(temp_dict, "%s/LatentCodes_%s_%s.pth" % (save_root, base_name, self.model_name))


    def fitting_single_images(self, hdf_file_path,image_index, save_root):
        self.load_data(hdf_file_path,image_index)
        # base_name = os.path.basename(img_path)[4:-4]

        self.perform_fitting()
        self.save_res(f'processed_image{image_index}', save_root)
    
    def render_gaze_redirection_gif(self,hdf_file_path,image_index, save_root,num_frames=10):
        if os.path.exists(save_root):
            shutil.rmtree(save_root)
            os.mkdir(save_root)
        else:
            os.mkdir(save_root)

        self.load_data(hdf_file_path,image_index)
        # base_name = os.path.basename(img_path)[4:-4]
        self.perform_fitting()

        self.tar_code_info ={}
        direction_list = [LOWER_RIGHT,UPPER_RIGHT,UPPER_LEFT,LOWER_LEFT,LOWER_RIGHT]
        morph_res_seq = []
        vec_results_seq = []
        for i in range(len(direction_list)-1):
            start = direction_list[i]
            end = direction_list[i+1]
            
            for i in tqdm(range(num_frames)):
                tv = 1.0 - (i / (num_frames - 1))
                input_face_gaze = start * tv + end * (1 - tv)
                rendered_results,cam_info,face_gaze = self.render_utils.render_face_with_gaze(self.net,self.res_code_info,face_gaze=input_face_gaze,scale_factor = 1,gaze_dim = self.eye_gaze_dim,cam_info=self.cam_info)
                rendered_results = cv2.cvtColor(rendered_results, cv2.COLOR_BGR2RGB)
                morph_res_seq.append(rendered_results)
                rendered_results_vect,e_v2,e_h2,_ = self.render_utils.render_gaze_vect(rendered_results,cam_info,face_gaze)
                vec_results_seq.append(rendered_results_vect)
        
        morph_save_path = os.path.join(save_root,'gaze_redirection_gif.gif')
        imageio.mimsave(morph_save_path, morph_res_seq, 'GIF', duration=self.duration)






    def gridsample_face_gaze(self,hdf_file_path,image_index,save_root,vis_vect=True,resolution=21,print_freq = 10,cam_index=0):
        self.load_data(hdf_file_path,image_index,cam_idx=cam_index)
        self.perform_fitting()
        e_v_map = np.zeros((resolution,resolution))
        e_h_map = np.zeros((resolution,resolution))

        loop_bar1 = tqdm(enumerate(np.linspace(1,-1,resolution,endpoint=True)),leave=True, position = 0, desc=" row loop")

        if os.path.exists(save_root):
            shutil.rmtree(save_root)
            os.mkdir(save_root)
        else:
            os.mkdir(save_root)

        for row_idx,pitch in loop_bar1:
            for col_idx,yaw in enumerate(np.linspace(1,-1,resolution,endpoint=True)):
                input_gaze = torch.tensor([pitch,yaw])
                rendered_results,cam_info,face_gaze = self.render_utils.render_face_with_gaze(self.net,self.res_code_info,face_gaze=input_gaze,scale_factor = 1,gaze_dim = self.eye_gaze_dim,cam_info=self.cam_info)
                if self.vis_vect:
                    rendered_results,e_v2,e_h2,_ = self.render_utils.render_gaze_vect(rendered_results,cam_info,face_gaze)
                e_v_map[row_idx,col_idx] = e_v2 * 180 / np.pi
                e_h_map[row_idx,col_idx] = e_h2 * 180 / np.pi ##in degree
                if (row_idx * resolution + col_idx ) % print_freq == 0:
                    cv2.imwrite(os.path.join(save_root,f'grid_sample{pitch,yaw}.png'),rendered_results)
        
        e_total_map = e_v_map + e_h_map

        self._polt_2d_map_with_colorbar(e_v_map,title='vertical_error',save_root=save_root)
        self._polt_2d_map_with_colorbar(e_h_map,title='horizontal_error',save_root=save_root)
        self._polt_2d_map_with_colorbar(e_total_map,title='total_error',save_root=save_root)

        np.save(os.path.join(save_root,'vertical_error_map.npy'),e_v_map)
        np.save(os.path.join(save_root,'horizontal_error_map.npy'),e_h_map)
        np.save(os.path.join(save_root,'total_error_map.npy'),e_total_map)
        
    def sample_face_gaze_ground_truth_image(self,hdf_file_path,image_sample_num,resolution=21,cam_index=0):
        '''
        sample ground truth label and compute error (gap between estimator and input gaze)
        and draw the error distribution over gaze space

        image_sample_num: how many gt sample you would like to have
        resolution: the number of bins on each axis in the 2d error map
        '''
        e_v1_list = []
        e_h1_list = []
        pitch_list = []
        yaw_list=[]

        for image_index in range(image_sample_num):
            self.load_data(hdf_file_path,image_index,cam_idx=cam_index)
            self.perform_fitting()

            inmat_np = torch.linalg.inv(self.res_cam_info['batch_inv_inmats']).detach().cpu().numpy()
            inmat_np = inmat_np.reshape((3,3))
            distortion_np = np.zeros([1,5])
            inmat_np[0,0] *=10; inmat_np[1,1] *=10; inmat_np[0,2] *=10; inmat_np[1,2] *=10
            cam_info = {'camera_matrix':inmat_np, 'camera_distortion':distortion_np}
            
            pitch_list.append(self.gaze_tensor[0].cpu().detach().numpy())
            yaw_list.append(self.gaze_tensor[1].cpu().detach().numpy())

            gt_img = (self.img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            try:
                gt_img,e_v1,e_h1,_ = self.render_utils.render_gaze_vect(gt_img,cam_info=cam_info,face_gaze=self.gaze_tensor)
                e_v1_list.append(e_v1 * 180/np.pi)
                e_h1_list.append(e_h1 * 180/np.pi)
            except:
                print('no face detected!')
        

        e_v1_map = self._get_2d_error_hitogram(pitch_list,yaw_list,error_value=e_v1_list,resolution=resolution)
        e_h1_map = self._get_2d_error_hitogram(pitch_list,yaw_list,error_value=e_h1_list,resolution=resolution)
        e_total_map = e_v1_map + e_h1_map

        self._polt_2d_map_with_colorbar(e_v1_map,title='vertical_error gt',interpolation='bilinear')
        self._polt_2d_map_with_colorbar(e_h1_map,title='horizontal_error gt',interpolation='bilinear')
        self._polt_2d_map_with_colorbar(e_total_map,title='total_error gt',interpolation='bilinear')

    def full_evaluation(self,dataset_dir,subjects_included,save_root,sample_size = 200, print_freq = 10):

        subjects_index = {}
        subject_metrics = {}
        sample_per_subject = sample_size//len(subjects_included) 
        print(f'Total sample size: {sample_size}, per subject sample size:{sample_per_subject} ')
        for subject_id in subjects_included:
            
            file_path = os.path.join(dataset_dir,f'processed_test_{subject_id}')
            hdf_file = h5py.File(file_path, 'r', swmr=True)
            n = hdf_file['cam_index'].shape[0]

            if sample_per_subject > n:
                print('too many sample selected!')
            subjects_index[subject_id] = random.sample(range(0,n),min(sample_per_subject,n))

            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            hdf_file.close()

            subject_metrics[subject_id] = self.evaluation_subject(dataset_dir,f'processed_test_{subject_id}',save_root=os.path.join(save_root,'eval_'+subject_id),\
                                            print_freq=print_freq,\
                                            indexs=subjects_index[subject_id])
        
        metrics_dict = {
        'SSIM':0,
        'PSNR':0,
        'LPIPS':0,
        'L1_loss':0,
        'vertical_error':0,
        'horizontal_error':0,
        'vertical_error_ref':0,
        'horizontal_error_ref':0,
        'vertical_pred_gap':0,
        'horizontal_pred_gap':0,
        'sample_size':0
        }
        total_count = 0
        for subject_id,metrics in subject_metrics.items():
            for key in metrics.keys():
                if key != 'sample_size':
                    metrics_dict[key] += np.sum(metrics[key])
            total_count += metrics['sample_size']

        t = PrettyTable(['Metrics', 'Value'])
        for k,v in metrics_dict.items():
            if k != 'sample_size':
                mean_value = np.sum(v)/total_count
                t.add_row([k,mean_value])
        t.add_row(['sample_size',total_count])
    
        print(t)

        with open(os.path.join(save_root,'error_table.txt'), 'w') as w:
            w.write(f'Model_name: {self.model_path}\n')
            w.write(str(t))
        



    def evaluation_subject(self,input_dir,subjects_name,save_root,print_freq = 10, indexs=None):
        if os.path.exists(save_root):
            shutil.rmtree(save_root)
            os.mkdir(save_root)
        else:
            os.mkdir(save_root)
        hdf_file_path = os.path.join(input_dir,subjects_name)
        self.hdf = h5py.File(hdf_file_path, 'r', swmr=True)
        sample_size = self.hdf['cam_index'].shape[0]
        output_dict = {
        'SSIM':[0]*sample_size,
        'PSNR':[0]*sample_size,
        'LPIPS':[0]*sample_size,
        'L1_loss':[0]*sample_size,
        'vertical_error':[0]*sample_size,
        'horizontal_error':[0]*sample_size,
        'vertical_error_ref':[0]*sample_size,
        'horizontal_error_ref':[0]*sample_size,
        'vertical_pred_gap':[0]*sample_size,
        'horizontal_pred_gap':[0]*sample_size,
        'sample_size':0
        }
        count = 0
        if indexs is None:
            loop_bar = range(sample_size)
        else:
            loop_bar = indexs


        for image_index in tqdm(loop_bar,leave=True,position=2):
            self.load_data(hdf_file_path,image_index,cam_idx=-1) ##inlcude all camera views
            self.perform_fitting()

            rendered_results,cam_info,face_gaze = self.render_utils.render_face_with_gaze(self.net,self.res_code_info,face_gaze=self.gaze_tensor,scale_factor = 1,gaze_dim = self.eye_gaze_dim,cam_info=self.cam_info)
            
            # inmat_np = torch.linalg.inv(self.res_cam_info['batch_inv_inmats']).detach().cpu().numpy()
            # inmat_np = inmat_np.reshape((3,3))
            # distortion_np = np.zeros([1,5])
            # inmat_np[0,0] *=10; inmat_np[1,1] *=10; inmat_np[0,2] *=10; inmat_np[1,2] *=10
            # cam_info = {'camera_matrix':inmat_np, 'camera_distortion':distortion_np}
            
            gt_img = (self.img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            head_mask = self._update_head_mask(rendered_results)
            gt_img[head_mask<0.5]=255
            rendered_results[head_mask<0.5]=255

            eval_metrics = calc_eval_metrics(rendered_results.copy(),gt_img.copy(),vis=False,L1_loss=True)
            
            #visualiza difference map
            difference = cv2.subtract(gt_img,rendered_results)
            Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
            difference[mask != 255] = [0, 0, 255]
            diff_rendered = rendered_results.copy()
            diff_rendered[mask != 255] = [0, 0, 255]

            if self.vis_vect:
                rendered_results,e_v2,e_h2,pred_gaze = self.render_utils.render_gaze_vect(rendered_results,cam_info,face_gaze)
                gt_img,e_v1,e_h1,pred_gaze_gt = self.render_utils.render_gaze_vect(gt_img,cam_info=cam_info,face_gaze=self.gaze_tensor)
                

                if e_v2 == -1 or e_v1 == -1:
                    # no face detected in estimator
                    print('skip this sample!')
                    continue
                    
                output_dict['vertical_error'][count] = e_v2
                output_dict['horizontal_error'][count] = e_h2
                output_dict['vertical_error_ref'][count] = abs(e_v1 - e_v2)
                output_dict['horizontal_error_ref'][count] = abs(e_h1 - e_h2)
                output_dict['vertical_pred_gap'][count] = abs(pred_gaze[0] - pred_gaze_gt[0])
                output_dict['horizontal_pred_gap'][count] = abs(pred_gaze[1] - pred_gaze_gt[1])

                        ## compute gaze gap between label and estimated
            for k,v in eval_metrics.items():
                output_dict[k][count] = v
            count += 1

            if count % print_freq == 0:
                res_img = np.concatenate([gt_img, rendered_results,difference,diff_rendered], axis=1)
                cv2.imwrite(os.path.join(save_root,f'testing_image{count}.png'),res_img)
            
        output_dict['sample_size'] = count
        with open(os.path.join(save_root,f"{subjects_name}_eval_metrics.pkl"),'wb') as file:
            pickle.dump(output_dict,file)

        ##visualization

        t = PrettyTable(['Metrics', 'Value'])
        for k,v in output_dict.items():
            if k != 'sample_size':
                mean_value = np.sum(v)/count
                t.add_row([k,mean_value])
        t.add_row(['sample_size',count])
        print(t)

        with open(os.path.join(save_root,'error_table.txt'), 'w') as w:
            w.write(str(t))
        
        return output_dict


    def render_face_gaze_and_ground_truth_image(self,hdf_file_path,image_index,save_root,vis_vect=True):
        '''
        compare the ground truth image and rendered image given the same gaze label
        '''

        self.load_data(hdf_file_path,image_index,cam_idx=-1)
        self.perform_fitting()
        rendered_results,cam_info,face_gaze = self.render_utils.render_face_with_gaze(self.net,self.res_code_info,face_gaze=self.gaze_tensor,scale_factor = 1,gaze_dim = self.eye_gaze_dim,cam_info=self.cam_info)
        if self.vis_vect:
            rendered_results,e_v2,e_h2,_ = self.render_utils.render_gaze_vect(rendered_results,cam_info,face_gaze)

        inmat_np = torch.linalg.inv(self.res_cam_info['batch_inv_inmats']).detach().cpu().numpy()
        inmat_np = inmat_np.reshape((3,3))
        distortion_np = np.zeros([1,5])
        inmat_np[0,0] *=10; inmat_np[1,1] *=10; inmat_np[0,2] *=10; inmat_np[1,2] *=10
        cam_info = {'camera_matrix':inmat_np, 'camera_distortion':distortion_np}
        
        gt_img = (self.img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img,e_v1,e_h1,_ = self.render_utils.render_gaze_vect(gt_img,cam_info=cam_info,face_gaze=self.gaze_tensor)

        uncropped_gt_img = cv2.cvtColor(self.uncropped_gt_image, cv2.COLOR_BGR2RGB)
        uncropped_gt_img,e_v0,e_h0,_ = self.render_utils.render_gaze_vect(uncropped_gt_img,cam_info=cam_info,face_gaze=self.gaze_tensor)

        if e_h0 < 0 or e_v0 < 0 or e_h1 < 0 or e_v1 < 0 or e_h2 < 0 or e_v2 < 0 :
            # any face not detected
            return
        self.error_ave_dict['ave_e_h0']+= [e_h0]
        self.error_ave_dict['ave_e_v0']+= [e_v0]
        self.error_ave_dict['ave_e_h1']+= [e_h1]
        self.error_ave_dict['ave_e_v1']+= [e_v1]
        self.error_ave_dict['ave_e_h2']+= [e_h2]
        self.error_ave_dict['ave_e_v2']+= [e_v2]


        res_img = np.concatenate([uncropped_gt_img,gt_img, rendered_results], axis=1)

        cv2.imwrite(os.path.join(save_root,f'gt_and_rendered_image{image_index}.png'),res_img)
        print('#######current average error##########')
        for key,value in self.error_ave_dict.items():
            print(" %s : %.5f" % (key , stats.trim_mean(value, 0)))

    def _display_current_rendered_image(self,pred_dict,img_tensor):
        coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
        coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        gt_img = (img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        res_img = np.concatenate([gt_img, coarse_fg_rgb], axis=1)


        cv2.imshow('current rendering', res_img)
        cv2.waitKey(0) 
        #closing all open windows 
        cv2.destroyAllWindows() 
    
    def _polt_2d_map_with_colorbar(self, data,extent=[180/np.pi,-180/np.pi,-180/np.pi,180/np.pi],title='temp',interpolation = 'antialiased',save_root = ''):
        #data = np.arange(100, 0, -1).reshape(10, 10)
        fig, ax = plt.subplots()
        plt.xlabel('yaw angle (in degree)')
        plt.ylabel('pitch angle (in degree)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(data, extent=extent,interpolation=interpolation)
        ax.plot([-30,-30,43,43,-30],[-30,15,15,-30,-30])  #pitch [-28.64,14.323], yaw [-28.64,42.97]
                
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_title(title)
        if save_root:
            plt.savefig(os.path.join(save_root,title))
        else:
            plt.show()
        

    def _get_2d_error_hitogram(self,x,y,error_value,range=[[-1,1],[-1,1]],resolution=21):
        '''
        get 2d histogram for error distribution in grid gaze space
        '''
        H, _, _ = np.histogram2d(x,y,bins=resolution,range=range,density=False,weights=error_value) #total sum
        H1, _, _ = np.histogram2d(x,y,bins=resolution,range=range,density=False) # bin_count
        ave = np.divide(H,H1,out=np.zeros_like(H),where=H1!=0)

        ave = np.flipud(ave)
        ave = np.fliplr(ave)
        return ave

    def _update_head_mask(self,rendered_results):
        rendered_results[self.head_mask_np<0.5]=0.0
        dif_mask = cv2.inRange(rendered_results,np.array([250,250,250]),np.array([255,255,255]))
        overall_mask = cv2.bitwise_and(255 - dif_mask,self.head_mask_np)
        return overall_mask
        

def str2bool(v):
    return v.lower() in ('true', '1')

if __name__ == "__main__":
    torch.manual_seed(45)  # cpu
    torch.cuda.manual_seed(55)  # gpu
    np.random.seed(65)  # numpy
    random.seed(75)  # random and transforms
    # torch.backends.cudnn.deterministic = True  # cudnn
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(description='a framework for fitting a single image using HeadNeRF')
    parser.add_argument("--config_file_path", type=str,help='Path to load config file')
    args = parser.parse_args()

    config_path = args.config_file_path    
    config = load_config(config_path)
    train_config = config["training_config"]
    eval_config = config["eval_config"]
    train_config.add("base_opt",config["base_opt"])
    #####
    model_path = eval_config.model_path
    save_root = eval_config.save_root
    hdf_file = eval_config.hdf_file
    image_index = eval_config.image_index
    gaze_feat_dim = eval_config.gaze_dim
    scale_factor = eval_config.eye_gaze_scale_factor
    vis_vect = eval_config.vis_gaze_vect
    use_6D_rotattion = eval_config.D6_rotation
    model_name = eval_config.model_name
    subject_included = eval_config.subject_included
    cam_included = eval_config.cam_included

    #####
    tt = FittingImage(model_path, save_root, gpu_id=0,config=train_config,include_eye_gaze=True,eye_gaze_dim=gaze_feat_dim,gaze_scale_factor=scale_factor,vis_vect=vis_vect,D6_rotation=use_6D_rotattion,model_name=model_name)
    # # #tt.fitting_single_images(hdf_file,image_index, save_root)

    # for image_index in range(50):
    #     tt.render_face_gaze_and_ground_truth_image(hdf_file,image_index,save_root='experiment_document/gaze_and_gt_image/')
    if eval_config.mode == 'gridsample_face_gaze':
        if eval_config.choice == 'multi_cam':
            subject = subject_included[0]
            hdf_file = os.path.join(hdf_file,f'processed_{subject}')
            for cam_id in cam_included:
                subfolder = f'gridsample_images/cam{cam_id}'
                tt.gridsample_face_gaze(hdf_file,image_index,save_root=os.path.join(save_root,subfolder),resolution=eval_config.resolution,\
                                        print_freq=eval_config.print_freq,\
                                        cam_index=cam_id) #grid sample gaze space
        elif eval_config.choice == 'multi_sub':
            cam_id = cam_included[0]
            hdf_file_temp = hdf_file
            for subject_id in subject_included:
                hdf_file = os.path.join(hdf_file_temp,f'processed_{subject_id}')
                subfolder = f'gridsample_images/{subject_id}'
                tt.gridsample_face_gaze(hdf_file,image_index,save_root=os.path.join(save_root,subfolder),resolution=eval_config.resolution,\
                                        print_freq=eval_config.print_freq,\
                                        cam_index=cam_id) #grid sample gaze space

    if eval_config.mode == 'gridsample_ground_truth_gaze':
        tt.sample_face_gaze_ground_truth_image(hdf_file,image_sample_num=eval_config.sample_size,resolution=eval_config.resolution) ##sample gt images and bilinear interpolate
        
    ## 10 cam for 10 subject
    # for subject_id in subject_included:
    #     print(f'Sampling {subject_id} !')
    #     hdf_file = 'XGaze_data/processed_data_10cam/processed_' + subject_id
    #     save_root_base = os.path.join('experiment_document/gridsample_images/',subject_id)
    #     #tt.gridsample_face_gaze(hdf_file,image_index,save_root=save_root,resolution=21,print_freq=10)
    #     for cam_id in cam_included:
    #         save_root = os.path.join(save_root_base,str(cam_id))
    #         try:
    #             tt.gridsample_face_gaze(hdf_file,image_index,save_root=save_root,resolution=11,print_freq=1,cam_index=cam_id)
    #         except:
    #             pass

    if eval_config.mode == 'full_evaluation':
        sub_folder = 'evaluation_output/full_evaluation'
        tt.full_evaluation(dataset_dir=hdf_file,\
                            subjects_included=subject_included,\
                            save_root=os.path.join(save_root,sub_folder),sample_size=eval_config.sample_size,print_freq=eval_config.print_freq)

    if eval_config.mode == 'render_gaze_redirction':
        sub_folder = 'gaze_redirection_output'
        for subject in subject_included:
            hdf_file =os.path.join(hdf_file,f'processed_test_{subject}')
            tt.render_gaze_redirection_gif(hdf_file_path=hdf_file,
                                            image_index=image_index,
                                            save_root=os.path.join(save_root,sub_folder,subject),num_frames=10)

