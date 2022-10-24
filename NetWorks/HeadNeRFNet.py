from telnetlib import GA
from numpy import average
import torch
import torch.nn as nn
from .utils import Embedder, CalcRayColor, GenSamplePoints, FineSample
from .models import MLPforNeRF,MLPforGaze,MLPforHeadNeRF_Gaze
from NetWorks.neural_renderer import NeuralRenderer
import torch.nn.functional as F
from HeadNeRFOptions import BaseOptions
from .Coordmap import AddCoords

###add gaze branch feature here, feat size output torch.Size([1, 256, 32, 32])
class HeadNeRFNet_Gaze(nn.Module):
    def __init__(self, opt: BaseOptions, include_vd, hier_sampling, eye_gaze_dim=64,include_vp=True) -> None:
        super().__init__()

        self.hier_sampling = hier_sampling
        self.include_vd = include_vd
        self.eye_gaze_dim = eye_gaze_dim
        self.include_vp = include_vp
        self._build_info(opt)
        self._build_tool_funcs()
        self.add_coord = AddCoords()
        

    def _build_info(self, opt: BaseOptions):
        
        self.num_sample_coarse = opt.num_sample_coarse
        self.num_sample_fine = opt.num_sample_fine

        self.vp_n_freqs = 10
        self.include_input_for_vp_embeder = True

        self.vd_n_freqs = 4
        self.include_input_for_vd_embeder = True

        self.mlp_h_channel = opt.mlp_hidden_nchannels

        self.auxi_shape_code_dims = opt.auxi_shape_code_dims
        self.auxi_appea_code_dims = opt.auxi_appea_code_dims
        
        self.base_shape_code_dims = opt.iden_code_dims + opt.expr_code_dims
        self.base_appea_code_dims = opt.text_code_dims + opt.illu_code_dims
        
        self.featmap_size = opt.featmap_size
        self.featmap_nc = opt.featmap_nc        # num_channel
        self.pred_img_size = opt.pred_img_size
        self.opt = opt
        

    def _build_tool_funcs(self):

        vp_channels = self.base_shape_code_dims
        vp_channels += self.vp_n_freqs * 6 + 3 if self.include_input_for_vp_embeder else self.vp_n_freqs * 6
        
        self.vp_encoder = Embedder(N_freqs=self.vp_n_freqs, include_input=self.include_input_for_vp_embeder)
        
        vd_channels = self.base_appea_code_dims
        if self.include_vd:
            tv = self.vd_n_freqs * 6 + 3 if self.include_input_for_vd_embeder else self.vd_n_freqs * 6
            vd_channels += tv
            self.vd_encoder = Embedder(N_freqs=self.vd_n_freqs, include_input=self.include_input_for_vd_embeder)

                
        
        self.sample_func = GenSamplePoints(self.opt)
        
        if self.hier_sampling:
            self.fine_samp_func = FineSample(self.opt)
        
        # self.fg_CD_predictor = MLPforNeRF(vp_channels=vp_channels, vd_channels=vd_channels, 
        #                                             h_channel=self.mlp_h_channel, res_nfeat=self.featmap_nc)

        if self.hier_sampling:
            self.fine_fg_CD_predictor = MLPforNeRF(vp_channels=vp_channels, vd_channels=vd_channels, 
                                                    h_channel=self.mlp_h_channel, res_nfeat=self.featmap_nc)

        self.calc_color_func = CalcRayColor()
        self.neural_render = NeuralRenderer(bg_type=self.opt.bg_type, feat_nc=self.featmap_nc,  out_dim=3, final_actvn=True, 
                                                min_feat=32, featmap_size=self.featmap_size, img_size=self.pred_img_size)
        
        #self.gaze_feat_predictor = MLPforGaze(input_channels=1 + self.eye_gaze_dim + 2, h_channel = 256, res_nfeat=self.featmap_nc)
        if self.include_vp:
            gaze_channels = 3+self.eye_gaze_dim+63
        else:
            gaze_channels = 3+self.eye_gaze_dim

        self.fg_CD_predictor= MLPforHeadNeRF_Gaze(vp_channels=vp_channels,vd_channels=vd_channels,
                                                    gaze_channels=gaze_channels,h_channel=self.mlp_h_channel,res_nfeat=self.featmap_nc)

    def eye_gaze_branch(self,input_gaze,eye_mask_tensor,FGvp_embedder,include_vp = False,use_temp=False):
        #coord_map = get_coord_maps(size = self.featmap_size).repeat(batch_size,1,1,1)
        if use_temp:
            input_gaze = torch.zeros_like(input_gaze)

        batch_size = eye_mask_tensor.size(0)
        img_size = eye_mask_tensor.size(-1)
        
        Gaze_feat = eye_mask_tensor.view(batch_size,1,img_size,img_size)#eye_mask_tensor: torch.Size([2, 1, 512, 512])
        input_gaze = input_gaze.view(batch_size,self.eye_gaze_dim,1,1).repeat(1,1,self.featmap_size,self.featmap_size)
        avg_pool = nn.AvgPool2d(16, stride=16,padding=0)
        Gaze_feat = avg_pool(Gaze_feat.float())#eye_mask_tensor: torch.Size([2, 1, 32, 32])
        Gaze_feat = torch.cat([Gaze_feat,input_gaze],dim = 1) #torch.Size([2, 1 + 2n, 32, 32] 
        Gaze_feat = self.add_coord(Gaze_feat) #add coord map -> torch.Size([2, 1 + 2n + 2, 32, 32] 

        # FGvp_embedder:([1, 3, 1024, 64])
        Gaze_feat = Gaze_feat.view(batch_size,-1,self.featmap_size * self.featmap_size).unsqueeze(-1).repeat(1,1,1,64) #torch.Size([2, 1 + 2n, 1024,64] 
        if include_vp:
            Gaze_feat = torch.cat([FGvp_embedder, Gaze_feat], dim=1)
        #Gaze_feat = self.gaze_feat_predictor(Gaze_feat) ##eye_mask_tensor: torch.Size([2, 256, 32, 32] 
        return Gaze_feat # (batch_size,feat_dim,map_pixel,point_num)
        
    def calc_color_with_code(self, fg_vps, shape_code, appea_code, FGvp_embedder, 
                             FGvd_embedder, FG_zdists, FG_zvals, fine_level,input_gaze,eye_mask_tensor,for_train):
        
        ori_FGvp_embedder = torch.cat([FGvp_embedder, shape_code], dim=1) #torch.Size([1, 242, 1024, 64]) position encoder and id+exp
        Gaze_embedder = self.eye_gaze_branch(input_gaze,eye_mask_tensor,FGvp_embedder,include_vp=self.include_vp)

        if for_train:
            Temp_Gaze_embedder = self.eye_gaze_branch(input_gaze,eye_mask_tensor,FGvp_embedder,include_vp=self.include_vp,use_temp=True)
        else:
            Temp_Gaze_embedder = None

        if self.include_vd:
            ori_FGvd_embedder = torch.cat([FGvd_embedder, appea_code], dim=1)
        else:
            ori_FGvd_embedder = appea_code # torch.Size([1, 127, 1024, 64])
        
        ##for each pixel(1024) we sample 64 points and each points we predict F(x) (256) and density (1)
        if fine_level:
            FGmlp_FGvp_rgb, FGmlp_FGvp_density = self.fine_fg_CD_predictor(ori_FGvp_embedder, ori_FGvd_embedder)
        else:
            #FGmlp_FGvp_rgb, FGmlp_FGvp_density = self.fg_CD_predictor(ori_FGvp_embedder, ori_FGvd_embedder)#neural radiance field torch.Size([1, 256, 1024, 64]),torch.Size([1, 1, 1024, 64])
            FGmlp_FGvp_rgb, FGmlp_FGvp_density,FGmlp_rgb_temp,FGmlp_density_temp = self.fg_CD_predictor(ori_FGvp_embedder, ori_FGvd_embedder,Gaze_embedder,Temp_Gaze_embedder,for_train=for_train)
        
        ##feature map I_f(256x32x32) is achieved by volumn rendering strategy  
        fg_feat, bg_alpha, batch_ray_depth, ori_batch_weight = self.calc_color_func(fg_vps, FGmlp_FGvp_rgb,
                                                                                    FGmlp_FGvp_density,
                                                                                    FG_zdists,
                                                                                    FG_zvals) #torch.Size([1, 256, 1024]), torch.Size([1, 1, 1024]), torch.Size([1, 1, 1024]), torch.Size([1, 1, 1024, 64])

        #Assume that the gaze will influence the density, so we add the feat map after the integration
        #add gaze branch feature here, feat size torch.Size([1, 256, 1024, 64])

        batch_size = fg_feat.size(0)
        fg_feat = fg_feat.view(batch_size, self.featmap_nc, self.featmap_size, self.featmap_size) #torch.Size([1, 256, 32,32])

        bg_alpha = bg_alpha.view(batch_size, 1, self.featmap_size, self.featmap_size)# torch.Size([1, 1, 32, 32])

        bg_featmap = self.neural_render.get_bg_featmap() #torch.Size([1, 256, 32, 32])
        bg_img = self.neural_render(bg_featmap) #torch.Size([1, 3, 512, 512])

        ##Map feature map I_f(256x32x32) to image I (3x256x256)
        merge_featmap = fg_feat + bg_alpha * bg_featmap  #torch.Size([1, 256, 32, 32])
        merge_img = self.neural_render(merge_featmap) #torch.Size([1, 3, 512, 512])

        res = {
            "merge_img": merge_img, 
            "bg_img": bg_img,
        }
        # #####Template image rendering##########
        if for_train:
            fg_feat_temp, bg_alpha_temp, batch_ray_depth_temp, ori_batch_weight_temp = self.calc_color_func(fg_vps, FGmlp_rgb_temp,
                                                                                FGmlp_density_temp,
                                                                                FG_zdists,
                                                                                FG_zvals)
            fg_feat_temp = fg_feat_temp.view(batch_size, self.featmap_nc, self.featmap_size, self.featmap_size) #torch.Size([1, 256, 32,32]) 
            bg_alpha_temp = bg_alpha_temp.view(batch_size, 1, self.featmap_size, self.featmap_size)# torch.Size([1, 1, 32, 32])    
            
            bg_featmap = self.neural_render.get_bg_featmap() #torch.Size([1, 256, 32, 32])

            template_featmap = fg_feat_temp + bg_alpha_temp * bg_featmap  #torch.Size([1, 256, 32, 32])
            template_img = self.neural_render(template_featmap) 
               
            res["template_img"] = template_img
        #######################################
        
        return res, ori_batch_weight


    def _forward(
            self, 
            for_train, 
            batch_xy, batch_uv, 
            bg_code, shape_code, appea_code, 
            batch_Rmats, batch_Tvecs, batch_inv_inmats, dist_expr,**kwargs
        ):
        
        # cam - to - world
        batch_size, tv, n_r = batch_xy.size() #torch.Size([1, 2, 1024])
        assert tv == 2
        assert bg_code is None
        fg_sample_dict = self.sample_func(batch_xy, batch_Rmats, batch_Tvecs, batch_inv_inmats, for_train) #dict_keys(['pts', 'dirs', 'zvals', 'z_dists', 'batch_ray_o', 'batch_ray_d', 'batch_ray_l'])
        fg_vps = fg_sample_dict["pts"]  #torch.Size([1, 3, 1024, 64])
        fg_dirs = fg_sample_dict["dirs"] #torch.Size([1, 3, 1024, 64])

        FGvp_embedder = self.vp_encoder(fg_vps) #torch.Size([1, 3, 1024, 64]) -> torch.Size([1, 63, 1024, 64])
        
        if self.include_vd:
            FGvd_embedder = self.vd_encoder(fg_dirs)
        else:
            FGvd_embedder = None

        FG_zvals = fg_sample_dict["zvals"]
        FG_zdists = fg_sample_dict["z_dists"]
        
        cur_shape_code = shape_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, self.num_sample_coarse) #torch.Size([1, 179]) -> torch.Size([1, 179, 1024, 64])
        cur_appea_code = appea_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, self.num_sample_coarse) #torch.Size([1, 127]) -> torch.Size([1, 127, 1024, 64])


        input_gaze = kwargs["input_gaze"].cuda()
        eye_mask_tensor = kwargs["eye_mask"].cuda()
        c_ori_res, batch_weight = self.calc_color_with_code(
            fg_vps, cur_shape_code, cur_appea_code, FGvp_embedder, FGvd_embedder, FG_zdists, FG_zvals, fine_level = False,
             input_gaze=input_gaze, eye_mask_tensor=eye_mask_tensor,for_train=for_train
        )
        
        res_dict = {
            "coarse_dict":c_ori_res,
        }

        if self.hier_sampling:
            
            fine_sample_dict = self.fine_samp_func(batch_weight, fg_sample_dict, for_train)
            fine_fg_vps = fine_sample_dict["pts"]
            fine_fg_dirs = fine_sample_dict["dirs"]

            fine_FGvp_embedder = self.vp_encoder(fine_fg_vps)
            if self.include_vd:
                fine_FGvd_embedder = self.vd_encoder(fine_fg_dirs)
            else:
                fine_FGvd_embedder = None

            fine_FG_zvals = fine_sample_dict["zvals"]
            fine_FG_zdists = fine_sample_dict["z_dists"]
            
            num_sample = self.num_sample_coarse + self.num_sample_fine
            
            cur_shape_code = shape_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, num_sample)
            cur_appea_code = appea_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, num_sample)
            
            f_ori_res, _= self.calc_color_with_code(
               cur_shape_code, cur_appea_code, fine_FGvp_embedder, fine_FGvd_embedder, fine_FG_zdists, fine_FG_zvals, 
               fine_level=True
            )
            
            res_dict["fine_dict"] = f_ori_res

        return res_dict
    

    def forward(
                self,
                mode, 
                batch_xy, batch_uv,
                bg_code, shape_code, appea_code, 
                batch_Rmats, batch_Tvecs, batch_inv_inmats, dist_expr = False, **kwargs
        ):
        assert mode in ["train", "test"]
        return self._forward(
            mode == "train",
            batch_xy, batch_uv,
            bg_code, shape_code, appea_code, 
            batch_Rmats, batch_Tvecs, batch_inv_inmats, dist_expr,**kwargs
        )

############original version of HeadNeRF################################
class HeadNeRFNet(nn.Module):
    def __init__(self, opt: BaseOptions, include_vd, hier_sampling,include_gaze=True,eye_gaze_dim=2) -> None:
        super().__init__()

        self.hier_sampling = hier_sampling
        self.include_vd = include_vd
        self.include_gaze = include_gaze
        self.eye_gaze_dim = eye_gaze_dim
        self._build_info(opt)
        self._build_tool_funcs()
        

    def _build_info(self, opt: BaseOptions):
        
        self.num_sample_coarse = opt.num_sample_coarse
        self.num_sample_fine = opt.num_sample_fine

        self.vp_n_freqs = 10
        self.include_input_for_vp_embeder = True

        self.vd_n_freqs = 4
        self.include_input_for_vd_embeder = True

        self.mlp_h_channel = opt.mlp_hidden_nchannels

        self.auxi_shape_code_dims = opt.auxi_shape_code_dims
        self.auxi_appea_code_dims = opt.auxi_appea_code_dims
        
        self.base_shape_code_dims = opt.iden_code_dims + opt.expr_code_dims
        self.base_appea_code_dims = opt.text_code_dims + opt.illu_code_dims
        
        self.featmap_size = opt.featmap_size
        self.featmap_nc = opt.featmap_nc        # num_channel
        self.pred_img_size = opt.pred_img_size
        self.opt = opt
        

    def _build_tool_funcs(self):

        vp_channels = self.base_shape_code_dims
        vp_channels += self.vp_n_freqs * 6 + 3 if self.include_input_for_vp_embeder else self.vp_n_freqs * 6
        if self.include_gaze:
            vp_channels += self.eye_gaze_dim
        
        self.vp_encoder = Embedder(N_freqs=self.vp_n_freqs, include_input=self.include_input_for_vp_embeder)
        
        vd_channels = self.base_appea_code_dims
        if self.include_vd:
            tv = self.vd_n_freqs * 6 + 3 if self.include_input_for_vd_embeder else self.vd_n_freqs * 6
            vd_channels += tv
            self.vd_encoder = Embedder(N_freqs=self.vd_n_freqs, include_input=self.include_input_for_vd_embeder)
        if self.include_gaze:
            vd_channels += 0 #self.eye_gaze_dim
                
        
        self.sample_func = GenSamplePoints(self.opt)
        
        if self.hier_sampling:
            self.fine_samp_func = FineSample(self.opt)
        
        self.fg_CD_predictor = MLPforNeRF(vp_channels=vp_channels, vd_channels=vd_channels, 
                                                    h_channel=self.mlp_h_channel, res_nfeat=self.featmap_nc)
        if self.hier_sampling:
            self.fine_fg_CD_predictor = MLPforNeRF(vp_channels=vp_channels, vd_channels=vd_channels, 
                                                    h_channel=self.mlp_h_channel, res_nfeat=self.featmap_nc)

        self.calc_color_func = CalcRayColor()
        self.neural_render = NeuralRenderer(bg_type=self.opt.bg_type, feat_nc=self.featmap_nc,  out_dim=3, final_actvn=True, 
                                                min_feat=32, featmap_size=self.featmap_size, img_size=self.pred_img_size)
        
        
    def calc_color_with_code(self, fg_vps, shape_code, appea_code, FGvp_embedder, 
                             FGvd_embedder, FG_zdists, FG_zvals, fine_level):
        
        ori_FGvp_embedder = torch.cat([FGvp_embedder, shape_code], dim=1) #torch.Size([1, 242, 1024, 64]) position encoder and id+exp
        
        if self.include_vd:
            ori_FGvd_embedder = torch.cat([FGvd_embedder, appea_code], dim=1)
        else:
            ori_FGvd_embedder = appea_code # torch.Size([1, 127, 1024, 64])
        
        ##for each pixel(1024) we sample 64 points and each points we predict F(x) (256) and density (1)
        if fine_level:
            FGmlp_FGvp_rgb, FGmlp_FGvp_density = self.fine_fg_CD_predictor(ori_FGvp_embedder, ori_FGvd_embedder)
        else:
            FGmlp_FGvp_rgb, FGmlp_FGvp_density = self.fg_CD_predictor(ori_FGvp_embedder, ori_FGvd_embedder)#neural radiance field torch.Size([1, 256, 1024, 64]),torch.Size([1, 1, 1024, 64])

        ##feature map I_f(256x32x32) is achieved by volumn rendering strategy  
        fg_feat, bg_alpha, batch_ray_depth, ori_batch_weight = self.calc_color_func(fg_vps, FGmlp_FGvp_rgb,
                                                                                    FGmlp_FGvp_density,
                                                                                    FG_zdists,
                                                                                    FG_zvals) #torch.Size([1, 256, 1024]), torch.Size([1, 1, 1024]), torch.Size([1, 1, 1024]), torch.Size([1, 1, 1024, 64])
        
        ###add gaze branch feature here, feat size torch.Size([1, 256, 32, 32])
        #TODO
        batch_size = fg_feat.size(0)
        fg_feat = fg_feat.view(batch_size, self.featmap_nc, self.featmap_size, self.featmap_size) #torch.Size([1, 256, 32, 32])

        bg_alpha = bg_alpha.view(batch_size, 1, self.featmap_size, self.featmap_size)# torch.Size([1, 1, 32, 32])

        bg_featmap = self.neural_render.get_bg_featmap() #torch.Size([1, 256, 32, 32])
        bg_img = self.neural_render(bg_featmap) #torch.Size([1, 3, 512, 512])

        ##Map feature map I_f(256x32x32) to image I (3x256x256)
        merge_featmap = fg_feat + bg_alpha * bg_featmap #torch.Size([1, 256, 32, 32])
        merge_img = self.neural_render(merge_featmap) #torch.Size([1, 3, 512, 512])

        res = {
            "merge_img": merge_img, 
            "bg_img": bg_img
        }
        
        return res, ori_batch_weight


    def _forward(
            self, 
            for_train, 
            batch_xy, batch_uv, 
            bg_code, shape_code, appea_code, 
            batch_Rmats, batch_Tvecs, batch_inv_inmats, dist_expr
        ):
        
        # cam - to - world
        batch_size, tv, n_r = batch_xy.size() #torch.Size([1, 2, 1024])
        assert tv == 2
        assert bg_code is None
        fg_sample_dict = self.sample_func(batch_xy, batch_Rmats, batch_Tvecs, batch_inv_inmats, for_train) #dict_keys(['pts', 'dirs', 'zvals', 'z_dists', 'batch_ray_o', 'batch_ray_d', 'batch_ray_l'])
        fg_vps = fg_sample_dict["pts"]  #torch.Size([1, 3, 1024, 64])
        fg_dirs = fg_sample_dict["dirs"] #torch.Size([1, 3, 1024, 64])

        FGvp_embedder = self.vp_encoder(fg_vps) #torch.Size([1, 3, 1024, 64]) -> torch.Size([1, 63, 1024, 64])
        
        if self.include_vd:
            FGvd_embedder = self.vd_encoder(fg_dirs)
        else:
            FGvd_embedder = None

        FG_zvals = fg_sample_dict["zvals"]
        FG_zdists = fg_sample_dict["z_dists"]
        
        cur_shape_code = shape_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, self.num_sample_coarse) #torch.Size([1, 179]) -> torch.Size([1, 179, 1024, 64])
        cur_appea_code = appea_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, self.num_sample_coarse) #torch.Size([1, 127]) -> torch.Size([1, 127, 1024, 64])

        c_ori_res, batch_weight = self.calc_color_with_code(
            fg_vps, cur_shape_code, cur_appea_code, FGvp_embedder, FGvd_embedder, FG_zdists, FG_zvals, fine_level = False
        )
        
        res_dict = {
            "coarse_dict":c_ori_res,
        }

        if self.hier_sampling:
            
            fine_sample_dict = self.fine_samp_func(batch_weight, fg_sample_dict, for_train)
            fine_fg_vps = fine_sample_dict["pts"]
            fine_fg_dirs = fine_sample_dict["dirs"]

            fine_FGvp_embedder = self.vp_encoder(fine_fg_vps)
            if self.include_vd:
                fine_FGvd_embedder = self.vd_encoder(fine_fg_dirs)
            else:
                fine_FGvd_embedder = None

            fine_FG_zvals = fine_sample_dict["zvals"]
            fine_FG_zdists = fine_sample_dict["z_dists"]
            
            num_sample = self.num_sample_coarse + self.num_sample_fine
            
            cur_shape_code = shape_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, num_sample)
            cur_appea_code = appea_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, num_sample)
            
            f_ori_res, _= self.calc_color_with_code(
               cur_shape_code, cur_appea_code, fine_FGvp_embedder, fine_FGvd_embedder, fine_FG_zdists, fine_FG_zvals, 
               fine_level=True
            )
            
            res_dict["fine_dict"] = f_ori_res

        return res_dict
    

    def forward(
                self,
                mode, 
                batch_xy, batch_uv,
                bg_code, shape_code, appea_code, 
                batch_Rmats, batch_Tvecs, batch_inv_inmats, dist_expr = False, **kwargs
        ):
        assert mode in ["train", "test"]
        return self._forward(
            mode == "train",
            batch_xy, batch_uv,
            bg_code, shape_code, appea_code, 
            batch_Rmats, batch_Tvecs, batch_inv_inmats, dist_expr
        )
