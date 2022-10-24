
from tkinter.tix import X_REGION
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Coordmap import AddCoords
def _xavier_init(net_layer):
    """
    Performs the Xavier weight initialization of the net layer.
    """
    torch.nn.init.xavier_uniform_(net_layer.weight.data)
    

class MLPforNeRF(nn.Module):

    def __init__(self, vp_channels, vd_channels, n_layers = 8, h_channel = 256, res_nfeat = 3) -> None:
        super().__init__()

        self.vp_channels = vp_channels
        self.vd_channels = vd_channels
        self.n_layers = n_layers
        self.h_channel = h_channel

        self.skips = [n_layers // 2]
        self.res_nfeat = res_nfeat
        
        self._make_layers()


    def _make_layers(self):
        # layers = []
        # layers.append(nn.Conv2d(self.vp_channels, self.h_channel, kernel_size=1, stride= 1, padding=0))
        self.add_module("FeaExt_module_0", nn.Conv2d(self.vp_channels, self.h_channel, kernel_size=1, stride= 1, padding=0))
        # _xavier_init(self._modules["FeaExt_module_0"])
        # self._modules["FeaExt_module_0"].bias.data[:] = 0.0
        
        for i in range(0, self.n_layers - 1):
            if i in self.skips:
                # layers.append(nn.Conv2d(self.h_channel + self.vp_channels, self.h_channel, kernel_size=1, stride=1, padding=0))
                self.add_module("FeaExt_module_%d"%(i + 1), 
                        nn.Conv2d(self.h_channel + self.vp_channels, self.h_channel, kernel_size=1, stride=1, padding=0))
            else:
                # layers.append(nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))
                self.add_module("FeaExt_module_%d"%(i + 1), 
                        nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))

            _xavier_init(self._modules["FeaExt_module_%d"%(i + 1)])
        # self.feature_module_list = layers
        self.add_module("density_module", nn.Conv2d(self.h_channel, 1, kernel_size=1, stride=1, padding=0))
        _xavier_init(self._modules["density_module"])
        self._modules["density_module"].bias.data[:] = 0.0
        
        self.add_module("RGB_layer_0", nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))
        _xavier_init(self._modules["RGB_layer_0"])
        
        self.add_module("RGB_layer_1", nn.Conv2d(self.h_channel +  self.vd_channels, self.h_channel//2, kernel_size=1, stride=1, padding=0))
        # _xavier_init(self._modules["RGB_layer_1"])
        # self._modules["RGB_layer_1"].bias.data[:] = 0.0
        
        self.add_module("RGB_layer_2", nn.Conv2d(self.h_channel//2, self.res_nfeat, kernel_size=1, stride=1, padding=0))


    def forward(self, batch_embed_vps, batch_embed_vds):
        '''
        batch_embed_vps: [B, C_1, N_r, N_s]
        batch_embed_vds: [B, C_2, N_r, N_s]
        '''

        x = batch_embed_vps
        for i in range(self.n_layers):
            x = self._modules["FeaExt_module_%d"%i](x)
            x = F.relu(x)

            if i in self.skips:
                x = torch.cat([batch_embed_vps, x], dim=1)
            
        density = self._modules["density_module"](x)
        x = self._modules["RGB_layer_0"](x)
        x = self._modules["RGB_layer_1"](torch.cat([x, batch_embed_vds], dim = 1))
        x = F.relu(x)
        rgb = self._modules["RGB_layer_2"](x)
        
        density = F.relu(density)
        if self.res_nfeat == 3:
            rgb = torch.sigmoid(rgb)

        return rgb, density

class MLPforHeadNeRF_Gaze(nn.Module):

    def __init__(self, vp_channels, vd_channels,gaze_channels, n_layers = 8, h_channel = 256, res_nfeat = 3) -> None:
        super().__init__()

        self.vp_channels = vp_channels
        self.vd_channels = vd_channels
        self.n_layers = n_layers
        self.h_channel = h_channel

        self.skips = [n_layers // 2]
        self.res_nfeat = res_nfeat
        
        self._make_layers()
        self.gaze_layers = MLPforGaze(gaze_channels,h_channel=h_channel,res_nfeat=res_nfeat)


    def _make_layers(self):
        # layers = []
        # layers.append(nn.Conv2d(self.vp_channels, self.h_channel, kernel_size=1, stride= 1, padding=0))
        self.add_module("FeaExt_module_0", nn.Conv2d(self.vp_channels, self.h_channel, kernel_size=1, stride= 1, padding=0))
        # _xavier_init(self._modules["FeaExt_module_0"])
        # self._modules["FeaExt_module_0"].bias.data[:] = 0.0
        
        for i in range(0, self.n_layers - 1):
            if i in self.skips:
                # layers.append(nn.Conv2d(self.h_channel + self.vp_channels, self.h_channel, kernel_size=1, stride=1, padding=0))
                self.add_module("FeaExt_module_%d"%(i + 1), 
                        nn.Conv2d(self.h_channel + self.vp_channels, self.h_channel, kernel_size=1, stride=1, padding=0))
            else:
                # layers.append(nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))
                self.add_module("FeaExt_module_%d"%(i + 1), 
                        nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))

            _xavier_init(self._modules["FeaExt_module_%d"%(i + 1)])
        # self.feature_module_list = layers
        self.add_module("density_module", nn.Conv2d(self.h_channel, 1, kernel_size=1, stride=1, padding=0))
        _xavier_init(self._modules["density_module"])
        self._modules["density_module"].bias.data[:] = 0.0
        
        self.add_module("RGB_layer_0", nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))
        _xavier_init(self._modules["RGB_layer_0"])
        
        self.add_module("RGB_layer_1", nn.Conv2d(self.h_channel +  self.vd_channels, self.h_channel//2, kernel_size=1, stride=1, padding=0))
        # _xavier_init(self._modules["RGB_layer_1"])
        # self._modules["RGB_layer_1"].bias.data[:] = 0.0
        
        self.add_module("RGB_layer_2", nn.Conv2d(self.h_channel//2, self.res_nfeat, kernel_size=1, stride=1, padding=0))



    def forward(self, batch_embed_vps, batch_embed_vds,batch_embed_gaze,batch_embed_gaze_temp,for_train = True):
        '''
        batch_embed_vps: [B, C_1, N_r, N_s]
        batch_embed_vds: [B, C_2, N_r, N_s]
        '''
        rgb_feat,density_feat = self.gaze_layers(batch_embed_gaze)

        x = batch_embed_vps
        for i in range(self.n_layers):
            x = self._modules["FeaExt_module_%d"%i](x)
            x = F.relu(x)

            if i in self.skips:
                x = torch.cat([batch_embed_vps, x], dim=1)
        

        if for_train:
            rgb_feat_temp,density_feat_temp = self.gaze_layers(batch_embed_gaze_temp)
            temp_x = x.clone()
            ####template prediction
            density_temp = self._modules["density_module"](temp_x + density_feat_temp)
            density_temp = F.relu(density_temp)

            temp_x = self._modules["RGB_layer_0"](temp_x)
            temp_x = self._modules["RGB_layer_1"](torch.cat([temp_x, batch_embed_vds], dim = 1))
            temp_x = F.relu(temp_x)
            rgb_temp = self._modules["RGB_layer_2"](temp_x + rgb_feat_temp)
            if self.res_nfeat == 3:
                rgb_temp = torch.sigmoid(rgb_temp)

        ####current prediction
        density = self._modules["density_module"](x + density_feat)
        density = F.relu(density)
        x = self._modules["RGB_layer_0"](x)
        x = self._modules["RGB_layer_1"](torch.cat([x, batch_embed_vds], dim = 1))
        x = F.relu(x)
        rgb = self._modules["RGB_layer_2"](x + rgb_feat)
        if self.res_nfeat == 3:
            rgb = torch.sigmoid(rgb)

    
        return (rgb, density, rgb_temp, density_temp) if for_train else (rgb,density,rgb,density)


class MLPforGaze(nn.Module):
    def __init__(self, gaze_channels, n_layers = 8, h_channel = 256, res_nfeat = 3) -> None:
        super().__init__()

        self.gaze_channels = gaze_channels
        self.n_layers = n_layers
        self.h_channel = h_channel

        self.skips = [n_layers // 2]
        self.res_nfeat = res_nfeat
        
        self._make_layers()


    def _make_layers(self):
        # layers = []
        # layers.append(nn.Conv2d(self.vp_channels, self.h_channel, kernel_size=1, stride= 1, padding=0))
        self.add_module("Gaze_FeaExt_module_0", nn.Conv2d(self.gaze_channels, self.h_channel, kernel_size=1, stride= 1, padding=0))
        # _xavier_init(self._modules["FeaExt_module_0"])
        # self._modules["FeaExt_module_0"].bias.data[:] = 0.0
        
        for i in range(0, self.n_layers - 1):
            if i in self.skips:
                # layers.append(nn.Conv2d(self.h_channel + self.vp_channels, self.h_channel, kernel_size=1, stride=1, padding=0))
                self.add_module("Gaze_FeaExt_module_%d"%(i + 1), 
                        nn.Conv2d(self.h_channel + self.gaze_channels, self.h_channel, kernel_size=1, stride=1, padding=0))
            else:
                # layers.append(nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))
                self.add_module("Gaze_FeaExt_module_%d"%(i + 1), 
                        nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))

            _xavier_init(self._modules["Gaze_FeaExt_module_%d"%(i + 1)])
        # self.feature_module_list = layers
        # self.add_module("density_module", nn.Conv2d(self.h_channel, 1, kernel_size=1, stride=1, padding=0))
        # _xavier_init(self._modules["density_module"])
        # self._modules["density_module"].bias.data[:] = 0.0
        
        self.add_module("Gaze_RGB_layer_0", nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))
        _xavier_init(self._modules["Gaze_RGB_layer_0"])
        
        self.add_module("Gaze_RGB_layer_1", nn.Conv2d(self.h_channel, self.h_channel//2, kernel_size=1, stride=1, padding=0))
        # _xavier_init(self._modules["RGB_layer_1"])
        # self._modules["RGB_layer_1"].bias.data[:] = 0.0

    def forward(self, batch_embed_gaze):
        '''
        batch_embed_vps: [B, C_1, N_r, N_s]
        batch_embed_vds: [B, C_2, N_r, N_s]
        '''

        x = batch_embed_gaze
        for i in range(self.n_layers):
            x = self._modules["Gaze_FeaExt_module_%d"%i](x)
            x = F.relu(x)

            if i in self.skips:
                x = torch.cat([batch_embed_gaze, x], dim=1)
            
        density_feat = x.clone()
        rgb_feat = self._modules["Gaze_RGB_layer_0"](x)
        rgb_feat = self._modules["Gaze_RGB_layer_1"](x)
        rgb_feat = F.relu(rgb_feat)
         
        return rgb_feat, density_feat