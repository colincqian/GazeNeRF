import cv2
from DataProcess.Gen_mask_lm_3DMM import Data_Processor
from train_headnerf import load_config,Dict2Class
import time
import torch
from prettytable import PrettyTable

class gaze2code_disentanglement(object):
    def __init__(self,base_opt,image_size=512,intermediate_size=256):
        self.base_opt = base_opt
        self.data_process = Data_Processor('','',img_size=image_size,intermediate_size=intermediate_size)
        self.img1_cache = None
        self.img2_cache = None

    def compute_image_attribute_displacement(self,img1,img2,vis=False,use_img1_cache=False,use_img2_cache=False):
        if (not use_img1_cache) or self.img1_cache is None:
            _,_,_,nl3dmm_param1 = self.data_process.process_mask_and_landmark_for_single_image(img1)
            self.img1_cache = nl3dmm_param1
        else:
            print('use img1 cache')
            nl3dmm_param1 = self.img1_cache

        if (not use_img2_cache) or self.img2_cache is None:
            _,_,_,nl3dmm_param2 = self.data_process.process_mask_and_landmark_for_single_image(img2)
            self.img2_cache = nl3dmm_param2
        else:
            print('use img2 cache')
            nl3dmm_param2 = self.img2_cache

        code1 = nl3dmm_param1['code']
        code2 = nl3dmm_param2['code']

        code_dict1 = disentanglement_function(code1,self.base_opt)
        code_dict2 = disentanglement_function(code2,self.base_opt)

        table,res = compute_code_displacement(code_dict1,code_dict2,vis=vis)
        
        if vis:
            print(table)

        return res
    
def disentanglement_function(base_code,opt):
    base_iden = base_code[ :opt.iden_code_dims]
    base_expr = base_code[ opt.iden_code_dims:opt.iden_code_dims + opt.expr_code_dims]
    base_text = base_code[ opt.iden_code_dims + opt.expr_code_dims:opt.iden_code_dims 
                                                        + opt.expr_code_dims + opt.text_code_dims]
    base_illu = base_code[ opt.iden_code_dims + opt.expr_code_dims + opt.text_code_dims:]
    
    appear = torch.concat([base_text,base_illu])
    shape = torch.concat([base_iden,base_expr])
    
    return {"base_iden":base_iden,
            "base_expr":base_expr,
            "base_text":base_text,
            "base_illu":base_illu,
            'appear_code':appear,
            'shape_code':shape}

def compute_code_displacement(code_dict1,code_dict2,vis=False):
    t = PrettyTable(['Codes', 'Value'])
    res={}
    for key in code_dict1.keys():
        res[key] = torch.mean(torch.abs(code_dict1[key] - code_dict2[key]))

    if vis:
        for k,v in res.items():
            t.add_row([k,v.item()])
    return t,res



if __name__ == "__main__":
    start = time.time()
    config = load_config("config/full_evaluation.yml")
    base_opt = Dict2Class(config["base_opt"])
    dt = gaze2code_disentanglement(base_opt=base_opt)

    data_process = Data_Processor('','',img_size=512,intermediate_size=256)

    ref_img_strong = cv2.imread("experiment_document/disentanglement_error/strong_disentanglement/grid_sample(0.0, 0.0).png")
    ref_img_weak = cv2.imread("experiment_document/disentanglement_error/weak_disentanglement/grid_sample(0.0, 0.0).png")
    image1 = cv2.imread("")
    image2 = cv2.imread("")

    test_image_name_list = ['grid_sample(1.0, 1.0).png','grid_sample(1.0, -1.0).png','grid_sample(-1.0, 1.0).png','grid_sample(-1.0, -1.0).png']

    dt.compute_image_attribute_displacement(img1=,img2=)

    end = time.time()
    print(f"time_elapse:{end - start}")

    import ipdb
    ipdb.set_trace()