import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from typing import List

import cv2
import csv
import pickle as pkl

trans_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_train_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'train'
    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=trans, is_shuffle=is_shuffle, is_load_label=True)
    #dataloader returns image torch.Size([3, 224, 224]) and label(Gaze)  [2] array
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader


def get_test_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'test'
    test_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                           transform=trans, is_shuffle=is_shuffle, is_load_label=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader


# class GazeDataset(Dataset):
#     def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None, is_shuffle=True,
#                  index_file=None, is_load_label=True):
#         self.path = dataset_path
#         self.hdfs = {}
#         self.sub_folder = sub_folder
#         self.is_load_label = is_load_label

#         # assert len(set(keys_to_use) - set(all_keys)) == 0
#         # Select keys
#         # TODO: select only people with sufficient entries?
#         self.selected_keys = [k for k in keys_to_use]
#         assert len(self.selected_keys) > 0

#         for num_i in range(0, len(self.selected_keys)):
#             file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
#             self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
#             # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
#             assert self.hdfs[num_i].swmr_mode

#         # Construct mapping from full-data index to key and person-specific index
#         if index_file is None:
#             self.idx_to_kv = []
#             for num_i in range(0, len(self.selected_keys)):
#                 n = self.hdfs[num_i]["face_patch"].shape[0]
#                 self.idx_to_kv += [(num_i, i) for i in range(n)]
#         else:
#             print('load the file: ', index_file)
#             self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

#         for num_i in range(0, len(self.hdfs)):
#             if self.hdfs[num_i]:
#                 self.hdfs[num_i].close()
#                 self.hdfs[num_i] = None

#         if is_shuffle:
#             random.shuffle(self.idx_to_kv)  # random the order to stable the training

#         self.hdf = None
#         self.transform = transform

#     def __len__(self):
#         return len(self.idx_to_kv)

#     def __del__(self):
#         for num_i in range(0, len(self.hdfs)):
#             if self.hdfs[num_i]:
#                 self.hdfs[num_i].close()
#                 self.hdfs[num_i] = None

#     def __getitem__(self, idx):
#         key, idx = self.idx_to_kv[idx]

#         self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
#         assert self.hdf.swmr_mode

#         # Get face image
#         image = self.hdf['face_patch'][idx, :]
#         image = image[:, :, [2, 1, 0]]  # from BGR to RGB
#         image = self.transform(image)

#         # Get labels
#         if self.is_load_label:
#             gaze_label = self.hdf['face_gaze'][idx, :]
#             gaze_label = gaze_label.astype('float')
#             return image, gaze_label
#         else:
#             return image

import json
import os

def get_train_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'train'
    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=trans, is_shuffle=is_shuffle, is_load_label=True)
    #dataloader returns image torch.Size([3, 224, 224]) and label(Gaze)  [2] array
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader


def get_data_loader(data_dir,
                    annotation_path,
                    batch_size,
                    num_workers=4,
                    is_shuffle=True):
    gpu_id = 0
    frame_selected=[0]
    subject_selected=[0]
    device = torch.device("cuda:%d" % gpu_id)
    options = BaseOptions()

    dataset = XGaze_raw(data_dir,
                        frame_selected,
                        subject_selected,
                        annotation_path,
                        device,
                        options,
                        is_shuffle=is_shuffle
                        )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

class BaseOptions(object):
    def __init__(self, para_dict = None) -> None:
        super().__init__()
        
        self.bg_type = "white" # white: white bg, black: black bg.
        
        self.iden_code_dims = 100
        self.expr_code_dims = 79
        self.text_code_dims = 100
        self.illu_code_dims = 27

        self.auxi_shape_code_dims = 179
        self.auxi_appea_code_dims = 127
        
        # self.num_ray_per_img = 972 #972, 1200, 1452, 1728, 2028, 2352
        self.num_sample_coarse = 64
        self.num_sample_fine = 128

        self.world_z1 = 2.5
        self.world_z2 = -3.5
        self.mlp_hidden_nchannels = 384

        if para_dict is None:
            self.featmap_size = 64
            self.featmap_nc = 256       # nc: num_of_channel
            self.pred_img_size = 512
        else:
            self.featmap_size = para_dict["featmap_size"]
            self.featmap_nc = para_dict["featmap_nc"]
            self.pred_img_size = para_dict["pred_img_size"]


class XGaze_raw(Dataset):
    def __init__(self,dataset_path: str,frame_selected:list,subject_selected:list,annotation_path:str,device,opt:BaseOptions,is_shuffle=True):
        self.path = dataset_path
        self.annot_path = annotation_path
        self.frame_selected = frame_selected
        self.subject_selected = subject_selected
        self.device = device
        self.opt = opt

        self.data_dic = {} #{imgname, path, img, gaze_label,head_pose_label,camera parameter annot, camera parameter 3dmm, code info, image mask}
        self.data_info = {} #subject_id_frame_id_camera_id
        self.idx_to_kv = [] #map idx to (subject_id,frame_id,camera_id)


        img_size = (self.opt.pred_img_size, self.opt.pred_img_size)
        self.pred_img_size = self.opt.pred_img_size
        self.featmap_size = self.opt.featmap_size
        self.featmap_nc = self.opt.featmap_nc

        for subject_id in self.subject_selected:
            gen_annotation = self.load_annot(os.path.join(self.annot_path,'subject%04d.csv'%(subject_id,)))
            for frame_id in self.frame_selected:
                for cam_id in range(18):
                    annotation = next(gen_annotation)
                    img_name = 'frame%04d_cam%02d' %(frame_id,cam_id)
                    self.img_path = os.path.join(dataset_path,img_name+'.png')
                    
                    if not self.check_file_existance():
                        print(f'cannot find img file or 3dmm model parameter of {img_name}')
                        continue

                    img = cv2.imread(self.img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32)/255.0
                    self.gt_img_size = img.shape[0]
                    if self.gt_img_size != self.pred_img_size:
                        img = cv2.resize(img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

                    mask_img = cv2.imread(self.img_path.replace(".png","_mask.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
                    if mask_img.shape[0] != self.pred_img_size:
                        mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

                    img[mask_img < 0.5] = 1.0
                    img_tensor = (torch.from_numpy(img).permute(2, 0, 1)).unsqueeze(0).to(self.device)
                    mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)

                    
                    
                    self.load_3dmm_params(self.img_path.replace(".png","_nl3dmm.pkl"))

                    self.data_info[(subject_id,frame_id,cam_id)] = {
                        'imgname' : img_name,
                        'img_path': self.img_path,
                        'img' : img_tensor,
                        'gaze': torch.tensor(annotation[:2],device=self.device),  ##only available in training set
                        'head_pose': torch.tensor(annotation[5:11],device=self.device),
                        'camera_parameter': torch.zeros(1),
                        '_3dmm': {'cam_info':self.cam_info,
                                  'code_info':self.code_info},
                        'img_mask' : mask_tensor
                    }
                    self.idx_to_kv.append((subject_id,frame_id,cam_id))
        
                
        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        key = self.idx_to_kv[idx]
        #assert type(self.data_info[key]) == 'dict'
        return self.data_info[key]

    def check_file_existance(self):
        return os.path.exists(self.img_path) & \
        os.path.exists(self.img_path.replace(".png","_mask.png")) & \
        os.path.exists(self.img_path.replace(".png","_nl3dmm.pkl"))

        


    def load_3dmm_params(self,para_3dmm_path):
        # load init codes from the results generated by solving 3DMM rendering opt.
        with open(para_3dmm_path, "rb") as f: nl3dmm_para_dict = pkl.load(f)
        base_code = nl3dmm_para_dict["code"].detach().unsqueeze(0).to(self.device)
        
        base_iden = base_code[:, :self.opt.iden_code_dims]
        base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
        base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims 
                                                            + self.opt.expr_code_dims + self.opt.text_code_dims]
        base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]
        
        self.base_c2w_Rmat = nl3dmm_para_dict["c2w_Rmat"].detach().unsqueeze(0)
        self.base_c2w_Tvec = nl3dmm_para_dict["c2w_Tvec"].detach().unsqueeze(0).unsqueeze(-1)
        self.base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"].detach().unsqueeze(0)
        self.base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"].detach().unsqueeze(0).unsqueeze(-1)

        temp_inmat = nl3dmm_para_dict["inmat"].detach().unsqueeze(0)
        temp_inmat[:, :2, :] *= (self.featmap_size / self.gt_img_size)
        
        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0
        
        #self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.to(self.device),
            "batch_Tvecs": self.base_c2w_Tvec.to(self.device),
            "batch_inv_inmats": self.temp_inv_inmat.to(self.device)
        }

        self.code_info = {
            "base_iden" : base_iden,
            "base_expr" : base_expr,
            "base_text" : base_text,
            "base_illu" : base_illu,
            "inmat" : temp_inmat,
            "inv_inmat" : temp_inv_inmat
        }



    def load_annot(self,file_path):
        with open(file_path) as f:
            csv_f = csv.reader(f,delimiter=',')
            for row in csv_f:
                yield np.array(row[2:],dtype=np.float) 



if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    Dataloader = get_data_loader('/home/colinqian/Project/HeadNeRF/headnerf/XGaze_utils/playground',
                    '/home/colinqian/Project/HeadNeRF/headnerf/XGaze_utils',
                    batch_size=4,num_workers=0)
    for iter,batch in enumerate(Dataloader):
        import ipdb;
        ipdb.set_trace()
        pass