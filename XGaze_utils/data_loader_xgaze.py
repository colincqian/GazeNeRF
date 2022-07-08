from logging import raiseExceptions
from signal import raise_signal
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from typing import List

import json
import os

import h5py
import cv2
import csv
import pickle as pkl

from XGaze_utils.XGaze_camera_Loader import Camera_Loader


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

def get_data_loader(    mode='train',
                        batch_size=8,
                        num_workers=4,
                        dataset_config=None):

    if dataset_config is None:
        print('dataset configure file required!!')
        raise
    torch.manual_seed(0)
    dataset_config['sub_folder'] = mode #'train' or 'test'
    
    #XGaze_dataset = GazeDataset_normailzed(**dataset_config)
    XGaze_dataset = GazeDataset_normailzed_from_hdf(**dataset_config)

    if mode=='train':
        #training,validation random split
        train_size = int(0.95*len(XGaze_dataset));validation_size = len(XGaze_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(XGaze_dataset, [train_size, validation_size])

        train_loader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=True)
        val_loader = DataLoader(val_dataset,batch_size=1,num_workers=num_workers,drop_last=True)

        return (train_loader,val_loader)
    else:
        print('Not implement test dataloader!!')
        raise NotImplementedError
        


#########put this in config file after all testing########################
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

################data loader for normalized data from hdf file#############################        
class GazeDataset_normailzed_from_hdf(Dataset):
    def __init__(self, dataset_path: str,
                 opt: BaseOptions,
                 keys_to_use: List[str] = None, 
                 sub_folder='',
                 camera_dir='',
                 _3dmm_data_dir='',
                 transform=None, 
                 is_shuffle=True,
                 index_file=None, 
                 is_load_label=True,
                 device = 'cpu',
                 filter_view=False):
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label
        self.camera_loader = Camera_Loader(camera_dir)
        self._3dmm_data_dir = _3dmm_data_dir
        self.device = device
        self.filter_view = filter_view
        if opt is not None:
            self.opt = opt
        else:
            print('option class required, input of opt is None!!')
            raise
        self.img_size = (self.opt.pred_img_size, self.opt.pred_img_size)
        self.pred_img_size = self.opt.pred_img_size
        self.featmap_size = self.opt.featmap_size
        self.featmap_nc = self.opt.featmap_nc
        

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        if self.filter_view:
            ##filter out some severe camera view
            dist_index = [(np.linalg.norm(self.camera_loader[i]['cam_translation']),i) for i in range(18)]
            dist_index.sort()
            self.valid_camera_index = {index for dist,index in dist_index[:10]}#keep camera with 10 least distance

        self.selected_keys = [k for k in keys_to_use] #list of h5 file name
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path,f'processed_subject{self.selected_keys[num_i].zfill(4)}')
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                hdfs_file = self.hdfs[num_i]
                n = hdfs_file["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)
                                    if hdfs_file['valid_mask'][i] ] #our processed image if from 1
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.transform = transform

        #self.debug_iter(0)

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]

        file_path = os.path.join(self.path,f'processed_subject{self.selected_keys[key].zfill(4)}')
        self.hdf = h5py.File(file_path, 'r', swmr=True)
        assert self.hdf.swmr_mode

        #img_name = str(idx+1).zfill(6)+'.png'
        #img_path = os.path.join(self._3dmm_data_dir,img_name)
        img_index = idx

        # Get face image
        #<KeysViewHDF5 ['cam_index', 'face_gaze', 'face_head_pose', 'face_mat_norm', 'face_patch',     'frame_index']>
        #               (10098, 1)    (10098, 2)     (10098, 2)     (10098, 3, 3)  (10098, 224, 224, 3)  (10098, 1)
        #                       
        #image = self.hdf['face_patch'][idx, :] ##(224,224,3)
        image = self.hdf['face_patch'][img_index]

        #image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        if self.transform is not None:
            image = self.transform(image)
        image = image.astype(np.float32)/255.0

        self.gt_img_size = image.shape[0]
        if self.gt_img_size != self.pred_img_size:
            image = cv2.resize(image, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

        mask_img =  self.hdf['mask'][img_index]
        eye_mask_img = self.hdf['eye_mask'][img_index]
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

        if eye_mask_img.shape[0] != self.pred_img_size:
            eye_mask_img = cv2.resize(eye_mask_img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        

        image[mask_img < 0.5] = 1.0
        img_tensor = (torch.from_numpy(image).permute(2, 0, 1)).unsqueeze(0).to(self.device)#not sure RGB or BRG
        #img_tensor = (torch.from_numpy(image)).unsqueeze(0).to(self.device)#not sure RGB or BRG
        mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)
        eye_mask_tensor = torch.from_numpy(eye_mask_img[None, :, :]).unsqueeze(0).to(self.device)

        
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][img_index]
            gaze_label = gaze_label.astype('float')
            gaze_tensor = (torch.from_numpy(gaze_label)).to(self.device)
        else:
            gaze_tensor = torch.tensor([None,None])

        camera_index = self.hdf['cam_index'][img_index][0]
        camera_parameter = self.camera_loader[camera_index-1]  ##ground truth camera info

        self.load_3dmm_params(img_index)

        data_info = {
                        'img' : img_tensor,
                        'gaze': gaze_tensor,  ##only available in training set
                        'camera_parameter': camera_parameter,
                        '_3dmm': {'cam_info':self.cam_info,
                                  'code_info':self.code_info},
                        'img_mask' : mask_tensor,
                        'eye_mask' : eye_mask_tensor
                    }
        return data_info

    def load_3dmm_params(self,index):
        # load init codes from the results generated by solving 3DMM rendering opt.
        nl3dmm_para_dict = self.hdf['nl3dmm']
        base_code = nl3dmm_para_dict["code"][index].detach().unsqueeze(0).to(self.device)
        
        base_iden = base_code[:, :self.opt.iden_code_dims]
        base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
        base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims 
                                                            + self.opt.expr_code_dims + self.opt.text_code_dims]
        base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]
        
        self.base_c2w_Rmat = nl3dmm_para_dict["c2w_Rmat"][index].detach().unsqueeze(0)
        self.base_c2w_Tvec = nl3dmm_para_dict["c2w_Tvec"][index].detach().unsqueeze(0).unsqueeze(-1)
        self.base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"][index].detach().unsqueeze(0)
        self.base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"][index].detach().unsqueeze(0).unsqueeze(-1)

        temp_inmat = nl3dmm_para_dict["inmat"][index].detach().unsqueeze(0)
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

    def debug_iter(self,idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        img_name = str(idx+1).zfill(6)+'.png'
        img_path = os.path.join(self._3dmm_data_dir,img_name)

        mask_img = cv2.imread(img_path.replace(".png","_mask.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
        
        import ipdb 
        ipdb.set_trace()
        self.load_3dmm_params(os.path.join(self._3dmm_data_dir,img_name.replace(".png","_nl3dmm.pkl")))

        # Get face image
        #<KeysViewHDF5 ['cam_index', 'face_gaze', 'face_head_pose', 'face_mat_norm', 'face_patch',     'frame_index']>
        #               (10098, 1)    (10098, 2)     (10098, 2)     (10098, 3, 3)  (10098, 224, 224, 3)  (10098, 1)
        #                       
        image_load = cv2.imread(img_path)

        image_load = image_load[:, :, [2, 1, 0]]  # from BGR to RGB
        image_load = image_load.astype(np.float32)/255.0
        image_load[mask_img < 0.5] = 1.0

        cv2.imshow('image mask', mask_img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        cv2.imshow('image after masking',image_load)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 

################data loader for normalized data#############################
class GazeDataset_normailzed(Dataset):
    def __init__(self, dataset_path: str,
                 opt: BaseOptions,
                 keys_to_use: List[str] = None, 
                 sub_folder='',
                 camera_dir='',
                 _3dmm_data_dir='',
                 transform=None, 
                 is_shuffle=True,
                 index_file=None, 
                 is_load_label=True,
                 device = 'cpu',
                 filter_view=False):
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label
        self.camera_loader = Camera_Loader(camera_dir)
        self._3dmm_data_dir = _3dmm_data_dir
        self.device = device
        self.filter_view = filter_view
        if opt is not None:
            self.opt = opt
        else:
            print('option class required, input of opt is None!!')
            raise
        self.img_size = (self.opt.pred_img_size, self.opt.pred_img_size)
        self.pred_img_size = self.opt.pred_img_size
        self.featmap_size = self.opt.featmap_size
        self.featmap_nc = self.opt.featmap_nc
        

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        if self.filter_view:
            ##filter out some severe camera view
            dist_index = [(np.linalg.norm(self.camera_loader[i]['cam_translation']),i) for i in range(18)]
            dist_index.sort()
            self.valid_camera_index = {index for dist,index in dist_index[:10]}#keep camera with 10 least distance

        self.selected_keys = [k for k in keys_to_use] #list of h5 file name
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                hdfs_file = self.hdfs[num_i]
                n = hdfs_file["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)
                                    if self.is_valid_data_sample(i,hdfs_file)] #our processed image if from 1
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.transform = transform

        #self.debug_iter(0)

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        img_name = str(idx+1).zfill(6)+'.png'
        img_path = os.path.join(self._3dmm_data_dir,img_name)

        # Get face image
        #<KeysViewHDF5 ['cam_index', 'face_gaze', 'face_head_pose', 'face_mat_norm', 'face_patch',     'frame_index']>
        #               (10098, 1)    (10098, 2)     (10098, 2)     (10098, 3, 3)  (10098, 224, 224, 3)  (10098, 1)
        #                       
        #image = self.hdf['face_patch'][idx, :] ##(224,224,3)
        image = cv2.imread(img_path)##(250,250,3)

        #image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        if self.transform is not None:
            image = self.transform(image)
        image = image.astype(np.float32)/255.0

        self.gt_img_size = image.shape[0]
        if self.gt_img_size != self.pred_img_size:
            image = cv2.resize(image, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

        mask_img = cv2.imread(img_path.replace(".png","_mask.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
        eye_mask_img = cv2.imread(img_path.replace(".png","_mask_eye.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

        if eye_mask_img.shape[0] != self.pred_img_size:
            eye_mask_img = cv2.resize(eye_mask_img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        

        image[mask_img < 0.5] = 1.0
        img_tensor = (torch.from_numpy(image).permute(2, 0, 1)).unsqueeze(0).to(self.device)#not sure RGB or BRG
        #img_tensor = (torch.from_numpy(image)).unsqueeze(0).to(self.device)#not sure RGB or BRG
        mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)
        eye_mask_tensor = torch.from_numpy(eye_mask_img[None, :, :]).unsqueeze(0).to(self.device)

        
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :]
            gaze_label = gaze_label.astype('float')
            gaze_tensor = (torch.from_numpy(gaze_label)).to(self.device)
        else:
            gaze_tensor = torch.tensor([None,None])

        head_pose = self.hdf['face_head_pose'][idx, :]
        head_pose = head_pose.astype('float')
        head_pose = (torch.from_numpy(head_pose)).to(self.device)

        camera_index = self.hdf['cam_index'][idx,:][0]
        camera_parameter = self.camera_loader[camera_index-1]  ##ground truth camera info

        self.load_3dmm_params(os.path.join(self._3dmm_data_dir,img_name.replace(".png","_nl3dmm.pkl")))

        data_info = {
                        'imgname' : img_name,
                        'img_path': img_path,
                        'img' : img_tensor,
                        'gaze': gaze_tensor,  ##only available in training set
                        'head_pose': head_pose,
                        'camera_parameter': camera_parameter,
                        '_3dmm': {'cam_info':self.cam_info,
                                  'code_info':self.code_info},
                        'img_mask' : mask_tensor,
                        'eye_mask' : eye_mask_tensor
                    }
        return data_info

    def is_valid_data_sample(self,i,hdfs_file):
        mm3d_param_exist = os.path.exists(os.path.join(self._3dmm_data_dir,str(i+1).zfill(6) + "_nl3dmm.pkl"))
        
        mask_file = os.path.join(self._3dmm_data_dir,str(i+1).zfill(6) + "_mask.png")
        mask_img = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        valid_mask_img = bool((mask_img>125).any())

        if self.filter_view:
            camera_index  = hdfs_file['cam_index'][i,:]
            is_valid_camera = (camera_index[0] in self.valid_camera_index)
        else:
            is_valid_camera = True


        return mm3d_param_exist & valid_mask_img & is_valid_camera

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

    def debug_iter(self,idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        img_name = str(idx+1).zfill(6)+'.png'
        img_path = os.path.join(self._3dmm_data_dir,img_name)

        mask_img = cv2.imread(img_path.replace(".png","_mask.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
        
        import ipdb 
        ipdb.set_trace()
        self.load_3dmm_params(os.path.join(self._3dmm_data_dir,img_name.replace(".png","_nl3dmm.pkl")))

        # Get face image
        #<KeysViewHDF5 ['cam_index', 'face_gaze', 'face_head_pose', 'face_mat_norm', 'face_patch',     'frame_index']>
        #               (10098, 1)    (10098, 2)     (10098, 2)     (10098, 3, 3)  (10098, 224, 224, 3)  (10098, 1)
        #                       
        image_load = cv2.imread(img_path)

        image_load = image_load[:, :, [2, 1, 0]]  # from BGR to RGB
        image_load = image_load.astype(np.float32)/255.0
        image_load[mask_img < 0.5] = 1.0

        cv2.imshow('image mask', mask_img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        cv2.imshow('image after masking',image_load)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 


################data loader for raw data#############################
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
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    # Dataloader = get_data_loader('/home/colinqian/Project/HeadNeRF/headnerf/XGaze_utils/playground',
    #                 '/home/colinqian/Project/HeadNeRF/headnerf/XGaze_utils',
    #                 batch_size=4,num_workers=0)
    # for iter,batch in enumerate(Dataloader):
    #     import ipdb;
    #     ipdb.set_trace()
    #     pass


    #################test normalized data#####################
    opt = BaseOptions()
    dataset_config={
        'dataset_path': './XGaze_Local/xgaze/',
        'opt': BaseOptions(),
        'keys_to_use':['subject0000.h5'], 
        'sub_folder':'train',
        'camera_dir':'./XGaze_Local/xgaze/camera_parameters',
        '_3dmm_data_dir':'./XGaze_Local/normalized_250_data',
        'transform':None, 
        'is_shuffle':False,
        'index_file':None, 
        'is_load_label':True,
        'device': 'cpu',
        'filter_view': True

    }
    gaze_dataset = GazeDataset_normailzed(**dataset_config)
    data_loader = DataLoader(gaze_dataset, batch_size=1, num_workers=4)
    for iter,batch in enumerate(data_loader):
        import ipdb;
        ipdb.set_trace()
        pass



