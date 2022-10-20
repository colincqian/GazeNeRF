import torch
from headnerf_trainer import Trainer
from XGaze_utils.data_loader_xgaze import get_data_loader

import numpy as np

import configparser
import yaml
import argparse


class BaseOptions(object):
    def __init__(self, base_opt_dic,para_dict = None) -> None:
        super().__init__()
        for key in base_opt_dic:
            setattr(self, key, base_opt_dic[key])
        
        if para_dict is not None:
            self.featmap_size = para_dict["featmap_size"]
            self.featmap_nc = para_dict["featmap_nc"]
            self.pred_img_size = para_dict["pred_img_size"]

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def parse_argument():
    parser = argparse.ArgumentParser(description='Specifiy config file path')
    parser.add_argument('--config_file', type=str, default='config/train.yml',
                        help='Directory of the config file')
    args = parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file,'r') as f:
        print(f'----Load Configuration from {config_file}----')
        data = yaml.load(f,Loader=yaml.FullLoader)
    return data

def run(config):
    kwargs = {}
    training_config = Dict2Class(config['training_config'])
    base_opt = config['base_opt']
    dataset_config = config['dataset_config']

    if training_config.headnerf_options:
        check_dict = torch.load(training_config.headnerf_options, map_location=torch.device("cpu"))
        para_dict = check_dict["para"]
        opt = BaseOptions(base_opt,para_dict)
        dataset_config['opt'] =  opt

    if training_config.use_gpu:
        # ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        np.random.seed(0)
        kwargs = {'num_workers': training_config.num_workers}

    # instantiate data loaders
    if training_config.is_train:
        data_loader = get_data_loader(
                        mode='train',
                        batch_size=training_config.batch_size,
                        num_workers=training_config.num_workers,
                        dataset_config=dataset_config
                        )
    else:
        data_loader = get_data_loader(
                        mode='test',
                        batch_size=training_config.batch_size,
                        num_workers=training_config.num_workers,
                        dataset_config=dataset_config
                        )

    # instantiate trainer
    trainer = Trainer(training_config, data_loader)

    # either train
    if training_config.is_train:
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()

if __name__ == '__main__':
    torch.manual_seed(0)
    args = parse_argument()
    config = load_config(args.config_file)
    run(config)