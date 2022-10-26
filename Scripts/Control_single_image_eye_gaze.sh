python Evaluation.py --model_path 'TrainedModels/anchor_template_disp_version/epoch_20_0.89_19.14_0.07_ckpt.pth.tar'\
                                       --hdf_file 'XGaze_data/processed_data_10cam/processed_subject0000'\
                                       --image_index 0 \
                                       --save_root 'logs/Control_single_image_eyegaze_output' \
                                       --gaze_dim 64 \
                                       --eye_gaze_scale_factor 1 \
                                       --vis_gaze_vect True \
                                       --D6_rotation False  \
                                       --model_name 'HeadNeRF' \
                                       --config_file_path 'config/train.yml'
                            