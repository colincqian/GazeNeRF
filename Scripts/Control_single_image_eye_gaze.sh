python Control_single_image_eyegaze.py --model_path 'TrainedModels/learning_template_change/face_mask_dim64_with_template_loss_ver6/epoch_24_0.92_21.27_0.04_ckpt.pth.tar'\
                                       --hdf_file 'XGaze_data/processed_data_10cam/processed_subject0000'\
                                       --image_index 0 \
                                       --save_root 'logs/Control_single_image_eyegaze_output' \
                                       --gaze_dim 64 \
                                       --eye_gaze_scale_factor 1 \
                                       --vis_gaze_vect True \
                                       --D6_rotation False  \
                                       --model_name 'HeadNeRF_Gaze' \
                                       --config_file_path 'config/train.yml'
                            