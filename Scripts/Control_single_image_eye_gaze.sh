python Control_single_image_eyegaze.py --model_path 'TrainedModels/10_cam_view_disentangle_results/epoch_10_0.90_19.50_0.06_ckpt.pth.tar'\
                                       --hdf_file 'XGaze_data/processed_data_10cam/processed_subject0000'\
                                       --image_index 0 \
                                       --save_root 'logs/Control_single_image_eyegaze_output' \
                                       --gaze_dim 64 \
                                       --eye_gaze_scale_factor 1 \
                                       --vis_gaze_vect True \
                                       --D6_rotation False 