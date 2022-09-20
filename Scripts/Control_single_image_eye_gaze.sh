python Control_single_image_eyegaze.py --model_path 'TrainedModels/disentangle_ablation/gaze64_stable_w3disp_included_gaze-1to1.tar'\
                                       --hdf_file 'XGaze_data/processed_data/processed_subject0000'\
                                       --image_index 0 \
                                       --save_root 'logs/Control_single_image_eyegaze_output' \
                                       --gaze_dim 64 \
                                       --eye_gaze_scale_factor 1 \
                                       --vis_gaze_vect True \
                                       --D6_rotation False 