python Control_single_image_eyegaze.py --model_path 'TrainedModels/baseline_results/epoch25/epoch_25_0.88_18.57_0.07_ckpt.pth.tar'\
                                       --hdf_file 'XGaze_data/processed_data/processed_subject0000'\
                                       --image_index 0 \
                                       --save_root 'logs/Control_single_image_eyegaze_output' \
                                       --gaze_dim 64 \
                                       --eye_gaze_scale_factor 1 \
                                       --vis_gaze_vect False