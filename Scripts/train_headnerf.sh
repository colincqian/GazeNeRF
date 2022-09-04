python train_headnerf.py --headnerf_options "TrainedModels/model_Reso32HR.pth" \
                         --batch_size 2 \
                         --gpu_id 3 \
                         --include_eye_gaze True \
                         --eye_gaze_dimension 66 \
                         --eye_gaze_scale_factor 1 \
                         --print_freq 50 \
                         --gaze_D6_rotation True \
                         --eye_gaze_disentangle False \
                         --comment '6D rotataion, gaze_dim 66, disentangle turned off ' 

