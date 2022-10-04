python train_headnerf.py --headnerf_options "TrainedModels/model_Reso32HR.pth" \
                         --batch_size 2 \
                         --gpu_id 3 \
                         --include_eye_gaze True \
                         --eye_gaze_dimension 64 \
                         --eye_gaze_scale_factor 1 \
                         --print_freq 50 \
                         --gaze_D6_rotation False \
                         --eye_gaze_disentangle True \
                         --comment 'gaze_dim 64,disentanglement True, 10 cam views' 

