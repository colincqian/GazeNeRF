python train_headnerf.py --headnerf_options "TrainedModels/model_Reso32HR.pth" \
                         --batch_size 2 \
                         --gpu_id 3 \
                         --include_eye_gaze True \
                         --eye_gaze_dimension 64 \
                         --eye_gaze_scale_factor 2 \
                         --comment 'train with displacement loss (weight 3), scale factor 2, no normalization to eye gaze( from -2 to 2)' \
                         --print_freq 50
