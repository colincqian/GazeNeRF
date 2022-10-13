python -m ipdb train_headnerf.py --headnerf_options "TrainedModels/model_Reso32HR.pth" \
                         --batch_size 1 \
                         --gpu_id 0\
                         --include_eye_gaze False \
                         --eye_gaze_dimension 64 \
                         --eye_gaze_scale_factor 1 \
                         --print_freq 50 \
                         --gaze_D6_rotation False \
                         --eye_gaze_disentangle False \
                         --comment 'test' 

