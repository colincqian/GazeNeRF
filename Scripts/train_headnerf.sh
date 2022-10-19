python -m ipdb train_headnerf.py --headnerf_options "TrainedModels/model_Reso32HR.pth" \
                         --batch_size 1 \
                         --gpu_id 6 \
                         --include_eye_gaze False \
                         --eye_gaze_dimension 64 \
                         --eye_gaze_scale_factor 1 \
                         --print_freq 50 \
                         --gaze_D6_rotation False \
                         --eye_gaze_disentangle False \
                         --comment 'new brach feature/learn_feat_disparity, use eye gaze dimension 64,add template loss(non-eye region),eye
                         loss only without head loss version 3, no vgg loss' 

