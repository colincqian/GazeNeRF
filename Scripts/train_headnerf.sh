python -m ipdb train_headnerf.py --headnerf_options "TrainedModels/model_Reso32HR.pth" \
                         --batch_size 2 \
                         --gpu_id 6 \
                         --include_eye_gaze False \
                         --eye_gaze_dimension 64 \
                         --eye_gaze_scale_factor 1 \
                         --print_freq 10 \
                         --gaze_D6_rotation False \
                         --eye_gaze_disentangle False \
                         --comment 'try add gaze feat before vol-rendering, use eye mask data info, 
                                        use full head template loss and data loss, eye loss weight 10,remove density' 

