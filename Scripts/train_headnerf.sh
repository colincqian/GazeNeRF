python train_headnerf.py --headnerf_options "TrainedModels/model_Reso32HR.pth" \
                         --batch_size 2 \
                         --gpu_id 3 \
                         --include_eye_gaze True \
                         --eye_gaze_dimension 64 \
                         --eye_gaze_scale_factor 2 \
                         --comment 'train with displacement loss, add eye gaze scale factor 2'
