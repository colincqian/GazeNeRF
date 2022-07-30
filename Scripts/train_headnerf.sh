python train_headnerf.py --headnerf_options "TrainedModels/model_Reso32HR.pth" \
                         --gpu_id 3 \
                         --batch_size 4 \
                         --include_eye_gaze True \
                         --eye_gaze_dimension 64 \
                         --comment 'train without disen module, gaze (normalized) only concatenated to shape code'
