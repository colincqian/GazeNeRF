python FittingSingleImage.py --model_path "TrainedModels/model_Reso32HR.pth" \
                             --img "./XGaze_data/normalized_250_data/000661.png"\
                             --mask "./XGaze_data/normalized_250_data/000661_mask.png" \
                             --para_3dmm "./XGaze_data/normalized_250_data/000661_nl3dmm.pkl" \
                             --save_root "test_data/fitting_res" \
                             --target_embedding "LatentCodeSamples/*/S025_E14_I01_P02.pth"


