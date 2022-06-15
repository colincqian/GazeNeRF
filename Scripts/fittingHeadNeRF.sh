python -m ipdb FittingSingleImage.py --model_path "logs/ckpt/epoch_24_ckpt.pth" \
                             --img "./XGaze_data/normalized_250_data/000661.png"\
                             --mask "./XGaze_data/normalized_250_data/000661_mask.png" \
                             --para_3dmm "./XGaze_data/normalized_250_data/000661_nl3dmm.pkl" \
                             --save_root "test_data/fitting_res" \
                             --target_embedding "LatentCodeSamples/model_Reso32/S025_E14_I01_P02.pth"


