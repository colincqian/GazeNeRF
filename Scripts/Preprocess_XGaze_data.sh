# # generating head's mask.
#python DataProcess/Gen_HeadMask.py --img_dir "XGaze_data/normalized_250_data"

# # generating 68-facial-landmarks by face-alignment, which is from 
# # https://github.com/1adrianb/face-alignment
#python DataProcess/Gen_Landmark.py --img_dir "XGaze_utils/xgaze_resize250"

# generating the 3DMM parameters
python Fitting3DMM/FittingNL3DMM.py --img_size 250 \
                                    --intermediate_size 125  \
                                    --batch_size 1 \
                                    --img_dir "XGaze_data/playground"
