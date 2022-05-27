# generating head's mask.
python DataProcess/Gen_HeadMask.py --img_dir "XGaze_data"

# generating 68-facial-landmarks by face-alignment, which is from 
# https://github.com/1adrianb/face-alignment
python DataProcess/Gen_Landmark.py --img_dir "XGaze_data"

# generating the 3DMM parameters
python Fitting3DMM/FittingNL3DMM.py --img_size 400 \
                                    --intermediate_size 200  \
                                    --batch_size 9 \
                                    --img_dir "XGaze_data"
