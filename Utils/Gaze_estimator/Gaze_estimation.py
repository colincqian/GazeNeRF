import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from Utils.Gaze_estimator.model import gaze_network
from glob import glob
import h5py

from Utils.Gaze_estimator.head_pose import HeadPoseEstimator

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def denormalize_gaze(img, face_model, hr, ht, cam, gaze):
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (256, 256)  # size of cropped eye image
    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R



def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (224,224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, landmarks_warped

def generate_face_gaze_images(img_dir,save_dir):
    for img_file_name in glob(os.path.join(img_dir,'*.JPG')) + glob(os.path.join(img_dir,'*.png')) :
            print('######load input face image#######: ', img_file_name)
            image = cv2.imread(img_file_name)

            face_patch_gaze,pred_gaze_np = face_gaze_estimiator(image,normalized_input=False)

            output_path = os.path.join(save_dir,os.path.basename(img_file_name).replace('.png','_results.png'))
            print('save output image to: ', output_path)

            cv2.putText(img=face_patch_gaze, text=str(pred_gaze_np), org=(0, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 255),thickness=1)

            cv2.imwrite(output_path, face_patch_gaze)

def face_gaze_estimiator(image,normalized_input=False,load_self_defined_camera=False,**kwargs):
        predictor = dlib.shape_predictor('Utils/Gaze_estimator/modules/shape_predictor_68_face_landmarks.dat')
        # face_detector = dlib.cnn_face_detection_model_v1('/home/colinqian/Project/ETH-XGaze/ETH-XGaze/modules/mmod_human_face_detector.dat')
        face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful
        detected_faces = face_detector(image, 1)
        if len(detected_faces) == 0:
            print('warning: no detected face')
            exit(0)
 
        shape = predictor(image, detected_faces[0]) ## only use the first detected face (assume that each input image only contains one face)
        shape = face_utils.shape_to_np(shape)
        landmarks = []
        for (x, y) in shape:
            landmarks.append((x, y))
        landmarks = np.asarray(landmarks)

        # load camera information
        if load_self_defined_camera:
            camera_matrix = kwargs['camera_matrix']
            camera_distortion = kwargs['camera_distortion']
        else:
            cam_file_name = '/home/colinqian/Project/ETH-XGaze/ETH-XGaze/example/input/cam00.xml'  # this is camera calibration information file obtained with OpenCV
            if not os.path.isfile(cam_file_name):
                print('no camera calibration file is found.')
                exit(0)
            fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
            camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalizations
            camera_distortion = fs.getNode('Distortion_Coefficients').mat()
        
        # load face model
        face_model_load = np.loadtxt('Utils/Gaze_estimator/models/face_model.txt')  # Generic face model with 3D facial landmarks
        landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
        face_model = face_model_load[landmark_use, :]
        # estimate the head pose,
        ## the complex way to get head pose information, eos library is required,  probably more accurrated
        # cam_id = 0
        # landmarks = landmarks.reshape(-1, 2)
        # head_pose_estimator = HeadPoseEstimator()
        # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix[cam_id])
        ## the easy way to get head pose information, fast and simple
        facePts = face_model.reshape(6, 1, 3)
        landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
        landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
        landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
        hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)
        # hr = np.zeros_like(hr)
        # ht[0,0] -= 20
        # ht[1,0] -= 20
        # ht[2,0] -= 20
        # data normalization method
        if normalized_input:
            img_normalized = image
            num_point = landmarks_sub.shape[0]
            landmarks_normalized =  landmarks_sub.reshape(num_point, 2)
        else:
            img_normalized, landmarks_normalized = normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)

        # cv2.imshow('current rendering', img_normalized)
        # cv2.waitKey(0) 
        # #closing all open windows 
        # cv2.destroyAllWindows()  

        model = gaze_network()
        model.cuda() # comment this line out if you are not using GPU
        pre_trained_model_path = 'Utils/Gaze_estimator/models/epoch_24_ckpt.pth.tar'
        if not os.path.isfile(pre_trained_model_path):
            print('the pre-trained gaze estimation model does not exist.')
            exit(0)
        else:
            pass
            #print('load the pre-trained model: ', pre_trained_model_path)
        ckpt = torch.load(pre_trained_model_path)
        model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
        model.eval()  # change it to the evaluation mode
        #input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
        input_var = img_normalized[:, :, [2, 1, 0]]
        input_var = trans(input_var)
        input_var = torch.autograd.Variable(input_var.float().cuda())
        input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
        # import ipdb
        # ipdb.set_trace()
        pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
        pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
        pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array
        #vec_norm = np.linalg.norm(pred_gaze_np)
        #pred_gaze_np = pred_gaze_np/vec_norm

        # draw the facial landmarks
        landmarks_normalized = landmarks_normalized.astype(int) # landmarks after data normalization
        for (x, y) in landmarks_normalized:
            cv2.circle(img_normalized, (x, y), 5, (0, 255, 0), -1)
        face_patch_gaze = draw_gaze(image, pred_gaze_np,color=(255,0,0))  # draw gaze direction on the normalized face image
        # cv2.imshow('current rendering', face_patch_gaze)
        # cv2.waitKey(0) 
        # #closing all open windows 
        # cv2.destroyAllWindows()  
        return face_patch_gaze,pred_gaze_np
    
def test_xgaze_dataset(subject_path,img_index,vis=False,normalized_input=True):
    '''
    subject_path: the path to the subject h5py file
    image index : the image index in the subject file
    vis: whether to visualize the image
    normalized_input: set true if the image in the h5py is already normalized
    '''
    hdf_file = h5py.File(subject_path, 'r', swmr=True)
    assert hdf_file.swmr_mode

    image = hdf_file['face_patch'][img_index]
    # if vis:
    #     cv2.imshow('original image',image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    gaze_label = hdf_file['face_gaze'][img_index]
    gaze_label = gaze_label.astype('float')

    face_patch_gaze, pred_gaze_np = face_gaze_estimiator(image,normalized_input=normalized_input)

    print(f'label gaze:{gaze_label}, pred gaze;{pred_gaze_np}')
    cv2.putText(img=face_patch_gaze, text=str(pred_gaze_np), org=(0, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 255),thickness=1)
    face_patch_gaze = draw_gaze(face_patch_gaze,gaze_label,color=(255,0,0))
    cv2.putText(img=face_patch_gaze, text=str(gaze_label), org=(0, 75), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 0, 0),thickness=1)
    if vis:
        cv2.imshow('result image',face_patch_gaze)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    return 




if __name__ == '__main__':
    img_dir = 'example/input'
    save_dir = 'example/output'
    generate_face_gaze_images(img_dir=img_dir,save_dir=save_dir)

    # subject_path = '/home/colinqian/Project/HeadNeRF/headnerf/XGaze_data/processed_data/processed_subject0000'
    # image_idx = 0
    # test_xgaze_dataset(subject_path,image_idx,vis=True)
    ###############################################
    # img_file_name = './example/input/cam00.JPG'
    # print('load input face image: ', img_file_name)
    # image = cv2.imread(img_file_name)

    # predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    # # face_detector = dlib.cnn_face_detection_model_v1('./modules/mmod_human_face_detector.dat')
    # face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful
    # detected_faces = face_detector(image, 1)
    # if len(detected_faces) == 0:
    #     print('warning: no detected face')
    #     exit(0)
    # print('detected one face')
    # shape = predictor(image, detected_faces[0]) ## only use the first detected face (assume that each input image only contains one face)
    # shape = face_utils.shape_to_np(shape)
    # landmarks = []
    # for (x, y) in shape:
    #     landmarks.append((x, y))
    # landmarks = np.asarray(landmarks)

    # # load camera information
    # cam_file_name = './example/input/cam00.xml'  # this is camera calibration information file obtained with OpenCV
    # if not os.path.isfile(cam_file_name):
    #     print('no camera calibration file is found.')
    #     exit(0)
    # fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    # camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
    # camera_distortion = fs.getNode('Distortion_Coefficients').mat()

    # print('estimate head pose')
    # # load face model
    # face_model_load = np.loadtxt('face_model.txt')  # Generic face model with 3D facial landmarks
    # landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    # face_model = face_model_load[landmark_use, :]
    # # estimate the head pose,
    # ## the complex way to get head pose information, eos library is required,  probably more accurrated
    # # landmarks = landmarks.reshape(-1, 2)
    # # head_pose_estimator = HeadPoseEstimator()
    # # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix[cam_id])
    # ## the easy way to get head pose information, fast and simple
    # facePts = face_model.reshape(6, 1, 3)
    # landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
    # landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    # landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
    # hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

    # # data normalization method
    # print('data normalization, i.e. crop the face image')
    # img_normalized, landmarks_normalized = normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)

    # print('load gaze estimator')
    # model = gaze_network()
    # model.cuda() # comment this line out if you are not using GPU
    # pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
    # if not os.path.isfile(pre_trained_model_path):
    #     print('the pre-trained gaze estimation model does not exist.')
    #     exit(0)
    # else:
    #     print('load the pre-trained model: ', pre_trained_model_path)
    # ckpt = torch.load(pre_trained_model_path)
    # model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
    # model.eval()  # change it to the evaluation mode
    # input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
    # input_var = trans(input_var)
    # input_var = torch.autograd.Variable(input_var.float().cuda())
    # input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
    # pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
    # pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
    # pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array

    # print('prepare the output')
    # # draw the facial landmarks
    # landmarks_normalized = landmarks_normalized.astype(int) # landmarks after data normalization
    # for (x, y) in landmarks_normalized:
    #     cv2.circle(img_normalized, (x, y), 5, (0, 255, 0), -1)
    # face_patch_gaze = draw_gaze(img_normalized, pred_gaze_np)  # draw gaze direction on the normalized face image
    # output_path = 'example/output/results_gaze.jpg'
    # print('save output image to: ', output_path)
    # cv2.imwrite(output_path, face_patch_gaze)
