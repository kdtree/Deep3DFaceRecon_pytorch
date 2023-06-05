"""
This script is the test script for Deep3DFaceRecon_pytorch
Test for video
"""

import os
import time
from options.test_video_options import TestVideoOptions
# from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
# from data.flist_dataset import default_flist_reader
# from scipy.io import loadmat, savemat
import cv2
from openface_utls import OpenFaceCSVReader

def read_video_frames(video_path):
    """Read video frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def read_video_data_with_openface(
        file_name,
        input_video_dir, # video->frames
        input_openface_dir # get lm5 from openface lm68
        ):
    """get image and lm5 from video and openface csv"""
    video_path = os.path.join(input_video_dir, file_name + '.mp4')
    csv_path = os.path.join(input_openface_dir, file_name + '.csv')
    if not os.path.exists(csv_path) or not os.path.exists(video_path):
        print(f'{file_name} is not available, skip.')
        return
    of_csv = OpenFaceCSVReader(csv_path)
    lm2d_5pts = of_csv.get_landmarks2d_5pts()
    frames = read_video_frames(video_path)
    if lm2d_5pts.shape[0] != len(frames):
        print(f'{file_name} has different number of frames, skip.')
        return
    return frames, lm2d_5pts

def run_instance(
        rank,
        opt,
        file_name_list,
        input_video_dir,
        input_openface_dir,
        output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)
    lm3d_std = load_lm3d(opt.bfm_folder)
    for file_name in file_name_list:
        print('Processing ', file_name)
        t0 = time.time()
        frames, lm2d_5pts = read_video_data_with_openface(
            file_name, input_video_dir, input_openface_dir)
        n_frames = len(frames)
        f_width = frames[0].shape[1]
        f_height = frames[0].shape[0]
        id_list = []
        exp_list = []
        tex_list = []
        angle_list = []
        gamma_list = []
        trans_list = []
        res_coeff_dict = {}
        output_file_dir = os.path.join(output_dir, file_name)
        os.makedirs(output_file_dir, exist_ok=True)
        for i in range(n_frames):
            # print('processing frame ', i)
            lm = lm2d_5pts[i]
            lm[:, -1] = f_height - 1 - lm[:, -1]
            _, im, lm, _ = align_img(Image.fromarray(frames[i]), lm, lm3d_std)
            im_tensor = torch.tensor(
                np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            lm_tensor = torch.tensor(lm).unsqueeze(0)
            data = {
                'imgs': im_tensor,
                'lms': lm_tensor
            }
            model.set_input(data)  # unpack data from data loader
            if i % 100 == 0:
                model.test()
                visuals = model.get_current_visuals()  # get image results
                visualizer.display_current_results(visuals, 0, opt.epoch, dataset=file_name, 
                save_results=True, count=i, name=f'{i:08d}', add_image=False,
                save_path_override=output_file_dir)
            else:
                model.test_no_render()

            coeff_dict = model.get_coeff_np()
            for key in coeff_dict:
                if key not in res_coeff_dict:
                    res_coeff_dict[key] = []
                res_coeff_dict[key].append(coeff_dict[key])
        
        for key in res_coeff_dict:
            res_coeff_dict[key] = np.concatenate(res_coeff_dict[key], axis=0)
            np.save(os.path.join(output_file_dir, f'{key}.npy'), res_coeff_dict[key])

        t1 = time.time()
        print('processing time: ', t1 - t0)

if __name__ == '__main__':
    opt = TestVideoOptions().parse()  # get test options
    fn_list = []
    if opt.video_folder is not None:
        for f in os.listdir(opt.video_folder):
            if f.endswith('.mp4'):
                fn_list.append(f[:-4])

        run_instance(0, opt, fn_list, opt.video_folder, opt.openface_folder, opt.output_folder)
                
    
