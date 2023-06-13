"""
This script is the test script for Deep3DFaceRecon_pytorch
Test for video, only dumps error.txt for temp usage
"""

import os
import time
import torch
import torch.multiprocessing as mp
from PIL import Image
import numpy as np
import cv2
from options.test_video_options import TestVideoOptions
# from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from util.load_mats import load_lm3d
from openface_utls import OpenFaceCSVReader
# import torch.distributed as dist

# pylint: disable=no-member, invalid-name, unspecified-encoding
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
        return None, None
    of_csv = OpenFaceCSVReader(csv_path)
    conf_val = of_csv.get_confidence()
    if np.min(conf_val) < 0.1:
        # Currently just discard sequence with low confidence frames
        print(f'{file_name} contains low confidence frames, skip.')
        return None, None
    lm2d_5pts = of_csv.get_landmarks2d_5pts()
    frames = read_video_frames(video_path)
    if lm2d_5pts.shape[0] != len(frames):
        print(f'{file_name} has different number of frames, skip.')
        return None, None
    return frames, lm2d_5pts

def run_instance(
        process_idx,
        n_processes,
        run_opt,
        file_name_list,
        input_video_dir,
        input_openface_dir,
        output_dir):
    """"run a single instance"""
    os.makedirs(output_dir, exist_ok=True)
    n_gpus = torch.cuda.device_count()
    device = torch.device(process_idx % n_gpus)
    print('use device:', device)
    torch.cuda.set_device(device)
    model = create_model(run_opt)
    model.setup(run_opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(run_opt)
    lm3d_std = load_lm3d(run_opt.bfm_folder)
    f_list = file_name_list[process_idx::n_processes]
    for file_name in f_list:
        try:
            print('Processing ', file_name)
            output_file_dir = os.path.join(output_dir, file_name)
            # if os.path.exists(os.path.join(output_file_dir, 'error.txt')):
            #     print(f'{file_name} already processed, skip.')
            #     continue
            frames, lm2d_5pts = read_video_data_with_openface(
                file_name, input_video_dir, input_openface_dir)
            if frames is None or lm2d_5pts is None:
                print(f'{file_name} is not available, skip.')
                continue
            n_frames = len(frames)
            # f_width = frames[0].shape[1]
            f_height = frames[0].shape[0]
            errors = []
            os.makedirs(output_file_dir, exist_ok=True)
            for i in range(n_frames):
                if i % 30 == 0 and os.name != 'nt': # ~1 sec for 30fps
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
                    model.test()
                    color_loss = model.compute_color_loss_for_test()
                    errors.append(color_loss)

            # save error to txt
            avg_error = np.average(errors)
            with open(os.path.join(output_file_dir, 'error.txt'), 'w') as f_:
                f_.write(f'{avg_error:.6f}')
            f_.close()
            print('average color loss: ', avg_error)
        # pylint: disable=bare-except
        except:
            print('error occurred, skip.')

if __name__ == '__main__':
    opt = TestVideoOptions().parse()  # get test options
    fn_list = []
    if opt.video_folder is not None:
        for f in os.listdir(opt.video_folder):
            if f.endswith('.mp4'):
                fn_list.append(f[:-4])
        # to ensure the order of processing
        fn_list.sort()
        # Only process the specified sub-part, for multi-machine processing
        if opt.n_parts > 1:
            # pylint: disable=invalid-name
            tot_len = len(fn_list)
            part_len = (tot_len + opt.n_parts-1) // opt.n_parts
            start_idx = opt.part_id * part_len
            end_idx = min((opt.part_id + 1) * part_len, tot_len)
            fn_list = fn_list[start_idx:end_idx]
        print("Number of videos to process: ", len(fn_list))
        # run_instance(0, opt, fn_list, opt.video_folder, opt.openface_folder, opt.output_folder)
        N_PROCESSES = 1 # opt.world_size
        mp.spawn(run_instance, args=(
            N_PROCESSES, opt, fn_list, opt.video_folder,
            opt.openface_folder, opt.output_folder),
            nprocs=N_PROCESSES, join=True)
