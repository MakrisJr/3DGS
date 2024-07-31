import os
import shutil
import json

import numpy as np
import pandas as pd

def copy_images(train_imgs, test_imgs, src_dir, dst_dir):
    # Create destination directories:
    # dst_dir
    # ├── train
    # └── test
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.exists(dst_dir + '/train'):
        os.makedirs(dst_dir + '/train')
    if not os.path.exists(dst_dir + '/test'):
        os.makedirs(dst_dir + '/test')

    # Copy train images
    for img_num in train_imgs:
        src_path = src_dir + f'/test/r_{img_num}.png'
        dst_path = dst_dir + f'/train/r_{img_num}.png'
        shutil.copy2(src_path, dst_path)
        print(f'Copied {src_path} to {dst_path}')

    # Copy test images
    for img_num in test_imgs:
        src_path = src_dir + f'/test/r_{img_num}.png'
        dst_path = dst_dir + f'/test/r_{img_num}.png'
        shutil.copy2(src_path, dst_path)
        print(f'Copied {src_path} to {dst_path}')


def filter_frames(frames, image_numbers):
    filtered_frames = []
    for frame in frames:
        # Extract the image number from the file path
        image_number = int(frame["file_path"].split('_')[-1])
        if image_number in image_numbers:
            filtered_frames.append(frame)
    return filtered_frames


def create_filtered_json(train_imgs, test_imgs, src_dir, dst_dir):
    # Create transofrms_train.json
    with open(os.path.join(src_dir, 'transforms_test.json'), 'r') as file:
        data = json.load(file)

    train_frames = filter_frames(data["frames"], train_imgs)
    
    train_json_data = {
        "camera_angle_x": data["camera_angle_x"],
        "frames": train_frames
    }
    
    with open(os.path.join(dst_dir, 'transforms_train.json'), 'w') as file:
        json.dump(train_json_data, file, indent=4)

    # Create transofrms_test.json
    with open(os.path.join(src_dir, 'transforms_test.json'), 'r') as file:
        data = json.load(file)
    
    test_frames = filter_frames(data["frames"], test_imgs)

    test_json_data = {
        "camera_angle_x": data["camera_angle_x"],
        "frames": test_frames
    }

    with open(os.path.join(dst_dir, 'transforms_test.json'), 'w') as file:
        json.dump(test_json_data, file, indent=4)


def generate_sine_wave_samples(start_period, end_period, window_size, dst_dir):
    num_windows = end_period - start_period
    total_samples = num_windows * window_size

    x = np.linspace(0, 2 * np.pi * num_windows, total_samples)
    y = np.sin(x)

    data = {
        f'sample_{i}': y[i::window_size] for i in range(window_size)
    }
    index = np.arange(start_period, end_period)

    df = pd.DataFrame(data, index=index)
    df.index.name = 'period'
    csv_path = os.path.join(dst_dir, 'sine_wave_samples.csv')
    df.to_csv(csv_path)
    print(f'Sine wave samples saved to {csv_path}')



if __name__ == '__main__':
    # SPECIFY THE TRAIN/TEST IMAGES
    start_period = 30
    end_period = 50
    sampling_ratio = 0.5
    train_imgs = [i for i in range(30, 50, int(1/sampling_ratio))]
    test_imgs = [i for i in range(30, 50)]
    print(test_imgs)

    # SPECIFY THE SOURCE AND DESTINATION DIRECTORIES FOR THE TRAIN/TEST IMAGES
    src_dir = '/home/anthosmakris/GS_exp/3DGS/data/nerf_synthetic/chair_src'
    dst_dir = '/home/anthosmakris/GS_exp/3DGS/data/nerf_synthetic/chair'


    copy_images(train_imgs, test_imgs, src_dir, dst_dir)
    create_filtered_json(train_imgs, test_imgs, src_dir, dst_dir)
    generate_sine_wave_samples(30, 50, 15, dst_dir)

