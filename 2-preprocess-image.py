import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from os.path import join
from os import makedirs, listdir

LUNG_WINDOW_LEVEL = -300
LUNG_WINDOW_WIDTH = 1400

def window_image(image, window_level, window_width):
    img_min = window_level - (window_width // 2)
    img_max = window_level + (window_width // 2)

    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

# min max normalize
def min_max_norm(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image

def crop_and_resample(data, data_dir, output_size):
    pbar = tqdm(df['Filename'])
    
    resampled_dir = join(data_dir, 'lung-window-roi-npy')
    os.makedirs(resampled_dir, exist_ok=True)

    for i, fname in enumerate(pbar):
        img_sitk = sitk.ReadImage(join(data_dir, 'LIDC-IDRI', fname))
        img_npy = np.load(join(data_dir, 'lung-window-npy', fname.replace('.nii.gz', '.npy')), 'r', allow_pickle=True)

        spacing = img_sitk.GetSpacing()
        size = img_sitk.GetSize()

        # center and radius of the cube
        center = (65, 65, 33)
        radius_mm = df['Diameter'][i]  # radius in millimeters
        physical_radius = radius_mm

        # convert center to physical coordinates
        center_physical = [center[i] * spacing[i] for i in range(3)]

        # calculate start and end coordinates in physical space
        start_physical = [center_physical[i] - physical_radius for i in range(3)]
        end_physical = [center_physical[i] + physical_radius for i in range(3)]

        # convert physical coordinates to voxel indices
        start_index = [int(start_physical[i] / spacing[i]) for i in range(3)]
        end_index = [int(end_physical[i] / spacing[i]) for i in range(3)]

        # ensure indices are within bounds
        start_index = [max(0, idx) for idx in start_index]
        end_index = [min(size[i] - 1, idx) for i, idx in enumerate(end_index)]

        # calculate roi size
        roi_size = [end_index[i] - start_index[i] + 1 for i in range(3)]

        # extract the region
        roi = sitk.RegionOfInterest(img_sitk, roi_size, start_index)

        # roi output size in voxels 
        roi_new_size = output_size

        # physical size of original roi
        roi_size = roi.GetSize()
        roi_spacing = roi.GetSpacing()
        roi_physical_size = [roi_size[i] * roi_spacing[i] for i in range(3)]

        # new spacing based on output voxel size
        roi_new_spacing = [roi_physical_size[i] / roi_new_size[i] for i in range(3)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(roi_new_size)
        resampler.SetOutputSpacing(roi_new_spacing)
        resampler.SetOutputOrigin(roi.GetOrigin())
        resampler.SetOutputDirection(roi.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)

        roi_resampled = resampler.Execute(roi)
        roi_resampled_npy = sitk.GetArrayFromImage(roi_resampled)
        
        # out_file = os.path.join(data_dir, 'lung-window-roi-npy', fname)
        # sitk.WriteImage(roi_resampled, out_file)
        
        out_file = os.path.join(data_dir, 'lung-window-roi-npy', fname.replace('.nii.gz', '.npy'))
        np.save(out_file, roi_resampled_npy)

        pbar.set_description(f"{roi_resampled_npy.shape}, {roi_resampled.GetSpacing()}")
        
if __name__ == "__main__":
    data_dir = './data'
    df = pd.read_csv('dataset_with_indeterminate.csv')[['Filename', 'Diameter']]
    pbar = tqdm(df['Filename'])

    stretched_npy_dir = join(data_dir, "lung-window-npy")
    os.makedirs(stretched_npy_dir, exist_ok=True)
    
    for fname in pbar:
        img_sitk = sitk.ReadImage(join(data_dir, 'LIDC-IDRI', fname))
        img_npy = sitk.GetArrayFromImage(img_sitk)

        lung_window_npy = window_image(img_npy, LUNG_WINDOW_LEVEL, LUNG_WINDOW_WIDTH)
        np.save(os.path.join(data_dir, 'lung-window-npy', fname.replace(".nii.gz", ".npy")), lung_window_npy)

        pbar.set_description(f'{lung_window_npy.max()}, {lung_window_npy.min()}')
    
    output_size = (48, 48, 48)
    crop_and_resample(data=df, data_dir=data_dir, output_size=output_size)