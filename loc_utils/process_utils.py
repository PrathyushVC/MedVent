import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import nibabel as nib
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from skimage.measure  import find_contours
from tqdm import tqdm
import seaborn as sns



                
def directory_scan(data_dir_vol,data_dir_labels, substrings_vol,substrings_labels,pat_idname='patient_id',img_colname='Volumes',label_colname='Masks'):
    """
    Scans a directory for volume and label files that contain any of the provided substrings,
    extracts patient IDs, and returns a DataFrame with file information.

    Parameters:
    -data_dir_vol: str, path to the directory containing the volume files
    -data_dir_labels: str, path to the directory containing the label files
    -substrings: list of str, substrings to filter files that contain any of them

    Returns:
    -DataFrame with columns: patient_id, vol_file_path, lab_file_path
    """

    all_files_vol = os.listdir(data_dir_vol)
    all_files_lab = os.listdir(data_dir_labels)

    vol_files = [file for file in all_files_vol if any(sub in file for sub in substrings_vol)]
    lab_files = [file for file in all_files_lab if any(sub in file for sub in substrings_labels)]


    file_info = []

    for vol_file in vol_files:
        patient_id = vol_file.split('_')[0]  #Assumes ID is the first block of the string which due to renameming is always true

        lab_file = next((file for file in lab_files if patient_id in file), None)

        file_info.append({
            pat_idname: patient_id,
            img_colname: os.path.join(data_dir_vol, vol_file),
            label_colname: os.path.join(data_dir_labels, lab_file) if lab_file else None
        })


    return pd.DataFrame(file_info)

