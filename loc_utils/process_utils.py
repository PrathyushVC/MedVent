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
from .check_data import check_meta
import seaborn as sns
import warnings



                
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



def resample(input_file,resampled_path, voxel_size= [1.0, 1.0, 1.0], method= "trilinear"):
    """
    Resamples an image to the specified voxel size using the given method.

    Parameters:
    - input_file: str, path to the input image file.
    - resampled_path: str, path to the directory where the resampled image will be saved.
    - voxel_size: list of float, the target voxel size for resampling.
    - method: str, the resampling method to use ("nearest" or "trilinear").

    Returns:
    - sitk.Image: The resampled SimpleITK image.

    Notes:
    Prints out the original and resampled size and spacing information.
    """

    print(f"Resampling {input_file}...")
    itk_image = sitk.ReadImage(input_file)
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    print(f"OG Size: {original_size}")
    print(f"OG Space: {original_spacing}")
    out_size = [int(np.round(original_size[0] * (original_spacing[0] / voxel_size[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / voxel_size[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / voxel_size[2])))]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(voxel_size)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if method == "nearest":
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    elif method == "trilinear":
        resample.SetInterpolator(sitk.sitkLinear)
    else:
        raise ValueError("Unknown interpolation method")
    try:
      resampled_image = resample.Execute(itk_image)
      print(f"resampled_size: {resampled_image.GetSize()}")
      print(f"resampled_spacing: {resampled_image.GetSpacing()}")

      sitk.WriteImage(resampled_image, resampled_path)
      print(f"Saved to {resampled_path}")
    except Exception as e:
          print(f"\ Error resampling {input_file}: {e}")

def resample_patient_data(df,volume_resampled_path,mask_resampled_path,resample_to=[1.0,1.0,1.0]):
    """
    Function to loop through a DataFrame and load files for each patient for processing.

    Parameters:
    - df: pandas DataFrame with 'patient_id', 'Volumes', and 'Masks' columns.
    - volume_resampled_path: str, path to the directory where resampled volumes will be saved.
    - mask_resampled_path: str, path to the directory where resampled masks will be saved.

    Returns:
    - patient_data: Dictionary containing patient_id as keys and tuples of (volume, mask) NIfTI objects as values.
    """

    unique_patient_ids = df['patient_id'].unique()
    #recast as Path type
    volume_resampled_path=Path(volume_resampled_path)
    mask_resampled_path=Path(mask_resampled_path)


    volume_resampled_path.mkdir(parents=True, exist_ok=True)
    mask_resampled_path.mkdir(parents=True, exist_ok=True)

    for patient_id in tqdm(unique_patient_ids, desc="Processing patients"):
        print(f"Processing patient ID: {patient_id}")
        filtered_df = df[df['patient_id'] == patient_id]
        if not filtered_df.empty:

          volume_path =volume_temp= Path(filtered_df['Volumes'].values[0])
          mask_path=mask_temp = Path(filtered_df['Masks'].values[0])

          print(volume_path.stem)

          try:

              volume_suffix = ''.join(volume_path.suffixes)  # Join all suffixes
              while volume_temp.suffix:
                volume_temp = volume_temp.with_suffix('')#In cases where processing .nii.gz or other zipped formats.
              volume_filename = Path(volume_temp.stem + f"_resampled" + volume_suffix)
              volume_path_out = volume_resampled_path / volume_filename

              print(volume_path_out)
              resample(input_file=volume_path, resampled_path=volume_path_out, voxel_size=resample_to,method="trilinear")

          except Exception as e:
              print(f"\generating paths for patient {patient_id}: {e}")

          try:
              mask_suffix = ''.join(mask_path.suffixes)  # Join all suffixes
              while mask_temp.suffix:
                mask_temp = mask_temp.with_suffix('')#In cases where processing .nii.gz or other zipped formats.
              mask_filename = Path(mask_temp.stem + f"_resampled" + mask_suffix)
              mask_path_out = mask_resampled_path / mask_filename

              print(mask_path_out)
              resample(input_file=mask_path, resampled_path=mask_path_out, voxel_size=resample_to,method="nearest")

          except Exception as e:
              print(f"\generating paths for patient {patient_id}: {e}")

def BFC_with_mask(input_image_path, output_image_path, mask_image_path=None, threshold_method='li', bins=200, shrink_factor=1,check_params=False):
    """
    Applies N4ITK Bias Field Correction to an MRI image with an optional mask.

    Parameters:
    - input_image_path (str): Path to the input NIfTI image.
    - output_image_path (str): Path to save the corrected output image.
    - mask_image_path (str, optional): Path to a mask image to restrict the bias field estimation.
    - threshold_method (str, optional): Thresholding method to use ('otsu' or 'li').
    - bins (int, optional): Number of histogram bins for Otsu thresholding.
    - shrink_factor (int, optional): Factor by which to shrink the image during bias field correction for faster processing.
    - check params(bool, optional): Print the pre and post dims to verify safe behavior
    Returns:
    - corrected_image (SimpleITK.Image): The bias-corrected image, or None if an error occurred.
    """
    try:
        print(f"Loading input image from {input_image_path}...")
        input_image = sitk.ReadImage(input_image_path)


        input_image_float32 = sitk.Cast(input_image, sitk.sitkFloat32)
        input_image_float32.CopyInformation(input_image)


        if shrink_factor > 1:
            print(f"Applying shrink factor of {shrink_factor} before thresholding...")
            shrunk_image = sitk.Shrink(input_image_float32, [shrink_factor] * input_image_float32.GetDimension())
        else:
            shrunk_image=input_image_float32# for variable name clarity


        if mask_image_path:
            print(f"Loading mask image from {mask_image_path}...")
            mask_image = sitk.ReadImage(mask_image_path)
        else:

            if threshold_method.lower() == 'otsu':
                print("Creating mask using Otsu thresholding...")
                mask_image = sitk.OtsuThreshold(shrunk_image, 0, 1, bins)
            elif threshold_method.lower() == 'li':
                print("Creating mask using Li thresholding...")
                mask_image = sitk.LiThreshold(shrunk_image, 0, 1)
            else:
                raise ValueError("Unsupported threshold method. Choose 'otsu' or 'li'.")
        if shrink_factor > 1:
            print("Rescaling mask back to original size and matching size with input image...")
            mask_image = sitk.Expand(mask_image, [shrink_factor] * mask_image.GetDimension())
            mask_image.SetSpacing(input_image_float32.GetSpacing())
            mask_image.SetOrigin(input_image_float32.GetOrigin())
            mask_image.SetDirection(input_image_float32.GetDirection())
            mask_image = sitk.Resample(mask_image, input_image_float32, sitk.Transform(), sitk.sitkNearestNeighbor, input_image_float32.GetOrigin(), input_image_float32.GetSpacing(), input_image_float32.GetDirection(), 0, mask_image.GetPixelID())

        print("Performing N4ITK Bias Field Correction...")
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(input_image_float32, mask_image)


        print(f"Saving corrected image to {output_image_path}...")
        sitk.WriteImage(corrected_image, output_image_path)


        print("Bias field correction completed successfully.")
        meta_input=check_meta(input_image)
        meta_cor=check_meta(corrected_image)
        print("Comparing metadata of input and corrected images...")
        for key in meta_input:
            if meta_input[key] != meta_cor[key]:
                print(f"Difference found in {key}:")
                print(f"  Input image: {meta_input[key]}")
                print(f"  Corrected image: {meta_cor[key]}")

        return corrected_image

    except Exception as e:
        raise RuntimeError(f"Error during bias field correction: {e}.")
    
def BFC_patient_data(df,BF_path,mask_image_path=None,bins=200):
    """

    """
    warnings.warn('This processing step will convert all images to type Float 32')
    unique_patient_ids = df['patient_id'].unique()
    #recast as Path type
    volume_BF_path=Path(BF_path)
    if mask_image_path is not None:
      mask_BF_path=Path(mask_image_path)


    volume_BF_path.mkdir(parents=True, exist_ok=True)

    for patient_id in tqdm(unique_patient_ids, desc="Processing patients"):
        print(f"Processing patient ID: {patient_id}")
        filtered_df = df[df['patient_id'] == patient_id]
        if not filtered_df.empty:

          volume_path = volume_temp = Path(filtered_df['Volumes'].values[0])
          mask_path = mask_temp = Path(filtered_df['Masks'].values[0])

          try:
              volume_suffix = ''.join(volume_path.suffixes)  # Join all suffixes
              while volume_temp.suffix:
                volume_temp = volume_temp.with_suffix('')#In cases where processing .nii.gz or other zipped formats.
              volume_filename = Path(volume_temp.stem + f"_BF" + volume_suffix)
              volume_path_out = volume_BF_path / volume_filename

              print(volume_path_out)
              BFC_with_mask(input_image_path=volume_path, output_image_path=volume_path_out, mask_image_path=mask_image_path,bins=bins)

          except Exception as e:
              raise RuntimeError(f"Error generating paths for patient {patient_id}: {e}")
def pull_data_pid(df,patient_id):
  """
  Function returns the data for the volume and mask of the data frame with PID.

  Parameters:
  - df: pandas DataFrame with 'patient_id', 'Volumes', and 'Masks' columns.
  -patient_id: PID to search for
  """
  filtered_df = df[df['patient_id'] == patient_id]


  if not filtered_df.empty:
      volume_path = filtered_df['Volumes'].values[0]
      mask_path = filtered_df['Masks'].values[0]
      print(volume_path)
      print(mask_path)

      # Load the volume file
      if volume_path.endswith('.nii') or volume_path.endswith('.nii.gz'):
          volume = nib.load(volume_path)
      elif volume_path.endswith('.dcm'):
          volume = sitk.ReadImage(volume_path)
      elif volume_path.endswith('.mha'):
          volume = sitk.ReadImage(volume_path)
      else:
          raise ValueError("Unsupported volume file format")

      # Load the mask file
      if mask_path.endswith('.nii') or mask_path.endswith('.nii.gz'):
          mask = nib.load(mask_path)
      elif mask_path.endswith('.dcm'):
          mask = sitk.ReadImage(mask_path)
      elif mask_path.endswith('.mha'):
          mask = sitk.ReadImage(mask_path)
      else:
          raise ValueError("Unsupported mask file format")

      return (volume, mask)
  else:
      print("No matching PID found.")









                