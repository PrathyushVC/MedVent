import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy.ndimage as ndi
import SimpleITK as sitk
import numpy as np



def find_small_components(lab_files, area_threshold, output_csv,sep='_',loc=0):
    
    """
    This function identifies and records small components in 3D label images based on a specified area threshold.
    
    Parameters:
    - lab_files: A list of file paths to the label images to be processed.
    - area_threshold: The minimum area required for a component to be considered significant.
    - output_csv: The file path where the results will be written in CSV format note the path must exist.
    - sep: The separator used in the filename to split and extract the relevant part for the output CSV.
    - loc: The location of the relevant part in the filename split by the separator.
    
    Returns:
    - creates pandas DataFrame containing the filename and slice number of small components, which is then written to the specified output CSV file.
    
    The function iterates through each label image, processing each slice individually. It applies the ndi.label function to identify connected components in the slice, then calculates the area of each component using ndi.sum. Components with an area below the specified threshold are considered small and their presence is recorded along with the slice number in the output CSV file.
    """
    output_data=[]
    for files in lab_files:
        img = nib.load(files)
        img_data = img.get_fdata()

    
        for z in range(img_data.shape[2]):
            slice_data = img_data[:, :, z]
            
        
            labeled_slice, num_features = ndi.label(slice_data)
            
        
            component_areas = ndi.sum(slice_data > 0, labeled_slice, range(1, num_features + 1))
            
        
            small_components = [i for i, area in enumerate(component_areas, 1) if area < area_threshold]
            
            
            if small_components:
                output_data.append({'filename': os.path.basename(files).split(sep)[loc], 'slice': z})
    df = pd.DataFrame(output_data)
    df.to_csv(output_csv, index=False)
    print(f'Results written to {output_csv}')

def plot_kde_vols(data,save_loc=None):
  """
  Plots a Kernel Density Estimate (KDE) for the given data.

  Parameters:y
  - data: numpy array, the data to be plotted.

  Returns:
  - None

  This function flattens the input data and plots its KDE using seaborn.
  """
  data_flat = data.flatten()
  plt.figure(figsize=(8, 6))
  sns.kdeplot(data_flat, bw_adjust=0.5)
  plt.title('Data Distribution (KDE)')
  plt.xlabel('Data Value')
  plt.ylabel('Density')
  plt.grid(True)
  if save_loc is None:
      plt.show()
  else:
      save_path = Path(save_loc)
      if save_path.parent.exists():
          plt.savefig(save_path)
          print(f"Plot saved to {save_path}")
      else:
          raise FileNotFoundError(f"The directory {save_path.parent} does not exist.")


def plot_multiple_kde(df,save_loc=None):
  """
  Plots Kernel Density Estimates (KDE) for volumes of multiple patients.

  Parameters:
  - df: pandas DataFrame, containing 'patient_id' and 'Volumes' columns.

  Returns:
  - None

  This function iterates through each unique patient ID in the DataFrame, loads the corresponding volume data,
  flattens it, and plots its KDE using seaborn.
  """
  unique_patient_ids = df['patient_id'].unique()
  plt.figure(figsize=(8, 6))
  plt.title('Data Distribution (KDE)')
  plt.xlabel('Data Value')
  plt.ylabel('Density')
  for patient_id in unique_patient_ids:
      filtered_df = df[df['patient_id'] == patient_id]
      volume_path = Path(filtered_df['Volumes'].values[0])

      volume = nib.load(volume_path).get_fdata()

      data_flat = volume.flatten()
      sns.kdeplot(data_flat, bw_adjust=0.5)
      plt.grid(True)
  if save_loc is None:
      plt.show()
  else:
      save_path = Path(save_loc)
      if save_path.parent.exists():
          plt.savefig(save_path)
          print(f"Plot saved to {save_path}")
      else:
          raise FileNotFoundError(f"The directory {save_path.parent} does not exist.")


def check_meta(image):#SITK supported image
    """
    Checks returns meta data.

    Parameters:
    - image: SimpleITK.Image or str, the input image to check. It can be a SimpleITK image or a path to a DICOM file.

    Returns:
    - meta_info: dict, a dictionary containing metadata information of the image, including:
        - Image Size
        - Image Spacing (voxel size)
        - Image Origin
        - Image Direction (orientation)
        - Pixel ID
        - Metadata Dictionary
    """
    if isinstance(image, sitk.Image):
        print("Input is a SimpleITK supported image.")
    elif isinstance(image, str) and image.lower().endswith('.dcm'):
        print("Input is a DICOM file.")
        image = sitk.ReadImage(image)
    else:
        raise ValueError("Unsupported image format. Please provide a SimpleITK image or a DICOM file.")
    meta_info = {
        "Image Size": image.GetSize(),
        "Image Spacing (voxel size)": image.GetSpacing(),
        "Image Origin": image.GetOrigin(),
        "Image Direction (orientation)": image.GetDirection(),
        "Pixel ID": image.GetPixelIDTypeAsString(),
        "Metadata Dictionary": {key: image.GetMetaData(key) for key in image.GetMetaDataKeys()}
    }

    return meta_info

def get_orientation(affine):
    """
    Function to provide a rough guess of image orientation based on the affine matrix.

    Parameters:
    - matrix from .nii affine field:

    Returns:
    - String indicating the orientation of the image.
    """
    # Extract the orientation of each axis from the affine matrix
    orientation = np.linalg.norm(affine[:3, :3], axis=0)
    x, y, z = orientation

    # Assuming that if one dimension is significantly larger, it's likely the slice thickness (axial)
    if np.isclose(x, y) and z > max(x, y):
        return 'Axial'
    elif np.isclose(x, z) and y > max(x, z):
        return 'Sagittal'
    elif np.isclose(y, z) and x > max(y, z):
        return 'Coronal'
    else:
        return 'Unknown'