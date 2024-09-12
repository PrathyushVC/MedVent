import nibabel as nib
import pandas as pd
def find_small_components(lab_files, area_threshold, output_csv,sep='_',loc=0):
    import scipy.ndimage as ndi
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

def plot_kde_vols(data):
  data_flat = data.flatten()
  plt.figure(figsize=(8, 6))
  sns.kdeplot(data_flat, bw_adjust=0.5)
  plt.title('Data Distribution (KDE)')
  plt.xlabel('Data Value')
  plt.ylabel('Density')
  plt.grid(True)
  plt.show()


def plot_multiple_kde(df):
  unique_patient_ids = df['patient_id'].unique()
  plt.figure(figsize=(8, 6))
  plt.title('Data Distribution (KDE)')
  plt.xlabel('Data Value')
  plt.ylabel('Density')
  for patient_id in unique_patient_ids:
      filtered_df = df[df['patient_id'] == patient_id]
      volume_path =volume_temp= Path(filtered_df['Volumes'].values[0])

      volume = nib.load(volume_path).get_fdata()

      data_flat = volume.flatten()
      sns.kdeplot(data_flat, bw_adjust=0.5)
      plt.grid(True)
  plt.show()