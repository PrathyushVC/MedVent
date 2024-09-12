import pdb
import os
import nibabel as nib
import numpy as np

vol_folder = r'D:\pc\cwru\ct_stricture\my_processing\resampled\volume'  # output directory for saving cropped scans
lab_folder = r'D:\pc\cwru\ct_stricture\my_processing\resampled\annotations'  # output directory for saving cropped labels

outvol_folder = r'D:\pc\cwru\ct_stricture\my_processing\padded\volume'  # output directory for saving cropped scans
outlab_folder = r'D:\pc\cwru\ct_stricture\my_processing\padded\annotation'  # output directory for saving cropped labels

patients = [i for i in os.listdir(vol_folder) if not i.startswith('.') and i.endswith('.nii')]

labels = [i for i in os.listdir(lab_folder) if not i.startswith('.') and i.endswith('.nii')]

m=np.zeros((1,len(patients)))
n=np.zeros((1,len(patients)))
for index in range(0,len(patients)):
    data = nib.load(vol_folder + os.sep + patients[index])
    volume = data.get_fdata()
    [m[0,index],n[0,index],c]=volume.shape
m_pad=int(np.max(m))
n_pad=int(np.max(n))
for index in range(0,len(patients)):
    data = nib.load(vol_folder + os.sep + patients[index])
    volume = data.get_fdata()
    data_label = nib.load(lab_folder + os.sep + labels[index])
    label_vol = data_label.get_fdata()
    [m_orig,n_orig,c]=volume.shape
    m_center = (m_pad - m_orig) / 2
    n_center = (n_pad - n_orig) / 2
    new_vol=np.pad(volume, ((int(np.floor(m_center)),int(np.ceil(m_center))), (int(np.floor(n_center)), int(np.ceil(n_center))), (0, 0)),'constant', constant_values=-1024)
    new_lab=np.pad(label_vol, ((int(np.floor(m_center)),int(np.ceil(m_center))), (int(np.floor(n_center)), int(np.ceil(n_center))), (0, 0)),'constant', constant_values=0)

    save_img = nib.Nifti1Image(new_vol, np.eye(4)) 
    save_img.to_filename(outvol_folder + os.sep + '{}_padded.nii'.format(os.path.split(patients[index])[1][:-7]))
    save_lab = nib.Nifti1Image(new_lab, np.eye(4)) 
    save_lab.to_filename(outlab_folder + os.sep + '{}_padded_annotation.nii'.format(os.path.split(patients[index])[1][:-7]))


   	