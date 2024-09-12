

import pdb
import nibabel as nib
from skimage import exposure as ex
import numpy as np
from skimage.morphology import convex_hull_image
from skimage.filters import threshold_otsu
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from skimage.morphology import binary_closing, disk
import cv2 
from skimage import measure
# foreground and background function
def foreground_detector(img):
    thresh_val = threshold_otsu(img)
    binary_image = img > thresh_val
    filtered = remove_small_objects(binary_image, min_size=500)
    #smoothed = binary_closing(filtered, footprint=disk(5))
    labels_mask = measure.label(filtered)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    mask = labels_mask
    return mask

def process_directories(patient_dir,labels_dir,outputvol_folder,outputlab_folder,output_folder, log_folder='log_folder'):
    patients = [i for i in os.listdir(patient_dir) if not i.startswith('.') and i.endswith('.nii.gz')]
    labels = [i for i in os.listdir(labels_dir) if not i.startswith('.') and i.endswith('.nii.gz')]

    for index in range(0,len(patients)):
        data = nib.load(patient_dir + os.sep + patients[index])
        volume = data.get_fdata()
        data_label = nib.load(labels_dir + os.sep + labels[index])
        label_vol = data_label.get_fdata()
        mask_volume = np.zeros_like(volume, dtype=np.uint16)
        if np.min(volume)<-1024:
            with open(os.path.join(log_folder,"logging_file.txt"), "a") as log_file:
                log_file.write(f"Patient {patients[index]} has a minimum value less than -1024.\n")
            continue
        for i in range(0,np.shape(volume)[2]):
        #for i in range(1):
            img = volume[:,:,i]
            fore = foreground_detector(img)
            '''
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(volume[y_min:y_max,x_min:x_max,1], cmap='gist_gray')#Inverse Gray
            ax[0].set_title('Original Image')
            ax[1].imshow(mask_volume[y_min:y_max,x_min:x_max,1], cmap='gist_gray')
            ax[1].set_title('Binary Image')
            plt.show()
            '''
            fore= np.multiply(fore, 1)
            mask_volume[:,:,i] = fore 
        seg_value = 1
        segmentation = np.where(mask_volume == seg_value)
        # Bounding Box
        bbox = [0, 0, 0, 0]
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
            bbox = [x_min, x_max, y_min, y_max]

        mask_volume[y_min:y_max,x_min:x_max,:]=1
        im=volume[y_min:y_max,x_min:x_max,:]
        lab=label_vol[y_min:y_max,x_min:x_max,:]
        save_img = nib.Nifti1Image(im, np.eye(4)) 
        save_img.to_filename(outputvol_folder + os.sep + '{}_cropped.nii'.format(os.path.split(patients[index])[1][:-7]))
        save_lab = nib.Nifti1Image(lab, np.eye(4)) 
        save_lab.to_filename(outputlab_folder + os.sep + '{}_cropped_annotation.nii'.format(os.path.split(patients[index])[1][:-7]))
        save_mask = nib.Nifti1Image(mask_volume, np.eye(4)) 
        save_mask.to_filename(output_folder + os.sep + '{}_foreground_mask.nii'.format(os.path.split(patients[index])[1][:-7]))



# main directory for reading data
def __main__():
    root = r'D:\cwru\ct_stricture\cte_unprocessed\Complete_CTE_NII'
    label=r'D:\cwru\ct_stricture\cte_unprocessed\CTE_annotations_NG_phase1'

    output_folder = r'D:\cwru\ct_stricture\my_processing\forward_mask'  # output directory for saving foreground mask

    outputvol_folder = r'D:\cwru\ct_stricture\my_processing\cropped_scans'  # output directory for saving cropped scans
    outputlab_folder = r'D:\cwru\ct_stricture\my_processing\cropped_scans_labels'  # output directory for saving cropped labels
###########################################################################################################
