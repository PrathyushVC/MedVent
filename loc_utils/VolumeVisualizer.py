import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import ipywidgets as widgets
from ipywidgets import interact
from matplotlib import cm
import SimpleITK as sitk
import nibabel as nib
from IPython.display import display

class VolumeVisualizer:
    def __init__(self, vol_data, mask_data=None, use_gpu=False, cmap='gray', coloraxis=None, figsize=(10, 10)):
        """
        Initialize the VolumeVisualizer class.

        Parameters:
        - vol_data: str, the path to the 3D volume file (.mha, .nii, or .mat).
        - mask_data: str, the path to the mask file, if any.
        - use_gpu: bool, whether to use GPU acceleration (requires CuPy).
        - cmap: str, the colormap to use for displaying the slices.
        - coloraxis: tuple, min and max limits for the color display.
        - figsize: tuple, the size of the figure to display.
        """

        self.vol_data = vol_data
        self.mask_data = mask_data
        self.cmap = cmap
        self.coloraxis = coloraxis
        self.figsize = figsize

        # Load the volume and mask
        self.volume = self._validate_data(data=self.vol_data)
        self.mask = self._validate_data(data=self.mask_data) if mask_data else None

        # Initialize rotation angle and rotated volume
        self.rotation_angle = 0
        self.num_slices = self.volume.shape[2]
        self.rotated_volume = self.volume.copy()  # Copy of the original volume to rotate

        self.setup_figure()

    def read_image(self, data):
        """
        Reads a 3D volume from a file path. Supported formats include .mha, .nii, .nii.gz, and .npy.

        Parameters:
        - data: str or numpy array, the path to the 3D volume file or the volume data itself.

        Returns:
        - numpy array, the 3D volume data.
        """
        if data is None:
            return None
        
        # Define loaders for each supported file type
        loaders = {
            '.mha': lambda path: sitk.GetArrayFromImage(sitk.ReadImage(path)),
            '.nii': lambda path: nib.load(path).get_fdata(),
            '.nii.gz': lambda path: nib.load(path).get_fdata(),
            '.npy': np.load,
        }

        # Get file extension and use appropriate loader
        file_extension = '.' + data.split('.')[-1].lower()
        if file_extension in loaders:
            return loaders[file_extension](data)

        if isinstance(data, np.ndarray):
            return data

        raise ValueError("Unsupported file format. Use .mha, .nii, .nii.gz, or .npy format.")

    def setup_figure(self):
        """Set up the interactive figure and controls."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.button_left = widgets.Button(description="Rotate Left")
        self.button_right = widgets.Button(description="Rotate Right")
        self.slice_slider = widgets.IntSlider(min=1, max=self.num_slices, value=self.num_slices // 2)

        # Link button events to call the update function
        self.button_left.on_click(self.rotate_left)
        self.button_right.on_click(self.rotate_right)

        # Use interact to automatically display the slider and link it to the update function
        interact(self.update_display, slice_idx=self.slice_slider)

        # Display the rotation buttons
        display(widgets.HBox([self.button_left, self.button_right]))

    def update_display(self, slice_idx):
        """Update the display for a given slice index, taking into account the current rotation."""
        self.ax.clear()
        im = self.rotated_volume[:, :, slice_idx - 1]
        self.ax.imshow(im, cmap=self.cmap, vmin=self.coloraxis[0] if self.coloraxis else 0,
                       vmax=self.coloraxis[1] if self.coloraxis else 1000)
        self.ax.set_title(f'Slice {slice_idx} (Rotated {self.rotation_angle} degrees)')

    def rotate_left(self, _):
        """Rotate the entire volume 90 degrees to the left and update the display."""
        self.rotation_angle = (self.rotation_angle - 90) % 360
        self.rotated_volume = np.rot90(self.rotated_volume, k=1, axes=(0, 1))
        self.update_display(self.slice_slider.value)  # Update the display with the current slice

    def rotate_right(self, _):
        """Rotate the entire volume 90 degrees to the right and update the display."""
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.rotated_volume = np.rot90(self.rotated_volume, k=-1, axes=(0, 1))
        self.update_display(self.slice_slider.value)  # Update the display with the current slice

    def _validate_data(self, data):
        """Validate and return the volume data based on its type."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, nib.Nifti1Image):
            return data.get_fdata()
        elif isinstance(data, str):
            return self.read_image(data)
        else:
            raise TypeError(f"Expected a string, but got {type(data).__name__}.")
