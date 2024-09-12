"""
This package provides a VolumeVisualizer class that can be used to visualize 3D volumes and their masks. It supports both CPU and GPU acceleration, and can read volumes from .mha, .nii, and numpy data. The visualizer can display a single slice of the volume, with optional mask and heatmap. It also supports interactive rotation of the volume.

"""
from .VolumeVisualizer import  *
from .check_data import *
from .process_utils import *

__version__ = "1.1.2"