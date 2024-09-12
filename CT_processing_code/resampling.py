from pathlib import Path
import os
from tqdm import tqdm
import medviz as viz
import pdb

inputpath = Path(r'D:\pc\cwru\ct_stricture\my_processing\cropped_scans')
input_files=inputpath.glob("*.nii")

outputpath = Path(r'D:\pc\cwru\ct_stricture\my_processing\resampled\volume')

input_files = list(input_files)
num_inputs = len(input_files)
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
for input_file in tqdm(input_files, total=num_inputs, desc="Progress"):
    print(f"Resampling {input_file.name}...")
    resampled_path = outputpath
    viz.resample(path=input_file, out_path=resampled_path, voxel_size=[1,1,1],method="trilinear")