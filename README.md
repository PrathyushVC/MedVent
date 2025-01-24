# Image_Processing_Demo
 INVent radiology image processing 

## Components

### Data Processing Utilities

Located in `loc_utils/process_utils.py`, this module contains functions for scanning directories for volume and label files, resampling patient data, applying bias field correction to MRI images, and additional data processing utilities. Key functions include:

- `directory_scan`: Scans specified directories for volume and label files based on provided substrings.
- `resample_patient_data`: Resamples the volume and mask data for each patient.
- `BFC_with_mask`: Applies N4ITK Bias Field Correction to an MRI image with an optional mask.
- `normalize_data`: Normalizes the volume data to a specified range.
- `filter_outliers`: Filters outliers from the volume data based on specified criteria.

### Data Quality Check

The `check_data` module located in `loc_utils/check_data.py` provides functions to validate the integrity and quality of the MRI and CT datasets. Key functionalities include:

- `validate_data_format`: Checks if the data files are in the correct format.
- `check_missing_values`: Identifies any missing values in the datasets.
- `report_data_statistics`: Generates statistics for the datasets to assess quality.


### Volume Visualization

The `VolumeVisualizer` class in `loc_utils/VolumeVisualizer.py` provides an interactive interface for visualizing 3D MRI volumes. It allows users to:

- Rotate the volume.
- Select slices to view.
- Display the volume with optional masks.
- Export visualizations as images

### Machine Learning Pipeline

The machine learning pipeline is implemented in `MRI processing_code/Machine_learning_pipe.ipynb`. This component includes:

- Data preparation and feature extraction from MRI data.
- Model training using Random Forest Classifier.
- Evaluation metrics such as accuracy, F1 score, and ROC AUC.

### CT Variability Analysis

The `CT_Variability_analysis.ipynb` notebook provides tools for analyzing variability in CT data. It includes:

- Loading and processing CT datasets.
- Performing statistical tests to compare different datasets.
- Visualizing variability through graphical representations.

### Annotation Check

The `Annotation_check.ipynb` notebook is designed to validate and check the annotations in the MRI data. It includes functions to identify small components in 3D label images based on specified area thresholds and provides detailed reporting on annotation quality.

## Usage

To use the package, you can run the Jupyter notebooks provided in the `MRI processing_code` and `CT_processing_code` directories. Each notebook contains detailed instructions and examples for using the various components of the package.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
