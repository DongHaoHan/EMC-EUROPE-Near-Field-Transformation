# 1.Introduction
This repository contains the implementation code for the paper titled "Deep Learning-Assisted Phaseless Near-Field Transformation for Accelerating Near-Field Scanning."
# 2.Usage Instructions
1) Execute 'Data_generation.py' in the Data Generation directory. Transfer the generated data file (Data.h5) to both the DCNN Training and Numerical Validation directories.
2) Run 'DCNN_training.py' in the DCNN Training directory. Move the trained model file (Trained_model.pth) to the Numerical Validation directory.
3) Execute the following scripts to produce the results:
'Testing_data.py'
'Patch_antennas_array.py'
# 3.Notes
1) Data used in the paper can be downloaded at https://drive.google.com/file/d/1_kEGpv4SX3bGnHktuQ0XS75VaxRhP3i8/view?usp=sharing
2) Trained_model.pth is available for direct download and can be utilized for testing without requiring additional training.
# 4.Maintainers
This project is owned and managed by Dong-Hao Han and Xing-Chang Wei from Zhejiang University, China.
