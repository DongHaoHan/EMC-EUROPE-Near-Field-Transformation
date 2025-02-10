import numpy as np
import torch
import matplotlib.pyplot as plt
from Model import DCNN
from torch.utils.data import DataLoader, TensorDataset
import h5py

# Function to calculate relative error between predicted and reference values
def relative(Pre, Tru):
    Pre = np.abs(Pre)
    Tru = np.abs(Tru)
    error = np.sqrt(np.sum((Pre - Tru) ** 2) / np.sum(Tru ** 2))  # Relative error formula
    return error

# Set device (GPU if available, otherwise CPU)
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
Model = DCNN().to(Device)
Model.load_state_dict(torch.load("Trained_model.pth", weights_only=True))

# Load data from HDF5 file
with h5py.File('Data.h5', 'r') as file:
    Dataset_input_1 = np.array(file['Magnitude_3G_10mm'])  # Main input data
    Dataset_input_2 = np.array(file['Labels_height'])      # Height labels
    Dataset_output = np.array(file['Magnitude_pre'])       # Target output data

    # Load additional magnitude data for different heights
    Magnitudes_ = {}
    for key in file.keys():
        if key not in ['Magnitude_3G_10mm', 'Magnitude_pre', 'Labels_height', 'Labels_fre']:
            Magnitudes_[key] = np.array(file[key])

# Split data into training, validation, and test sets
Num_data = Dataset_input_1.shape[0]
Train_end = int(0.9 * Num_data)
Val_end = int(0.95 * Num_data)

# Prepare test data
Test_input = Dataset_input_1[Val_end:]
Test_input_tensor = torch.tensor(Test_input, dtype=torch.float32)
Test_loader = DataLoader(TensorDataset(Test_input_tensor), batch_size=1, shuffle=False)

# Set model to evaluation mode
Model.eval()
with torch.no_grad():
    for i, (test_input,) in enumerate(Test_loader):
        if i == 1:
            # Create height labels for different scanning heights
            label_height_5mm = torch.tensor([0], dtype=torch.long).to(Device)  # 5mm
            label_height_7mm = torch.tensor([1], dtype=torch.long).to(Device)  # 7mm
            label_height_9mm = torch.tensor([2], dtype=torch.long).to(Device)  # 9mm
            label_height_11mm = torch.tensor([3], dtype=torch.long).to(Device)  # 11mm
            label_height_13mm = torch.tensor([4], dtype=torch.long).to(Device)  # 13mm
            label_height_15mm = torch.tensor([5], dtype=torch.long).to(Device)  # 15mm
            test_input1 = test_input.to(Device)

            # Predict magnitudes for different heights
            Pre_Mag_3G_5mm = Model(test_input1, label_height_5mm).cpu().detach().numpy().squeeze()
            Pre_Mag_3G_7mm = Model(test_input1, label_height_7mm).cpu().detach().numpy().squeeze()
            Pre_Mag_3G_9mm = Model(test_input1, label_height_9mm).cpu().detach().numpy().squeeze()
            Pre_Mag_3G_11mm = Model(test_input1, label_height_11mm).cpu().detach().numpy().squeeze()
            Pre_Mag_3G_13mm = Model(test_input1, label_height_13mm).cpu().detach().numpy().squeeze()
            Pre_Mag_3G_15mm = Model(test_input1, label_height_15mm).cpu().detach().numpy().squeeze()

            # Get reference magnitudes for comparison
            index = int(0.95 * Num_data + i)
            Reference_Mag_3G_5mm = Magnitudes_['3G_5mm'][index, 0, :, :]
            Reference_Mag_3G_7mm = Magnitudes_['3G_7mm'][index, 0, :, :]
            Reference_Mag_3G_9mm = Magnitudes_['3G_9mm'][index, 0, :, :]
            Reference_Mag_3G_11mm = Magnitudes_['3G_11mm'][index, 0, :, :]
            Reference_Mag_3G_13mm = Magnitudes_['3G_13mm'][index, 0, :, :]
            Reference_Mag_3G_15mm = Magnitudes_['3G_15mm'][index, 0, :, :]

            # Calculate and print relative errors
            print("Relative error 3G 5mm:", relative(Pre_Mag_3G_5mm, Reference_Mag_3G_5mm))
            print("Relative error 3G 7mm:", relative(Pre_Mag_3G_7mm, Reference_Mag_3G_7mm))
            print("Relative error 3G 9mm:", relative(Pre_Mag_3G_9mm, Reference_Mag_3G_9mm))
            print("Relative error 3G 11mm:", relative(Pre_Mag_3G_11mm, Reference_Mag_3G_11mm))
            print("Relative error 3G 13mm:", relative(Pre_Mag_3G_13mm, Reference_Mag_3G_13mm))
            print("Relative error 3G 15mm:", relative(Pre_Mag_3G_15mm, Reference_Mag_3G_15mm))

            # Plot and save reference and predicted magnitudes
            heights = [5, 7, 9, 11, 13, 15]
            Reference_mags = [Reference_Mag_3G_5mm, Reference_Mag_3G_7mm, Reference_Mag_3G_9mm, Reference_Mag_3G_11mm, Reference_Mag_3G_13mm, Reference_Mag_3G_15mm]
            pred_mags = [Pre_Mag_3G_5mm, Pre_Mag_3G_7mm, Pre_Mag_3G_9mm, Pre_Mag_3G_11mm, Pre_Mag_3G_13mm, Pre_Mag_3G_15mm]

            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 25

            for height, Reference_mag, pred_mag in zip(heights, Reference_mags, pred_mags):
                # Plot reference
                plt.figure()
                plt.imshow(Reference_mag,
                           extent=(-0.1 * 1e3, 0.1 * 1e3, -0.1 * 1e3, 0.1 * 1e3),
                           origin='upper',
                           aspect='auto',
                           cmap='jet',
                           interpolation='bicubic')
                plt.colorbar()
                plt.xlabel('X Axis (mm)')
                plt.ylabel('Y Axis (mm)')
                plt.title(f"Reference Hx at {height} mm")
                plt.tight_layout()
                plt.savefig(f'Reference Hx at {height} mm.png', dpi=300)
                plt.close()

                # Plot predicted magnitude
                plt.figure()
                plt.imshow(pred_mag,
                           extent=(-0.1 * 1e3, 0.1 * 1e3, -0.1 * 1e3, 0.1 * 1e3),
                           origin='upper',
                           aspect='auto',
                           cmap='jet',
                           interpolation='bicubic')
                plt.colorbar()
                plt.xlabel('X Axis (mm)')
                plt.ylabel('Y Axis (mm)')
                plt.title(f"Predicted Hx at {height} mm")
                plt.tight_layout()
                plt.savefig(f'Predicted Hx at {height} mm.png', dpi=300)
                plt.close()

            break  # Only process one sample for demonstration
