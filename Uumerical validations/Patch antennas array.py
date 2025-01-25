import numpy as np
import torch
import matplotlib.pyplot as plt
from Model import DCNN  

# Function to calculate relative error between predicted and true values
def relative(Pre, Tru):
    Pre = np.abs(Pre)
    Tru = np.abs(Tru)
    error = np.sqrt(np.sum((Pre - Tru) ** 2) / np.sum(Tru ** 2))  # Relative error formula
    return error

# Load the trained model

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = DCNN().to(Device)
Model.load_state_dict(torch.load("Trained_model.pth", weights_only=True))
Model.eval()  # Set model to evaluation mode

# Define grid dimensions
MM, NN = 41, 41

# Load and preprocess data 
with open('scanning_Spiral_line_3GHz_10mm.near', 'r') as file:
    Field = file.readlines()
Field = [list(map(float, line.split())) for line in Field]
Field = np.array(Field)
Hx_3G_10mm = abs(Field[:, 9] + 1j * Field[:, 10])  # Extract Hx field component
factor = Hx_3G_10mm.max()  # Normalization factor
Hx_3G_10mm = Hx_3G_10mm / factor  # Normalize data
Hx_3G_10mm = Hx_3G_10mm.reshape(MM, NN)  
Input = torch.tensor(Hx_3G_10mm[None, None, :, :], dtype=torch.float32).to(Device)  # Convert to tensor

with open('scanning_Spiral_line_3GHz_5mm.near', 'r') as file:
    Field = file.readlines()
Field = [list(map(float, line.split())) for line in Field]
Field = np.array(Field)
True_Mag_3G_5mm = abs(Field[:, 9] + 1j * Field[:, 10]) 
True_Mag_3G_5mm = True_Mag_3G_5mm / factor  
True_Mag_3G_5mm = True_Mag_3G_5mm.reshape(MM, NN)  

with open('scanning_Spiral_line_3GHz_15mm.near', 'r') as file:
    Field = file.readlines()
Field = [list(map(float, line.split())) for line in Field]
Field = np.array(Field)
True_Mag_3G_15mm = abs(Field[:, 9] + 1j * Field[:, 10])  
True_Mag_3G_15mm = True_Mag_3G_15mm / factor  
True_Mag_3G_15mm = True_Mag_3G_15mm.reshape(MM, NN) 

# Define height labels for prediction
label_height_15mm = torch.tensor([5], dtype=torch.long).to(Device)  # 15mm
label_height_5mm = torch.tensor([0], dtype=torch.long).to(Device)  # 5mm

# Predict magnitudes for 5mm and 15mm heights
Pre_Mag_3G_15mm = Model(Input, label_height_15mm).cpu().detach().numpy().squeeze()
Pre_Mag_3G_5mm = Model(Input, label_height_5mm).cpu().detach().numpy().squeeze()

# Calculate and print relative errors
print("Relative error 3G 5mm:", relative(Pre_Mag_3G_5mm, True_Mag_3G_5mm))
print("Relative error 3G 15mm:", relative(Pre_Mag_3G_15mm, True_Mag_3G_15mm))

# Plot settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25

# Plot predicted magnitude at 5mm
plt.figure()
plt.imshow(Pre_Mag_3G_5mm * factor,
          extent=(-0.1 * 1e3, 0.1 * 1e3, -0.1 * 1e3, 0.1 * 1e3),
          origin='upper',
          aspect='auto',
          cmap='jet',
          interpolation='bicubic')
plt.colorbar()
plt.xlabel('X Axis (mm)')
plt.ylabel('Y Axis (mm)')
plt.title(f"Predicted Hx at 5 mm")
plt.tight_layout()
plt.savefig(f'Array Predicted Hx at 5 mm.png', dpi=300)

# Plot ground truth magnitude at 5mm
plt.figure()
plt.imshow(True_Mag_3G_5mm * factor,
          extent=(-0.1 * 1e3, 0.1 * 1e3, -0.1 * 1e3, 0.1 * 1e3),
          origin='upper',
          aspect='auto',
          cmap='jet',
          interpolation='bicubic')
plt.colorbar()
plt.xlabel('X Axis (mm)')
plt.ylabel('Y Axis (mm)')
plt.title(f"Ground truth Hx at 5 mm")
plt.tight_layout()
plt.savefig(f'Array Ground truth Hx at 5 mm.png', dpi=300)

# Plot predicted magnitude at 15mm
plt.figure()
plt.imshow(Pre_Mag_3G_15mm * factor,
          extent=(-0.1 * 1e3, 0.1 * 1e3, -0.1 * 1e3, 0.1 * 1e3),
          origin='upper',
          aspect='auto',
          cmap='jet',
          interpolation='bicubic')
plt.colorbar()
plt.xlabel('X Axis (mm)')
plt.ylabel('Y Axis (mm)')
plt.title(f"Predicted Hx at 15 mm")
plt.tight_layout()
plt.savefig(f'Array Predicted Hx at 15 mm.png', dpi=300)

# Plot ground truth magnitude at 15mm
plt.figure()
plt.imshow(True_Mag_3G_15mm * factor,
          extent=(-0.1 * 1e3, 0.1 * 1e3, -0.1 * 1e3, 0.1 * 1e3),
          origin='upper',
          aspect='auto',
          cmap='jet',
          interpolation='bicubic')
plt.colorbar()
plt.xlabel('X Axis (mm)')
plt.ylabel('Y Axis (mm)')
plt.title(f"Ground truth Hx at 15 mm")
plt.tight_layout()
plt.savefig(f'Array Ground truth Hx at 15 mm.png', dpi=300)

plt.show()
