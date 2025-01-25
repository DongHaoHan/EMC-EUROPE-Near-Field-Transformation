import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import h5py
from Model import DCNN
import time
# Load and Prepare Data
with h5py.File('Data.h5', 'r') as file:
    Dataset_input_1 = np.array(file['Magnitude_3G_10mm'])
    Dataset_input_2 = np.array(file['Labels_height'])
    Dataset_output = np.array(file['Magnitude_pre'])
  
# Split data into training, validation, and testing sets

Num_data = Dataset_input_1.shape[0]
Train_end = int(0.9 * Num_data)
Val_end = int(0.95 * Num_data)

Train_input_1 = Dataset_input_1[:Train_end]
Val_input_1 = Dataset_input_1[Train_end:Val_end]
Test_input_1 = Dataset_input_1[Val_end:]

Train_input_2 = Dataset_input_2[:Train_end]
Val_input_2 = Dataset_input_2[Train_end:Val_end]
Test_input_2 = Dataset_input_2[Val_end:]

Train_output = Dataset_output[:Train_end]
Val_output = Dataset_output[Train_end:Val_end]
Test_output = Dataset_output[Val_end:]

Train_input_tensor_1 = torch.tensor(Train_input_1, dtype=torch.float32)
Train_input_tensor_2 = torch.tensor(Train_input_2, dtype=torch.long)
Val_input_tensor_1 = torch.tensor(Val_input_1, dtype=torch.float32)
Val_input_tensor_2 = torch.tensor(Val_input_2, dtype=torch.long)
Test_input_tensor_1 = torch.tensor(Test_input_1, dtype=torch.float32)
Test_input_tensor_2 = torch.tensor(Test_input_2, dtype=torch.long)

Train_output_tensor = torch.tensor(Train_output, dtype=torch.float32)
Val_output_tensor = torch.tensor(Val_output, dtype=torch.float32)
Test_output_tensor = torch.tensor(Test_output, dtype=torch.float32)

Batch_size = int(0.001 * Train_input_1.shape[0])
Train_loader = DataLoader(TensorDataset(Train_input_tensor_1, Train_input_tensor_2, Train_output_tensor),
                          batch_size=Batch_size, shuffle=True)
Val_loader = DataLoader(TensorDataset(Val_input_tensor_1, Val_input_tensor_2, Val_output_tensor),
                        batch_size=Batch_size, shuffle=False)
# Initialize Model, Loss, and Optimizer

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = DCNN().to(Device)
Criterion = nn.MSELoss()
lr = 0.001
Optimizer = optim.Adam(Model.parameters(), lr=lr)

Num_epochs = 4
Train_losses = []
Val_losses = []
start_time = time.time()

# Training Loop

for epoch in range(Num_epochs):
    Model.train()
    Running_train_loss = 0.0
    for batch_input_1, batch_input_2, batch_targets in Train_loader:
        batch_input_1 = batch_input_1.to(Device)
        batch_input_2 = batch_input_2.to(Device)
        batch_targets = batch_targets.to(Device)

        Optimizer.zero_grad()
        outputs = Model(batch_input_1, batch_input_2)
        loss = Criterion(outputs, batch_targets)
        loss.backward()
        Optimizer.step()
        Running_train_loss += loss.item() * batch_input_1.size(0)

    epoch_train_loss = Running_train_loss / len(Train_loader.dataset)
    Train_losses.append(epoch_train_loss)

    Model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_input_1, batch_input_2, batch_targets in Val_loader:
            batch_input_1 = batch_input_1.to(Device)
            batch_input_2 = batch_input_2.to(Device)
            batch_targets = batch_targets.to(Device)
            outputs = Model(batch_input_1, batch_input_2)
            loss = Criterion(outputs, batch_targets)
            running_val_loss += loss.item() * batch_input_1.size(0)

    epoch_val_loss = running_val_loss / len(Val_loader.dataset)
    Val_losses.append(epoch_val_loss)

    lr *= 0.98
    for param_group in Optimizer.param_groups:
        param_group['lr'] = lr

    print(f"Epoch [{epoch + 1}/{Num_epochs}] - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total Training Time: {total_time:.2f} seconds")
# Plot Training and Validation Loss
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22
plt.figure(figsize=(10, 6))
plt.plot(range(1, Num_epochs + 1), Train_losses, label="Training loss", linewidth=2.5)
plt.plot(range(1, Num_epochs + 1), Val_losses, label="Validation loss", linewidth=2.5)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss over epochs")
plt.legend()
plt.grid()
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.savefig('Loss.png', dpi=600)

#Save Trained Model

model_save_path = 'Trained_model.pth'
torch.save(Model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")
plt.show()
