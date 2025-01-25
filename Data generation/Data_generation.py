import numpy as np
import h5py
import matplotlib.pyplot as plt
import time

tic = time.time()

# Parameters
Num_data = 20000  # Number of Training Data
PM_num_min = 1  # Minimum number of points
PM_num_max = 5  # Maximum number of points
H_dipole = 0.001  # Dipole height
H_scan_10mm = 0.010  # Scanning height at 10mm
H_scan_pre = np.arange(0.005, 0.017, 0.002)
Fre_scan_3G = 3e9  # Fixed frequency at 3 GHz

MM, NN = 41, 41  # Number of scanning points in x and y directions
Scanning_point = MM * NN
Scanning_step = 0.005  # Scanning step
c = 2.99792458e8  # Speed of light
u = np.pi * 4e-7  # Permeability
epso_0 = 1e-9 / (36 * np.pi)  # Permittivity
Wave_impedance = np.sqrt(u / epso_0)  # Wave impedance

Dipole_moment_max_P = Wave_impedance  # Max dipole moment for Pz
Dipole_moment_min_P = -Dipole_moment_max_P
Dipole_moment_max_M = 1  # Max dipole moment for Mx and My
Dipole_moment_min_M = -Dipole_moment_max_M

# Generate coordinates for scanning points
X_coor = np.linspace(-Scanning_step * (MM - 1) / 2, Scanning_step * (MM - 1) / 2, MM)
Y_coor = np.linspace(-Scanning_step * (NN - 1) / 2, Scanning_step * (NN - 1) / 2, NN)
X_coor, Y_coor = np.meshgrid(X_coor, Y_coor)
X_flat = X_coor.flatten()
Y_flat = Y_coor.flatten()

X_min, X_max = X_coor.min(), X_coor.max()
Y_min, Y_max = Y_coor.min(), Y_coor.max()

# Data storage arrays
Magnitude_3G_10mm = np.zeros((Num_data, 1, MM, NN))
Magnitude_pre = np.zeros((Num_data, 1, MM, NN))
Labels_height = np.zeros((Num_data,), dtype=int)  # Array to store height labels

# Arrays for additional fields (frequencies and heights)
Magnitudes_ = {
    f"3G_{height}mm": np.zeros((Num_data, 1, MM, NN))
    for height in range(5, 16, 2)
}

# Function to compute the field
def compute_field(H_scan, fre):
    k = 2 * np.pi * fre / c  # Wave number for the given frequency
    Z1 = H_scan - H_dipole
    Z2 = H_scan + H_dipole

    r1 = np.sqrt(X ** 2 + Y ** 2 + Z1 ** 2)
    r2 = np.sqrt(X ** 2 + Y ** 2 + Z2 ** 2)

    fr1 = np.exp(-1j * k * r1) / r1
    fr2 = np.exp(-1j * k * r2) / r2

    g1r1 = (3 / (k * r1) ** 2 + 1j * 3 / (k * r1) - 1) * fr1
    g1r2 = (3 / (k * r2) ** 2 + 1j * 3 / (k * r2) - 1) * fr2

    g2r1 = (2 / (k * r1) ** 2 + 1j * 2 / (k * r1)) * fr1
    g2r2 = (2 / (k * r2) ** 2 + 1j * 2 / (k * r2)) * fr2

    g3r1 = (1 / (k * r1) + 1j) * fr1
    g3r2 = (1 / (k * r2) + 1j) * fr2

    HxMx = k ** 2 / (4 * np.pi) * (-(Y ** 2 + Z1 ** 2) / r1 ** 2 * g1r1 + g2r1 - (Y ** 2 + Z2 ** 2) / r2 ** 2 * g1r2 + g2r2)
    HxMy = k ** 2 / (4 * np.pi) * (X * Y / r1 ** 2 * g1r1 + X * Y / r2 ** 2 * g1r2)
    HxPz = k / (4 * np.pi) * (-Y / r1 * g3r1 - Y / r2 * g3r2)

    T = np.concatenate((HxMx, HxMy, HxPz), axis=1)
    Hx_dipole = np.dot(T, Dipoles)
    return Hx_dipole.reshape(MM, NN)

# Main loop
for data_num in range(Num_data):
    PM_num = np.random.randint(PM_num_min, PM_num_max + 1)

    X_init = (X_min + (X_max - X_min) * np.random.rand(PM_num, 1)) * 0.5
    Y_init = (Y_min + (Y_max - Y_min) * np.random.rand(PM_num, 1)) * 0.5

    Pz_real = Dipole_moment_min_P + (Dipole_moment_max_P - Dipole_moment_min_P) * np.random.rand(PM_num, 1)
    Pz_imag = Dipole_moment_min_P + (Dipole_moment_max_P - Dipole_moment_min_P) * np.random.rand(PM_num, 1)

    Mx_real = Dipole_moment_min_M + (Dipole_moment_max_M - Dipole_moment_min_M) * np.random.rand(PM_num, 1)
    My_real = Dipole_moment_min_M + (Dipole_moment_max_M - Dipole_moment_min_M) * np.random.rand(PM_num, 1)
    Mx_imag = Dipole_moment_min_M + (Dipole_moment_max_M - Dipole_moment_min_M) * np.random.rand(PM_num, 1)
    My_imag = Dipole_moment_min_M + (Dipole_moment_max_M - Dipole_moment_min_M) * np.random.rand(PM_num, 1)

    X_G = np.concatenate((X_init, Y_init, Mx_real, Mx_imag, My_real, My_imag, Pz_real, Pz_imag), axis=1)
    X_dipole = X_G[:, 0]
    Y_dipole = X_G[:, 1]

    X_dipole = np.tile(X_dipole, (Scanning_point, 1))
    Y_dipole = np.tile(Y_dipole, (Scanning_point, 1))

    X = np.tile(X_flat, (PM_num, 1)).T - X_dipole
    Y = np.tile(Y_flat, (PM_num, 1)).T - Y_dipole

    Mx = (X_G[:, 2] + 1j * X_G[:, 3]).reshape(-1, 1)
    My = (X_G[:, 4] + 1j * X_G[:, 5]).reshape(-1, 1)
    Pz = (X_G[:, 6] + 1j * X_G[:, 7]).reshape(-1, 1)
    Dipoles = np.concatenate([Mx, My, Pz], axis=0)

    # Compute and store normalized 3G 10mm magnitude
    Hx_10mm = compute_field(H_scan_10mm, Fre_scan_3G)
    Hx_10mm_normalized = Hx_10mm / np.max(np.abs(Hx_10mm))
    Magnitude_3G_10mm[data_num, 0, :, :] = np.abs(Hx_10mm_normalized)

    H_pre = np.random.choice(H_scan_pre)
    height_index = np.where(H_scan_pre == H_pre)[0][0]
    Labels_height[data_num] = height_index

    Hx_pre = compute_field(H_pre, Fre_scan_3G)
    Hx_pre_normalized = Hx_pre / np.max(np.abs(Hx_10mm))
    Magnitude_pre[data_num, 0, :, :] = np.abs(Hx_pre_normalized)

    # For the last 5% of data, compute and save all Magnitude_*G_*mm
    if data_num >= int(0.95 * Num_data):
        for height in range(5, 17, 2):  # Heights 5mm to 15mm
            Hx = compute_field(height / 1e3, Fre_scan_3G)
            Hx_normalized = Hx / np.max(np.abs(Hx_10mm))
            key = f"3G_{height}mm"
            Magnitudes_[key][data_num, 0, :, :] = np.abs(Hx_normalized)

# Save to HDF5
with h5py.File('Data.h5', 'w') as hf:
    hf.create_dataset('Magnitude_3G_10mm', data=Magnitude_3G_10mm)
    hf.create_dataset('Magnitude_pre', data=Magnitude_pre)
    hf.create_dataset('Labels_height', data=Labels_height)

    for key, data in Magnitudes_.items():
        hf.create_dataset(key, data=data)

# Visualization
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25
plt.figure()
plt.imshow(Magnitude_3G_10mm[0, 0, :, :],
           extent=(X_coor.min() * 1e3, X_coor.max() * 1e3, Y_coor.min() * 1e3, Y_coor.max() * 1e3),
           origin='upper',
           aspect='auto',
           cmap='jet',
           interpolation='bicubic')
plt.colorbar()
plt.title('10mm Height')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.2)
plt.tight_layout()

plt.figure()
plt.imshow(Magnitude_pre[0, 0, :, :],
           extent=(X_coor.min() * 1e3, X_coor.max() * 1e3, Y_coor.min() * 1e3, Y_coor.max() * 1e3),
           origin='upper',
           aspect='auto',
           cmap='jet',
           interpolation='bicubic')
plt.colorbar()
plt.title('Random Height')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.2)
plt.tight_layout()

toc = time.time()
print(f'Total time: {toc - tic:.2f} seconds')

plt.show()
