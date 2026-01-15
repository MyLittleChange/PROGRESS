import numpy as np

# Load the .npy file
data = np.load('/home/mila/c/chandhos/COINCIDE_code/COINCIDE_train/playground/data/TinyLLaVA-Instruction_recover_indices.npy')

# Print the array shape and contents
print("Array shape:", data.shape)
print("Array contents:")
print(data)