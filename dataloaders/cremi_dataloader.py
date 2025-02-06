import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import ToTensor
from scipy.ndimage import distance_transform_edt as dist
import matplotlib.pyplot as plt
import matplotlib

# Ensure Matplotlib uses a compatible backend
matplotlib.use('Agg')  # Use a non-GUI backend


def _crop_3d(volume_shape, shape=(16, 128, 128), method='random'):
    """Determine 3D cropping coordinates."""
    for dim, crop_dim in zip(volume_shape, shape):
        assert dim >= crop_dim, f"Image ({dim}) smaller than crop ({crop_dim})!"

    if method == 'random':
        crop_start = [np.random.randint(0, dim - crop_dim + 1) for dim, crop_dim in zip(volume_shape, shape)]
    elif method == 'center':
        crop_start = [(dim - crop_dim) // 2 for dim, crop_dim in zip(volume_shape, shape)]
    elif method == 'upperleft':
        crop_start = [0, 0, 0]
    else:
        raise ValueError(f"Unknown crop method {method}")

    return tuple(slice(start, start + crop_dim) for start, crop_dim in zip(crop_start, shape))

def crop_3d(volume, shape=(16, 128, 128), method='random'):
    """Crops a 3D volume to the specified shape."""
    crop_slices = _crop_3d(volume.shape, shape, method)
    return volume[crop_slices]

class Cremi3DDataset(Dataset):
    def __init__(self, root_dir="./CREMI", split="train", padded=True, use_clefts=False, dist_thresh=20, crop_size=(16, 128, 128), crop_method="random"):
        """
        Initializes the 3D CREMI dataset.

        Args:
            root_dir (str): Path to the CREMI dataset directory.
            split (str): One of 'train', 'val', or 'test'.
            padded (bool): Whether to use padded versions.
            use_clefts (bool): If True, use synaptic cleft segmentation (`clefts`) instead of neuron segmentation (`neuron_ids`).
            dist_thresh (int): Threshold for distance transform.
            crop_size (tuple): Desired crop size as (Depth, Height, Width).
            crop_method (str): Cropping method - 'random', 'center', or 'upperleft'.
        """
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.padded = padded
        self.use_clefts = use_clefts
        self.dist_thresh = dist_thresh
        self.crop_size = crop_size
        self.crop_method = crop_method

        # Assign dataset samples based on split
        if split == "train":
            samples = ["sample_A_20160501.hdf", "sample_B_20160501.hdf"]
        elif split == "val":
            samples = ["sample_C_20160501.hdf"]
        elif split == "test":
            samples = ["sample_A+_20160601.hdf", "sample_B+_20160601.hdf", "sample_C+_20160601.hdf"]
        else:
            raise ValueError(f"Invalid split: {split}. Choose from 'train', 'val', or 'test'.")

        # Correct padded filenames (adjust _padded_ placement)
        if padded:
            samples = [s.replace(".hdf", "").replace("_2016", "_padded_2016") + ".hdf" for s in samples]

        # Validate existence of files
        self.data_files = [os.path.join(root_dir, s) for s in samples if os.path.exists(os.path.join(root_dir, s))]

        if not self.data_files:
            print(f"❌ Error: No valid dataset files found in {root_dir}.")
            print("✅ Available files:", os.listdir(root_dir))
            raise FileNotFoundError("Dataset files missing!")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_path = self.data_files[index]

        # Load 3D volume from HDF5
        with h5py.File(file_path, "r") as f:
            raw = f["volumes/raw"][:]  # Shape: (Z, H, W)
            labels = f["volumes/labels/clefts"][:] if self.use_clefts else f["volumes/labels/neuron_ids"][:]

        # Normalize raw image intensity
        raw = raw.astype(np.float32) / 255.0  # Assuming 8-bit grayscale images

        # Ensure labels are binary
        labels = (labels > 0).astype(np.uint8)

        # Compute 3D distance transform
        distance_map = dist(labels)
        distance_map[distance_map > self.dist_thresh] = self.dist_thresh
        normalized_distance_map = distance_map / self.dist_thresh  # Normalize to [0,1]

        # Apply 3D cropping separately
        raw = crop_3d(raw, shape=self.crop_size, method=self.crop_method)
        labels = crop_3d(labels, shape=self.crop_size, method=self.crop_method)
        normalized_distance_map = crop_3d(normalized_distance_map, shape=self.crop_size, method=self.crop_method)

        assert raw.shape == labels.shape == normalized_distance_map.shape, "Mismatch in cropped shapes!"

        # Convert to tensors
        return (
            torch.tensor(raw, dtype=torch.float32).unsqueeze(0),
            torch.tensor(labels, dtype=torch.float32).unsqueeze(0),
            torch.tensor(normalized_distance_map, dtype=torch.float32).unsqueeze(0),
        )

def load_cremi_3d_dataset(root_dir="./CREMI", split="train", padded=False, use_clefts=False, dist_thresh=20, crop_size=(16, 128, 128), batch_size=2, shuffle=True):
    """Creates a DataLoader for the 3D CREMI dataset."""
    dataset = Cremi3DDataset(root_dir=root_dir, split=split, padded=padded, use_clefts=use_clefts, dist_thresh=dist_thresh, crop_size=crop_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_volume(volume, title="3D Volume"):
    """Plots a 3D volume using matplotlib voxels."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert volume to binary for visualization (adjust threshold if needed)
    volume_binary = (volume > 0).astype(np.uint8)

    # Create a 3D voxel plot
    ax.voxels(volume_binary, facecolors='blue', edgecolors='k', alpha=0.7)

    ax.set_xlabel("Width (X)")
    ax.set_ylabel("Height (Y)")
    ax.set_zlabel("Depth (Z)")
    ax.set_title(title)

    plt.show()



if __name__ == "__main__":
    train_loader = load_cremi_3d_dataset(root_dir="/home/ri/Desktop/Projects/Datasets/CREMI/", split="train", crop_size=(4, 32, 32), batch_size=2)
    
    print("Dataset successfully loaded!\n")

    for images, labels, dist_maps in train_loader:
        print(f"Batch Shape: {images.shape}")  # Expected: (batch_size, 1, D, H, W)
        print(f"Raw Image Min/Max: {images.min()} / {images.max()}")
        print(f"Label Min/Max: {labels.min()} / {labels.max()}")
        print(f"Distance Map Min/Max: {dist_maps.min()} / {dist_maps.max()}")
        break
    # Get a sample batch from the dataset
    for images, labels, dist_maps in train_loader:
        sample_image = images[0, 0].numpy()  # Shape: (D, H, W)
        sample_label = labels[0, 0].numpy()  # Shape: (D, H, W)
        break  # Take only one sample

    # Plot 3D raw image and segmentation mask
    plot_3d_volume(sample_image, title="3D Raw Image")
    # plot_3d_volume(sample_label, title="3D Segmentation Label")

