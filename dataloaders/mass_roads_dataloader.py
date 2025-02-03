import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from torch.utils.data.sampler import WeightedRandomSampler


class MassRoadsDataset(Dataset):
    def __init__(self, root_dir, split='train', window_size=128, stride=16, small_window_size=8, road_threshold=0.05, max_images=None, seed=42):
        """
        PyTorch Dataset for Mass Roads dataset. Extracts patches on-the-fly to prevent memory overload.
        """
        self.root_dir = root_dir
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.small_window_size = small_window_size
        self.road_threshold = road_threshold
        self.max_images = max_images
        self.seed = seed

        self.sat_dir = os.path.join(root_dir, split, 'sat')
        self.map_dir = os.path.join(root_dir, split, 'map')

        self.sat_files = sorted([f for f in os.listdir(self.sat_dir) if f.endswith('.tiff')])
        self.map_files = sorted([f for f in os.listdir(self.map_dir) if f.endswith('.tif')])

        assert len(self.sat_files) == len(self.map_files), "Mismatch between satellite and map files"

        if self.max_images:
            self.sat_files = self.sat_files[:self.max_images]
            self.map_files = self.map_files[:self.max_images]

        # Set random seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Precompute valid patches metadata
        self.patches_metadata = self._compute_patches_metadata()

    def _compute_patches_metadata(self):
        """Precompute valid patches per image without storing them in RAM."""
        patches_list = []

        for img_idx, (sat_file, map_file) in enumerate(zip(self.sat_files, self.map_files)):
            sat_path = os.path.join(self.sat_dir, sat_file)
            map_path = os.path.join(self.map_dir, map_file)

            with rasterio.open(sat_path) as sat_src:
                h, w = sat_src.shape  # Get image dimensions (H, W)

            # Compute valid patch positions
            for y in range(0, h - self.window_size + 1, self.stride):
                for x in range(0, w - self.window_size + 1, self.stride):
                    patches_list.append({
                        "image_idx": img_idx,
                        "y": y,
                        "x": x,
                    })  # Store patch metadata as a dictionary

        print(f"Total patches available: {len(patches_list)}")
        
        return patches_list

    def __len__(self):
        """Return the total number of patches in the dataset."""
        return len(self.patches_metadata)

    def __getitem__(self, idx):
        """Load only the required image and extract the corresponding patch dynamically."""
        patch_meta = self.patches_metadata[idx]
        image_idx, y, x = patch_meta["image_idx"], patch_meta["y"], patch_meta["x"]
        sat_filename = self.sat_files[image_idx] 
        map_filename = self.map_files[image_idx]


        sat_path = os.path.join(self.sat_dir, sat_filename)
        map_path = os.path.join(self.map_dir, map_filename)

        # Read only the required image on-demand
        with rasterio.open(sat_path) as sat_src:
            sat_image = sat_src.read().astype(np.float32) / 255.0  # Normalize

        with rasterio.open(map_path) as map_src:
            map_image = map_src.read(1).astype(np.uint8)
            map_image = (map_image > 127).astype(np.uint8)  # Convert to binary

        # Extract patch
        sat_patch = sat_image[:, y:y + self.window_size, x:x + self.window_size]
        map_patch = map_image[y:y + self.window_size, x:x + self.window_size]

        # Step 1: Apply smaller sliding window check
        if self._check_small_window(sat_patch):
            map_patch[:] = 0  # Remove roads if condition is met

        # Step 2: Calculate the percentage of road pixels
        road_percentage = np.sum(map_patch) / (self.window_size ** 2)

        # Step 3: If the patch does not contain enough roads, skip it
        if road_percentage < self.road_threshold:
            return None  # Skip this patch

        # Return patches + metadata
        return {
            "sat_patch": torch.tensor(sat_patch, dtype=torch.float32),
            "map_patch": torch.tensor(map_patch, dtype=torch.long),
            "metadata": patch_meta
        }

    def _check_small_window(self, sat_patch):
        """
        Checks if there are small fully white regions inside the patch.
        If the small window has all white pixels, return True.
        """
        small_h, small_w = self.small_window_size, self.small_window_size
        for y in range(0, sat_patch.shape[1] - small_h + 1, small_h):
            for x in range(0, sat_patch.shape[2] - small_w + 1, small_w):
                window = sat_patch[:, y:y + small_h, x:x + small_w]
                if np.all(window >= 0.99):  # Adjusted for float values
                    return True
        return False


def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Remove skipped patches

    if len(batch) == 0:
        return None, None, None  # Skip empty batches in DataLoader

    # Avoid memory spike by stacking only needed data
    sat_patches = torch.stack([item["sat_patch"] for item in batch], dim=0).contiguous()
    map_patches = torch.stack([item["map_patch"] for item in batch], dim=0).contiguous()
    metadata = [item["metadata"] for item in batch]

    return sat_patches, map_patches, metadata


def get_patch_sampler(dataset):
    """Creates a sampler to balance samples based on patch count per image."""
    patch_counts = np.zeros(len(dataset.sat_files))  # One per image

    # Count patches per image
    for patch in dataset.patches_metadata:
        patch_counts[patch["image_idx"]] += 1

    # Normalize to create sampling probabilities
    weights = 1.0 / (patch_counts + 1e-6)
    weights /= weights.sum()  # Normalize

    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return sampler
