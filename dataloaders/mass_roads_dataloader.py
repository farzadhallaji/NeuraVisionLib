import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import torch.nn.functional as F


class MassRoadsDataset(Dataset):
    def __init__(self, root_dir, split='train', window_size=128, stride=16, small_window_size=8, road_threshold=0.05, max_images=None):
        self.root_dir = root_dir
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.small_window_size = small_window_size
        self.road_threshold = road_threshold  # Threshold percentage of road pixels to keep a patch
        self.max_images = max_images  # Maximum number of images to process

        self.sat_dir = os.path.join(root_dir, split, 'sat')
        self.map_dir = os.path.join(root_dir, split, 'map')

        self.sat_files = sorted([f for f in os.listdir(self.sat_dir) if f.endswith('.tiff')])
        self.map_files = sorted([f for f in os.listdir(self.map_dir) if f.endswith('.tif')])

        assert len(self.sat_files) == len(self.map_files), "Mismatch between satellite and map files"

        # If max_images is provided, limit the dataset to that number of images
        if self.max_images:
            self.sat_files = self.sat_files[:self.max_images]
            self.map_files = self.map_files[:self.max_images]

    def __len__(self):
        return len(self.sat_files)

    def __getitem__(self, idx):
        sat_path = os.path.join(self.sat_dir, self.sat_files[idx])
        map_path = os.path.join(self.map_dir, self.map_files[idx])

        with rasterio.open(sat_path) as sat_src:
            sat_image = sat_src.read(out_dtype=np.float32)

        with rasterio.open(map_path) as map_src:
            map_image = map_src.read(1, out_dtype=np.uint8)

        sat_image /= 255.0  # Normalize the satellite image

        map_image = (map_image > 127).astype(np.float32)  # Convert map to binary (threshold at 127)

        if len(sat_image.shape) == 2:
            sat_image = np.expand_dims(sat_image, axis=0)

        sat_patches, map_patches = self._create_patches(sat_image, map_image)

        # Return the patches and the number of patches
        return sat_patches, map_patches, len(sat_patches)


    def _create_patches(self, sat_image, map_image):
        patches_sat = []
        patches_map = []

        h, w = sat_image.shape[1:]
        for y in range(0, h - self.window_size + 1, self.stride):
            for x in range(0, w - self.window_size + 1, self.stride):
                sat_patch = sat_image[:, y:y + self.window_size, x:x + self.window_size]
                map_patch = map_image[y:y + self.window_size, x:x + self.window_size]

                # Step 1: Apply smaller sliding window (small_window_size) to check for white pixels in the satellite patch
                if self._check_small_window(sat_patch):
                    # Step 2: Set the corresponding map patch to black
                    map_patch[:] = 0

                # Step 3: Calculate the percentage of white pixels in the map patch
                road_percentage = np.sum(map_patch) / (self.window_size ** 2)

                # Step 4: If road_percentage is less than the threshold, skip this patch
                if road_percentage >= self.road_threshold:
                    patches_sat.append(torch.tensor(sat_patch, dtype=torch.float32))
                    patches_map.append(torch.tensor(map_patch, dtype=torch.long))

        return patches_sat, patches_map

    def _check_small_window(self, sat_patch):
        small_h, small_w = self.small_window_size, self.small_window_size
        for y in range(0, sat_patch.shape[1] - small_h + 1, small_h):
            for x in range(0, sat_patch.shape[2] - small_w + 1, small_w):
                window = sat_patch[:, y:y + small_h, x:x + small_w]
                if np.all(window == 1):  # Check if the window is all white (or nearly white)
                    return True
        return False


def custom_collate_fn(batch):
    sat_patches, map_patches, _ = zip(*batch)
    sat_patches = [p for p in sat_patches if len(p) > 0]
    map_patches = [p for p in map_patches if len(p) > 0]

    if len(sat_patches) == 0 or len(map_patches) == 0:
        return torch.empty(0), torch.empty(0)

    sat_patches = torch.cat([torch.stack(p) for p in sat_patches], dim=0)
    map_patches = torch.cat([torch.stack(p) for p in map_patches], dim=0)

    return sat_patches, map_patches

