### **Dataset Configuration**
- **`name`**: Identifies the dataset (e.g., "MassRoads").
- **`type`**: Specifies the dataset type (e.g., "image").
- **`path`**: Points to the dataset's directory.
- **`split`**: Configures how the dataset is divided (e.g., subdirectory names or ratios).
- **`file_format`**: Specifies input and target file formats (e.g., `.tiff` for inputs and `.tif` for targets).
- **`custom_load_sample`**: Optionally defines a custom function for loading samples.

### **Preprocessing Configuration**
- **Global Toggles**:
  - `enabled`: Controls whether preprocessing is applied.
  - `use_seed`: Enables deterministic preprocessing using a seed (`seed: 42`).
- **Operations**:
  - **Resize**: Configurable for inputs, targets, or both; sets width and height.
  - **Normalization**: Standardizes inputs using `zscore` or `minmax`.
  - **Binarization**: Converts target masks to binary with a specified threshold.
  - **Sliding Window**: Extracts patches with sliding windows; includes fine-grained checks.
  - **Augmentation**: Adds variability with methods like random flips, rotations, and color jittering.
- **Custom Preprocessing**:
  - Allows user-defined operations for inputs, targets, or both, with specific parameters.

### **Loader Configuration**
- Configures data loading:
  - **Batching**: `batch_size: 16`.
  - **Parallelism**: Uses 4 workers (`num_workers: 4`).
  - **Shuffling**: Enabled for each epoch.
  - **Reproducibility**: Uses seeds for deterministic behavior (`seed: 42`).

### **Logging Configuration**
- Enables logging, specifies the log directory, and sets verbosity levels (`info`, `debug`, etc.).

### **Warnings**
- Enables alerts for missing files or mismatched dimensions.

