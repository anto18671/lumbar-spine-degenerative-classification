# ============================
# Import Libraries - Suppress Warnings
# ============================
from albumentations import HorizontalFlip, Rotate, ElasticTransform, Compose, Normalize, GaussNoise
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from monai.networks.nets import resnet, SwinUNETR, SegResNet, DynUNet, ViT
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import LabelEncoder
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch import nn
import pandas as pd
import numpy as np
import warnings
import pydicom
import random
import torch
import cv2
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================
# Configuration Parameters
# ============================

# Configuration flags to enable or disable optional features
CONFIG = {
    'enable_spatial_scaling': True ,         # Feature 1: Spatial Scaling Using Pixel Spacing
    'model_architecture': 'ResNet3D',        # Feature 2: Model Architecture ('ResNet3D', 'SwinUNETR', 'SegResNet', 'DynUNet', 'ViT')
}

# ============================
# Utility Functions
# ============================

def seed_everything(seed=42):
    """
    Set the seed for reproducibility across random modules.
    This ensures that the same random sequences are generated each time.
    """
    # Set seed for random
    random.seed(seed)
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # Set seed for CUDA operations
    torch.cuda.manual_seed_all(seed)

def set_device():
    """Set the device for torch tensors."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================
# Loading Data Functions
# ============================

def load_data():
    """
    Load labels and image paths into a DataFrame.
    """
    labels_df = pd.read_csv('data/train.csv')
    coordinates_df = pd.read_csv('data/train_label_coordinates.csv')
    
    # Build image path for each row in coordinates_df
    def get_image_path(row):
        study_id = row['study_id']
        series_id = row['series_id']
        instance_number = row['instance_number']
        image_path = os.path.join('data/train_images', str(study_id), str(series_id), f"{instance_number}.dcm")
        return image_path
    
    # Apply get_image_path function to each row in coordinates_df
    coordinates_df['image_path'] = coordinates_df.apply(get_image_path, axis=1)
    
    # Merge with labels_df on 'study_id' to get severity labels
    df = pd.merge(coordinates_df, labels_df, on='study_id', how='left')

    # Combine severity labels for central, left, and right stenosis
    severity_columns = [
        'spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3',
        'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5',
        'spinal_canal_stenosis_l5_s1', 'left_neural_foraminal_narrowing_l1_l2',
        'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4',
        'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1',
        'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3',
        'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5',
        'right_neural_foraminal_narrowing_l5_s1'
    ]

    # Map severity labels to integers (adjust as needed)
    severity_mapping = {'Normal': 0, 'Normal/Mild': 0, 'Mild': 0, 'Moderate': 1, 'Severe': 2}
    
    # Map severity labels to integers
    for col in severity_columns:
        df[col] = df[col].map(severity_mapping)
    
    # Assign the maximum severity label for each study across all columns (multi-task setup)
    df['central'] = df[['spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3',
                        'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5',
                        'spinal_canal_stenosis_l5_s1']].max(axis=1)
    
    df['left'] = df[['left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3',
                     'left_neural_foraminal_narrowing_l3_l4', 'left_neural_foraminal_narrowing_l4_l5',
                     'left_neural_foraminal_narrowing_l5_s1']].max(axis=1)
    
    df['right'] = df[['right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3',
                      'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5',
                      'right_neural_foraminal_narrowing_l5_s1']].max(axis=1)

    # Group by study_id and series_id to collect all image paths per series
    grouped_df = df.groupby(['study_id', 'series_id']).agg({
        'image_path': list,
        'central': 'first',
        'left': 'first',
        'right': 'first'
    }).reset_index()

    # Return the grouped DataFrame
    return grouped_df

# ============================
# Preprocessing Functions
# ============================

def preprocess_dicom(dicom_path):
    """
    Preprocess a DICOM image:
    - Load the DICOM file.
    - Correct image orientation.
    - Invert pixel values if needed.
    - Clip pixel intensities.
    - Normalize pixel values.
    - Apply CLAHE.
    - Resize image with aspect ratio preservation.
    """
    # Load DICOM file
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array.astype(np.float32)

    # Correct image orientation
    image = correct_image_orientation(image, dicom)

    # Invert pixel values if Photometric Interpretation is MONOCHROME1
    if dicom.PhotometricInterpretation == 'MONOCHROME1':
        image = np.max(image) - image

    # Intensity clipping
    p1, p99 = np.percentile(image, (1, 99))
    image = np.clip(image, p1, p99)

    # Min-Max scaling
    image = (image - image.min()) / (image.max() - image.min())

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = (image * 255).astype(np.uint8)
    image = clahe.apply(image)

    # Apply spatial scaling using PixelSpacing if enabled
    if CONFIG['enable_spatial_scaling']:
        pixel_spacing = dicom.PixelSpacing
        image = rescale_image(image, pixel_spacing)

    # Resize image to 224x224 with aspect ratio preservation
    image = resize_with_aspect_ratio(image, target_size=224)

    # Return the preprocessed image and DICOM dataset
    return image, dicom

def correct_image_orientation(image, dicom_dataset):
    """
    Correct the orientation of the image based on DICOM metadata.
    """
    # Check if 'ImageOrientationPatient' field exists in the DICOM dataset
    if 'ImageOrientationPatient' in dicom_dataset:
        # Extract the orientation from the DICOM metadata
        orientation = dicom_dataset.ImageOrientationPatient
        
        # Separate the row and column direction cosines from the orientation
        row_cosines = np.array(orientation[:3])
        col_cosines = np.array(orientation[3:])
        
        # Calculate the normal vector of the image plane by taking the cross product of the row and column cosines
        normal = np.cross(row_cosines, col_cosines)

        # Check if the normal vector's Z-component is negative, which indicates the image needs to be flipped vertically
        if normal[2] < 0:
            # Flip the image vertically
            image = np.flipud(image)
    
    # Return the corrected image
    return image

def resize_with_aspect_ratio(image, target_size):
    """
    Resize image while preserving aspect ratio and pad with zeros if necessary.
    """
    # Get the height and width of the image
    height, width = image.shape
    
    # Calculate the aspect ratio of the image (height divided by width)
    aspect_ratio = height / width

    # If the aspect ratio is greater than 1, the image is taller than it is wide
    if aspect_ratio > 1:
        new_width = int(target_size / aspect_ratio)
        image = cv2.resize(image, (new_width, target_size), interpolation=cv2.INTER_CUBIC)
        padding = (target_size - new_width) // 2
        image = cv2.copyMakeBorder(
            image, 0, 0, padding, target_size - new_width - padding,
            cv2.BORDER_CONSTANT, value=0
        )
    else:
        new_height = int(target_size * aspect_ratio)
        image = cv2.resize(image, (target_size, new_height), interpolation=cv2.INTER_CUBIC)
        padding = (target_size - new_height) // 2
        image = cv2.copyMakeBorder(
            image, padding, target_size - new_height - padding, 0, 0,
            cv2.BORDER_CONSTANT, value=0
        )
    
    # Return the resized image
    return image

def rescale_image(image, pixel_spacing, target_spacing=(0.5, 0.5)):
    """
    Rescale image using pixel spacing to ensure consistent physical dimensions.
    """
    # Scale factors for rescaling the image
    scale_factors = [ps / ts for ps, ts in zip(pixel_spacing, target_spacing)]
    # New size after rescaling
    new_size = [int(dim * scale) for dim, scale in zip(image.shape, scale_factors)]
    # Rescale the image using cubic interpolation
    image_rescaled = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC)
    # Return the rescaled image
    return image_rescaled

# ============================
# Custom Dataset
# ============================

def get_transforms(phase):
    """
    Get data augmentation transforms based on the phase (train/valid).
    """
    # Define transforms for training and validation phases
    if phase == 'train':
        return Compose([
            # Rotate images
            Rotate(limit=10, p=0.5),
            # Flip images horizontally
            HorizontalFlip(p=0.5),
            # Apply elastic transformations
            ElasticTransform(alpha=34, sigma=4, alpha_affine=None, p=0.5),
            # Add Gaussian noise
            GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            # Normalize pixel values to the range
            Normalize(mean=(0.5,), std=(0.5,)),
            # Convert the image to PyTorch tensor
            ToTensorV2(),
        ])
    else:
        return Compose([
            # Normalize pixel values to the range
            Normalize(mean=(0.5,), std=(0.5,)),
            # Convert the image to PyTorch tensor
            ToTensorV2(),
        ])

class SpineDataset(Dataset):
    """
    PyTorch Dataset for spine MRI images with sequences of images.
    """
    def __init__(self, df, transforms=None, slice_thickness_mean=None, slice_thickness_std=None,
                 pixel_spacing_mean=None, pixel_spacing_std=None, condition=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.slice_thickness_mean = slice_thickness_mean
        self.slice_thickness_std = slice_thickness_std
        self.pixel_spacing_mean = pixel_spacing_mean
        self.pixel_spacing_std = pixel_spacing_std
        self.condition = condition

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index):
        """
        Load and preprocess the image and metadata for the given index.
        """
        record = self.df.iloc[index]
        image_paths = record['image_path']
        images = []
        dicom_datasets = []

        # Load and preprocess each image
        for image_path in image_paths:
            if not os.path.exists(image_path):
                continue
            image, dicom_dataset = preprocess_dicom(image_path)
            images.append(image)
            dicom_datasets.append(dicom_dataset)

        if not images:
            raise ValueError(f"No images found for index {index}")

        images = np.stack(images, axis=0)

        # Resize or pad the sequence to a fixed length
        desired_slices = 32
        actual_slices = images.shape[0]

        if actual_slices < desired_slices:
            padding = desired_slices - actual_slices
            pad_width = ((0, padding), (0, 0), (0, 0))
            images = np.pad(images, pad_width, mode='constant', constant_values=0)
        elif actual_slices > desired_slices:
            indices = np.linspace(0, actual_slices - 1, desired_slices).astype(int)
            images = images[indices]

        if self.transforms:
            transformed_slices = [self.transforms(image=img)['image'] for img in images]
            images = torch.stack(transformed_slices, dim=0)
        else:
            images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)

        images = images.permute(1, 0, 2, 3)

        # Extract metadata from the first DICOM file
        dicom = dicom_datasets[0]
        slice_thickness = float(dicom.SliceThickness)
        pixel_spacing = float(dicom.PixelSpacing[0])
        spacing_between_slices = float(dicom.SpacingBetweenSlices)
        patient_position = dicom.PatientPosition
        image_position = dicom.ImagePositionPatient
        image_orientation = dicom.ImageOrientationPatient
        slice_location = float(dicom.SliceLocation)

        # Standardize metadata features
        slice_thickness_norm = (slice_thickness - self.slice_thickness_mean) / self.slice_thickness_std
        pixel_spacing_norm = (pixel_spacing - self.pixel_spacing_mean) / self.pixel_spacing_std
        spacing_between_slices_norm = (spacing_between_slices - self.slice_thickness_mean) / self.slice_thickness_std
        image_position_norm = np.array(image_position) / 1000.0
        image_orientation_norm = np.array(image_orientation) / 1.0
        slice_location_norm = slice_location / 1000.0

        # Encode patient position (e.g., HFS, FFS, etc.)
        patient_position_encoded = 1 if patient_position == 'HFS' else 0

        # Create metadata features tensor
        metadata_features = np.array([
            slice_thickness_norm,
            pixel_spacing_norm,
            spacing_between_slices_norm,
            patient_position_encoded,
            *image_position_norm,
            *image_orientation_norm,
            slice_location_norm,
        ], dtype=np.float32)

        metadata_features = torch.tensor(metadata_features)

        # Update label extraction based on the correct column names
        label_central = torch.tensor(record['central']).long()
        label_left = torch.tensor(record['left']).long()
        label_right = torch.tensor(record['right']).long()

        # Return images, metadata, and a dictionary of labels
        return images, metadata_features, {'central': label_central, 'left': label_left, 'right': label_right}

# ============================
# Data Preparation
# ============================

def prepare_data(df, fold=0, condition=None):
    """
    Prepare training and validation data loaders with metadata statistics.
    """
    # Print the initial number of samples
    print(f"Initial number of samples: {len(df)}")
    
    # Filter the DataFrame to include only rows where image paths exist and are not empty
    df = df[df['image_path'].apply(lambda paths: paths and all(os.path.exists(p) for p in paths))]
    
    # Print the number of samples after filtering for existing image paths
    print(f"Number of samples after filtering for existing image paths: {len(df)}")
    
    # Check for NaN values in important columns (i.e., 'central', 'left', 'right')
    print("Checking for NaN values...")
    nan_columns = df.columns[df.isna().any()].tolist()
    print(f"Columns with NaN values: {nan_columns}")
    
    # Handle NaN values - Option 1: Drop rows with NaN values
    df = df.dropna(subset=['central', 'left', 'right'])
    
    # Print the number of samples after handling NaN values
    print(f"Number of samples after handling NaN values: {len(df)}")

    # Filter for condition-specific tailoring (if you want to split by condition)
    if condition:
        df = df[df['condition'] == condition]
        # Print the number of samples after filtering by condition
        print(f"Number of samples after filtering by condition '{condition}': {len(df)}")
    
    # Stratified Group K-Fold to split the data into training and validation sets while preserving label distribution
    sgkf = StratifiedGroupKFold(n_splits=10)

    # Create a combined label as a string for stratification
    df['combined_label'] = df.apply(lambda row: f"{row['central']}_{row['left']}_{row['right']}", axis=1)

    # Encode the combined label as an integer using LabelEncoder
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['combined_label'])

    labels = df['encoded_label']  # Use the encoded label for stratification
    groups = df['study_id']       # Use 'study_id' for group stratification
    
    # Iterate over each fold and split the data into training and validation based on the current fold
    for current_fold, (train_idx, valid_idx) in enumerate(sgkf.split(df, labels, groups)):
        if current_fold == fold:
            # Select the data for the current fold
            train_df = df.iloc[train_idx].reset_index(drop=True)
            valid_df = df.iloc[valid_idx].reset_index(drop=True)
            break
    
    # Compute metadata statistics (mean and standard deviation) from the training data
    slice_thickness_list = []
    pixel_spacing_list = []
    
    # Loop over each record in the training data to gather slice thickness and pixel spacing from DICOM files
    for idx in range(len(train_df)):
        record = train_df.iloc[idx]
        image_paths = record['image_path']
        if not image_paths:
            continue
        # Use the first image to extract metadata
        first_image_path = image_paths[0]
        if not os.path.exists(first_image_path):
            continue
        dicom = pydicom.dcmread(first_image_path, stop_before_pixels=True)
        # Append slice thickness and pixel spacing values to their respective lists
        slice_thickness_list.append(float(dicom.SliceThickness))
        pixel_spacing_list.append(float(dicom.PixelSpacing[0]))
    
    # Handle cases where lists might be empty
    if not slice_thickness_list or not pixel_spacing_list:
        raise ValueError("No valid DICOM metadata found in training data.")
    
    # Calculate the mean and standard deviation of slice thickness and pixel spacing for normalization
    slice_thickness_mean = np.mean(slice_thickness_list)
    slice_thickness_std = np.std(slice_thickness_list)
    pixel_spacing_mean = np.mean(pixel_spacing_list)
    pixel_spacing_std = np.std(pixel_spacing_list)
    
    # Create the training dataset with transforms and computed metadata statistics
    train_dataset = SpineDataset(
        train_df,
        transforms=get_transforms('train'),
        slice_thickness_mean=slice_thickness_mean,
        slice_thickness_std=slice_thickness_std,
        pixel_spacing_mean=pixel_spacing_mean,
        pixel_spacing_std=pixel_spacing_std,
        condition=condition
    )
    
    # Create the validation dataset with transforms and the same metadata statistics for consistency
    valid_dataset = SpineDataset(
        valid_df,
        transforms=get_transforms('valid'),
        slice_thickness_mean=slice_thickness_mean,
        slice_thickness_std=slice_thickness_std,
        pixel_spacing_mean=pixel_spacing_mean,
        pixel_spacing_std=pixel_spacing_std,
        condition=condition
    )
    
    # Create DataLoader for the training dataset
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    # Create DataLoader for the validation dataset
    valid_loader = DataLoader(
        valid_dataset, batch_size=4, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Return the data loaders for training and validation
    return train_loader, valid_loader

# ============================
# Multi-task Loss Function
# ============================
class MultiTaskLoss(nn.Module):
    def __init__(self, class_weights):
        """
        Initialize the MultiTaskLoss with class weights.
        """
        super(MultiTaskLoss, self).__init__()
        # Store the class weights for each task
        self.class_weights = class_weights
        # Initialize CrossEntropyLoss with the provided class weights
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, outputs, labels):
        """
        Compute multi-task loss for central canal, left foramina, and right foramina predictions.
        """
        # Calculate loss for each task with class weights
        loss_central = self.loss_fn(outputs[0], labels['central'])
        loss_left = self.loss_fn(outputs[1], labels['left'])
        loss_right = self.loss_fn(outputs[2], labels['right'])

        # Return the sum of the losses
        return loss_central + loss_left + loss_right
    
# ============================
# Model Selection Function
# ============================

def get_model(num_classes):
    """
    Return the multi-task model architecture based on the configuration.
    """
    if CONFIG['model_architecture'] == 'ResNet3D':
        # Use 3D ResNet model
        backbone = resnet.resnet18(
            spatial_dims=3,                     # 3D convolutions
            n_input_channels=1,                 # 1 input channel (grayscale images)
            num_classes=3,                      # 3 output classes for multi-task learning
            pretrained=True,                    # Load pretrained weights
            feed_forward=False,                 # Required for MedicalNet pretrained weights
            shortcut_type='A',                  # Required for MedicalNet weights
            bias_downsample=True,               # Set bias_downsample to True or False as required
        )
        backbone.fc = nn.Identity()
        return ResNetMultiTaskModel(backbone, num_classes)

    if CONFIG['model_architecture'] == 'SwinUNETR':
        # Use SwinUNETR model
        backbone = SwinUNETR(
            img_size=(224, 224, 32),            # Input image size (adapted for 3D images)
            in_channels=1,                      # Number of input channels (grayscale images)
            out_channels=3,                     # 3 output channels for multi-task learning
            feature_size=48,                    # Feature size (adjust as needed)
            use_checkpoint=True,                # Memory-efficient training
        )
        return MultiTaskModel(backbone, num_classes)

    if CONFIG['model_architecture'] == 'SegResNet':
        # Use SegResNet model
        backbone = SegResNet(
            spatial_dims=3,                     # 3D input for MRI slices
            in_channels=1,                      # Number of input channels (grayscale images)
            out_channels=3,                     # 3 output channels for multi-task learning
            init_filters=16,                    # Initial number of filters (adjustable)
            dropout_prob=0.2,                   # Dropout probability for regularization
        )
        return MultiTaskModel(backbone, num_classes)

    if CONFIG['model_architecture'] == 'DynUNet':
        # Use DynUNet model
        backbone = DynUNet(
            spatial_dims=3,                     # 3D input for spine MRI data
            in_channels=1,                      # Number of input channels (grayscale images)
            out_channels=3,                     # 3 output channels for multi-task learning
            kernel_size=[3, 3, 3, 3],           # Kernel sizes for each level in the network
            strides=[1, 2, 2, 2],               # Strides for each level (adjust as needed)
            upsample_kernel_size=[2, 2, 2],     # Kernel sizes for upsampling layers
        )
        return MultiTaskModel(backbone, num_classes)

    if CONFIG['model_architecture'] == 'ViT':
        # Use Vision Transformer (ViT) model
        backbone = ViT(
            in_channels=1,                      # Number of input channels (grayscale images)
            img_size=(224, 224, 32),            # Input image size (adapted for 3D images)
            patch_size=(16, 16, 16),            # Patch size for dividing input into patches
            hidden_size=768,                    # Hidden size for the transformer layers
            num_classes=3,                      # 3 output channels for multi-task learning
            num_heads=12,                       # Number of attention heads
            mlp_dim=3072,                       # MLP layer dimension
            dropout_rate=0.1,                   # Dropout rate for regularization
        )
        return MultiTaskModel(backbone, num_classes)
        
# ============================
# Model Architectures
# ============================

class ResNetMultiTaskModel(nn.Module):
    """
    Generalized multi-task model wrapper specifically adapted for ResNet3D.
    """
    def __init__(self, backbone, num_classes, metadata_size=14):
        super(ResNetMultiTaskModel, self).__init__()
        self.backbone = backbone
        
        # Linear layer to project metadata to the same feature size as the image features
        self.metadata_fc = nn.Linear(metadata_size, 64)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Adjust the number of input features for fully connected layers to match the backbone output
        self.fc_central = nn.Linear(512 + 64, num_classes)
        self.fc_left = nn.Linear(512 + 64, num_classes)
        self.fc_right = nn.Linear(512 + 64, num_classes)

    def forward(self, x, metadata=None):
        # Pass the images through the backbone
        features = self.backbone(x)
        
        # Since the output is already pooled, no further pooling is needed
        pooled_features = features

        # If metadata is provided, process it
        if metadata is not None:
            metadata = self.metadata_fc(metadata)
            # Concatenate pooled features and metadata
            pooled_features = torch.cat([pooled_features, metadata], dim=1)

        # Regularization with dropout
        pooled_features = self.dropout(pooled_features)
        
        # Multi-task outputs
        out_central = self.fc_central(pooled_features)
        out_left = self.fc_left(pooled_features)
        out_right = self.fc_right(pooled_features)

        return out_central, out_left, out_right

class MultiTaskModel(nn.Module):
    """
    Generalized multi-task model wrapper.
    """
    def __init__(self, backbone, num_classes, metadata_size=14):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        
        # Linear layer to project metadata to the same feature size as the image features
        self.metadata_fc = nn.Linear(metadata_size, 3)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Adjust the number of input features for fully connected layers to match the backbone output
        self.fc_central = nn.Linear(3, num_classes)
        self.fc_left = nn.Linear(3, num_classes)
        self.fc_right = nn.Linear(3, num_classes)

    def forward(self, x, metadata=None):
        # Pass the images through the backbone
        features = self.backbone(x)
        
        # If metadata is provided, project it to match the feature dimensions
        if metadata is not None:
            metadata = self.metadata_fc(metadata)
            
            # Expand metadata to match the feature dimensions (B, C, D, H, W)
            metadata = metadata.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            metadata = metadata.expand(-1, -1, features.size(2), features.size(3), features.size(4))
            
            # Combine features and metadata
            features = features + metadata
        
        # Global average pooling over spatial dimensions
        pooled_features = features.mean(dim=[2, 3, 4])  # (B, C)
        
        # Regularization with dropout
        pooled_features = self.dropout(pooled_features)
        
        # Multi-task outputs
        out_central = self.fc_central(pooled_features)
        out_left = self.fc_left(pooled_features)
        out_right = self.fc_right(pooled_features)

        return out_central, out_left, out_right

# ============================
# Training and Validation Functions
# ============================

def train_one_epoch(model, optimizer, scheduler, dataloader, device, loss_fn, scaler):
    """
    Train the model for one epoch.
    """
    # Set the model to training mode, initialize running loss and total samples
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    # Wrap the dataloader with tqdm
    progress_bar = tqdm(dataloader, desc='Training')

    # Iterate over the training dataloader
    for images, metadata_features, labels in progress_bar:
        # Move data to the appropriate device
        images = images.to(device)
        metadata_features = metadata_features.to(device)
        labels = {key: val.to(device) for key, val in labels.items()}

        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass through the model
        with autocast():
            outputs = model(images, metadata_features)
            loss = loss_fn(outputs, labels)

        # Perform backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Update running loss and total samples
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar with average loss
        avg_loss = running_loss / total_samples
        progress_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})

    # Compute average loss over the training set
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Return the loss
    return epoch_loss

def validate_one_epoch(model, dataloader, device, loss_fn):
    """
    Validate the model for one epoch.
    """
    # Set the model to evaluation mode, initialize running loss and total samples
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    # Initialize lists to store predictions, labels, and probabilities
    all_preds = []
    all_labels = []
    all_probs = []

    # Wrap the dataloader with tqdm
    progress_bar = tqdm(dataloader, desc='Validating')

    # Set torch.no_grad() and autocast() for evaluation
    with torch.no_grad(), autocast():
        # Iterate over the validation dataloader
        for images, metadata_features, labels in progress_bar:
            # Move data to the appropriate device
            images = images.to(device)
            metadata_features = metadata_features.to(device)
            labels = {key: val.to(device) for key, val in labels.items()}

            # Forward pass through the model
            outputs = model(images, metadata_features)
            loss = loss_fn(outputs, labels)

            # Update running loss and total samples
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar with average loss
            avg_loss = running_loss / total_samples
            progress_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})

            # Compute probabilities and predictions
            probs_central = nn.functional.softmax(outputs[0], dim=1)
            probs_left = nn.functional.softmax(outputs[1], dim=1)
            probs_right = nn.functional.softmax(outputs[2], dim=1)

            preds_central = torch.argmax(probs_central, dim=1)
            preds_left = torch.argmax(probs_left, dim=1)
            preds_right = torch.argmax(probs_right, dim=1)

            # Collect results for metrics
            all_probs.extend([
                probs_central.cpu().numpy(),
                probs_left.cpu().numpy(),
                probs_right.cpu().numpy()
            ])
            all_preds.extend([
                preds_central.cpu().numpy(),
                preds_left.cpu().numpy(),
                preds_right.cpu().numpy()
            ])

    # Compute average loss over the validation set
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Return the loss, predictions, labels, and probabilities
    return epoch_loss, all_preds, all_labels, all_probs

# ============================
# Training Loop with Progressive Unfreezing
# ============================

def train_model(model, train_loader, valid_loader, device, num_epochs=20):
    """
    Train the model and perform validation with progressive unfreezing (optional).
    """
    # Define class weights to handle imbalanced data, moved to the appropriate device
    class_weights = torch.tensor([1.0, 2.5, 5.0], dtype=torch.float).to(device)
    
    # Define the loss function with the specified class weights
    loss_fn = MultiTaskLoss(class_weights)

    # Initialize the optimizer with weight decay to prevent overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-4)

    # Exponential learning rate scheduler with a decay factor
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Initialize a gradient scaler for mixed precision training
    scaler = GradScaler()

    # Variables to track best validation loss and early stopping
    best_val_loss = np.inf
    epochs_no_improve = 0
    
    # Number of epochs with no improvement to trigger early stopping
    n_epochs_stop = 10

    # Main training loop
    for epoch in range(num_epochs):
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{num_epochs}, LR: {lr:.8f}")

        # Training for one epoch
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader,
            device, loss_fn, scaler
        )

        # Validation after training for one epoch
        val_loss, _, _, _ = validate_one_epoch(
            model, valid_loader, device, loss_fn
        )

        # Print training and validation loss
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model if the validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model's state
            torch.save(model.state_dict(), 'best_model.pth')
            # Reset early stopping counter
            epochs_no_improve = 0
        else:
            # Increment early stopping counter
            epochs_no_improve += 1

        # Early stopping if validation loss does not improve for n_epochs_stop consecutive epochs
        if epochs_no_improve >= n_epochs_stop:
            print("Early stopping")
            break

    # Return the trained model
    return model

# ============================
# Evaluation Functions
# ============================

def evaluate_model(model, valid_loader, device):
    """
    Evaluate the model on the validation set.
    """
    # Define class weights to handle imbalanced data, moved to the appropriate device
    class_weights = torch.tensor([1.0, 2.5, 5.0], dtype=torch.float).to(device)
    
    # Use Weighted Cross-Entropy Loss
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Get validation predictions
    val_loss, val_preds, val_labels, val_probs = validate_one_epoch(
        model, valid_loader, device, loss_fn
    )

    # Classification report
    target_names = ['Normal/Mild', 'Moderate', 'Severe']
    print("Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    print("Confusion Matrix:")
    print(cm)

    # AUC-ROC per class
    val_labels_one_hot = np.eye(3)[val_labels]
    auc = roc_auc_score(
        val_labels_one_hot, val_probs,
        average='macro', multi_class='ovr'
    )
    
    # Print AUC-ROC
    print(f"AUC-ROC: {auc:.4f}")

    # Return validation loss, predictions, true labels, and probabilities
    return val_loss, val_preds, val_labels, val_probs

# ============================
# Main Execution
# ============================

def main():
    # Set device
    device = set_device()
    
    # Set seed for reproducibility
    seed_everything()
    
    # Load and prepare data
    df = load_data()
    
    # Specify condition if needed (e.g., 'stenosis' or 'herniation')
    condition = None
    
    # Prepare data loaders
    train_loader, valid_loader = prepare_data(df, condition=condition)
    
    # Initialize model based on the configuration
    model = get_model(num_classes=3).to(device)
    
    # Train model
    trained_model = train_model(model, train_loader, valid_loader, device)
    
    # Load the best model
    trained_model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate model
    evaluate_model(trained_model, valid_loader, device)
    
    # Cleanup
    torch.cuda.empty_cache()

# Execute the main function
if __name__ == '__main__':
    main()
