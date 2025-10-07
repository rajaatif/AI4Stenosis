# grading_model.py: Stenosis Grading Model (U-Net mit_b5 + Bottleneck Classifier)

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import gdown  # For Drive downloads (if seg weights need downloading here)
import os
# AngioUnet class (copied from original for self-containment)
class IntensityNormalization(nn.Module):
    """
    Intensity normalization layer for X-ray angiography images.  Normalizes
    the intensity values to a more consistent range, handling potential
    variations in contrast and brightness.
    """
    def __init__(self, clip_percentile=1.0):
        super(IntensityNormalization, self).__init__()
        self.clip_percentile = clip_percentile

    def forward(self, x):
        # Ensure tensor is contiguous for safety (prevents view/reshape issues)
        x = x.contiguous()
        
        # Flatten the image to calculate percentiles across the batch and spatial dimensions
        x_flat = x.reshape(x.size(0), -1)  # Use .reshape() instead of .view() for non-contiguous tensors

        # Calculate the lower and upper percentile values.
        lower_bound = torch.quantile(x_flat, self.clip_percentile / 100.0, dim=1, keepdim=True)
        upper_bound = torch.quantile(x_flat, (100.0 - self.clip_percentile) / 100.0, dim=1, keepdim=True)

        # Clip the intensities to the calculated bounds
        x_clipped = torch.max(torch.min(x_flat, upper_bound), lower_bound)

        # Reshape back to the original image shape
        x_clipped = x_clipped.reshape(x.size())  # Use .reshape() instead of .view()
        
        # Normalize the clipped intensities to the range [0, 1]
        x_normalized = (x_clipped - lower_bound.reshape(x.size(0), 1, 1, 1)) / \
                       (upper_bound - lower_bound).reshape(x.size(0), 1, 1, 1)  # Use .reshape() instead of .view()
        return x_normalized

class AngioUnet(nn.Module):
    def __init__(self, encoder_name="mit_b5", encoder_weights="imagenet", in_channels=3, classes=2,
                 attention_channels=32):  # Added attention_channels
        super(AngioUnet, self).__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        # 1. Intensity Normalization Layer (Physics-Informed)
        self.intensity_norm = IntensityNormalization()  # Apply normalization

        # 2. Attention Mechanism (Contextual Information)
        # Adapt attention to the output of the *last* decoder stage
        if hasattr(self.unet.decoder, 'blocks'):
            last_block = self.unet.decoder.blocks[-1]
            if hasattr(last_block, 'convs'):
                decoder_output_channels = last_block.convs[1][0].out_channels  # Access conv layer output channels
            elif hasattr(last_block, 'conv1'):
                decoder_output_channels = last_block.conv1[0].out_channels  # Access conv layer output channels
            else:
                raise ValueError("Decoder block structure not recognized.")
        elif hasattr(self.unet.decoder, 'convs'):
            decoder_output_channels = self.unet.decoder.convs[-1][0].out_channels  # Access conv layer output channels
        else:
            raise ValueError("Decoder structure not recognized.")
        self.attention = nn.Conv2d(decoder_output_channels, attention_channels, kernel_size=1)  # simple convolution
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Apply Intensity Normalization
        x = self.intensity_norm(x)
        # Pass through the base Unet
        x = self.unet(x)
        return x

# UnetBottleneckClassifier class (exactly as provided in your query/notebook)
class UnetBottleneckClassifier(nn.Module):
    def __init__(self, num_classes, encoder_name="mit_b5", encoder_weights="imagenet", freeze_layers=True):
        super(UnetBottleneckClassifier, self).__init__()

        # Load pre-trained U-Net with mit_b5 encoder
        self.unet = AngioUnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=2,
        )
        # Note: Loading seg weights will be handled externally in load_grading_model
        # (to avoid re-download; pass pre-loaded unet if needed)

        # Optionally freeze some encoder layers
        if freeze_layers:
            for name, param in self.unet.unet.encoder.named_parameters():
                # Freeze the early layers (you can fine-tune this range)
                if any(f'stage{i}' in name for i in [0, 1, 2]):  # Freeze stages 0, 1, 2
                    param.requires_grad = False

        # Get the number of features from the bottleneck layer
        bottleneck_features = self.unet.unet.encoder.out_channels[-1]

        # Create a classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_features, num_classes),
        )

    def forward(self, x):
        # Pass the input through the U-Net encoder
        x = self.unet.unet.encoder(x)[-1]

        # Pass the bottleneck output to the classifier
        x = self.classifier(x).float()

        return x

# Class mappings from your notebook (adjusted to 5 classes as per your mapping)
LABEL_TO_CLASS = {
    0: 'p0_20 (0-20%)',
    1: 'p20_50 (20-50%)',
    2: 'p50_70 (50-70%)',
    3: 'p70_90 (70-90%)',  # Includes p90_98, p99
    4: 'p100 (100%)'       # p100
}

# Google Drive download function (for seg weights if needed here)
def download_model_from_gdrive(gdrive_url, output_path):
    if '/file/d/' in gdrive_url:
        file_id = gdrive_url.split('/file/d/')[1].split('/')[0]
    else:
        raise ValueError("Invalid Google Drive URL.")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    print(f"Model downloaded to {output_path}")

def load_grading_model(seg_model_path="model_mit_b5_256_best_dice.pth", classifier_model_path="unet_bottleneck_classifier_new.pth", num_classes=5):
    """
    Load the full grading model:
    1. Download/load seg weights (for unet).
    2. Create UnetBottleneckClassifier (which uses AngioUnet).
    3. Load seg weights into the unet's state_dict (as per your notebook).
    4. Load the full saved classifier .pth on top (strict=False for mismatches).
    
    Args:
        seg_model_path: Path to segmentation weights (.pth).
        classifier_model_path: Path to saved full classifier .pth.
        num_classes: Number of output classes (5 per mapping).
    
    Returns:
        model: Loaded UnetBottleneckClassifier.
    """
    try:
        # Step 1: Ensure seg weights are available (download if missing)
        if not os.path.exists(seg_model_path):
            # Assume seg_gdrive_url is passed or hardcoded; update as needed
            seg_gdrive_url = "https://drive.google.com/file/d/1sytzRSEoSI6T2bKPOrl_iP-FoUmjcCju/view?usp=sharing"  # Your seg URL
            print("Downloading seg weights for grading model...")
            download_model_from_gdrive(seg_gdrive_url, seg_model_path)

        # Step 2: Create the classifier (AngioUnet is created internally)
        model = UnetBottleneckClassifier(num_classes=num_classes)

        # Step 3: Load seg weights into the unet (as per your notebook's init)
        seg_checkpoint = torch.load(seg_model_path, map_location="cpu")
        model.unet.load_state_dict(seg_checkpoint, strict=False)  # strict=False for any mismatches
        print("Seg weights loaded into grading model's unet.")

        # Step 4: Load the full saved classifier state_dict (includes classifier head)
        classifier_checkpoint = torch.load(classifier_model_path, map_location="cpu")
        model.load_state_dict(classifier_checkpoint, strict=False)  # Ignores unet mismatches (already loaded)
        print("Full classifier state_dict loaded successfully (with strict=False).")

        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load grading model: {e}")

def predict_grade(model, image_tensor, device, class_mapping=LABEL_TO_CLASS):
    """
    Predict stenosis grade from preprocessed image tensor.
    
    Args:
        model: Loaded UnetBottleneckClassifier.
        image_tensor: [1, 3, 256, 256] normalized tensor.
        device: torch.device.
        class_mapping: Dict for label to readable grade.
    
    Returns:
        str: Formatted grade (e.g., "Stenosis Grade: p70_90 (70-90%)").
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    grade_str = f"Stenosis Grade: {class_mapping[pred_class]} (Confidence: {confidence:.2%})"
    return grade_str