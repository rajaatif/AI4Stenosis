# app.py: Updated Gradio App with Segmentation Overlay + Stenosis Grading

import os
import torch
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import gdown  # For Drive downloads
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import segmentation_models_pytorch as smp

# Import grading module
from grading_model import load_grading_model, predict_grade, LABEL_TO_CLASS

# Existing classes (unchanged)
class IntensityNormalization(nn.Module):
    """
    Intensity normalization layer for X-ray angiography images.
    """
    def __init__(self, clip_percentile=1.0):
        super(IntensityNormalization, self).__init__()
        self.clip_percentile = clip_percentile

    def forward(self, x):
        x = x.contiguous()
        x_flat = x.reshape(x.size(0), -1)
        lower_bound = torch.quantile(x_flat, self.clip_percentile / 100.0, dim=1, keepdim=True)
        upper_bound = torch.quantile(x_flat, (100.0 - self.clip_percentile) / 100.0, dim=1, keepdim=True)
        x_clipped = torch.max(torch.min(x_flat, upper_bound), lower_bound)
        x_clipped = x_clipped.reshape(x.size())
        x_normalized = (x_clipped - lower_bound.reshape(x.size(0), 1, 1, 1)) / \
                       (upper_bound - lower_bound).reshape(x.size(0), 1, 1, 1)
        return x_normalized

class AngioUnet(nn.Module):
    def __init__(self, encoder_name="mit_b5", encoder_weights="imagenet", in_channels=3, classes=2,
                 attention_channels=32):
        super(AngioUnet, self).__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        self.intensity_norm = IntensityNormalization()
        # Attention (unused in forward)
        if hasattr(self.unet.decoder, 'blocks'):
            last_block = self.unet.decoder.blocks[-1]
            if hasattr(last_block, 'convs'):
                decoder_output_channels = last_block.convs[1][0].out_channels
            elif hasattr(last_block, 'conv1'):
                decoder_output_channels = last_block.conv1[0].out_channels
            else:
                raise ValueError("Decoder block structure not recognized.")
        elif hasattr(self.unet.decoder, 'convs'):
            decoder_output_channels = self.unet.decoder.convs[-1][0].out_channels
        else:
            raise ValueError("Decoder structure not recognized.")
        self.attention = nn.Conv2d(decoder_output_channels, attention_channels, kernel_size=1)

    def forward(self, x):
        x = self.intensity_norm(x)
        x = self.unet(x)
        return x

# Enhanced golden visualization (unchanged from last version)
def color_code_segmentation(mask, label_values, color='gold', smooth=True, edge_thickness=2):
    color_map = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'gold': [255, 255, 0]  # Pure bright yellow-gold
    }
    base_color = np.array(color_map.get(color, [255, 255, 0]), dtype=np.uint8)
    stenosis_mask = (mask == label_values[1]).astype(np.uint8)
    
    if smooth:
        kernel = np.ones((3, 3), np.uint8)
        stenosis_mask = cv2.morphologyEx(stenosis_mask, cv2.MORPH_CLOSE, kernel)
        stenosis_mask = cv2.GaussianBlur(stenosis_mask.astype(np.float32), (3, 3), 0)
        stenosis_mask = (stenosis_mask > 0.5).astype(np.uint8)
    
    edges = cv2.Canny(stenosis_mask * 255, 50, 150)
    edges = cv2.dilate(edges, np.ones((edge_thickness, edge_thickness), np.uint8), iterations=1)
    
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    colored_mask[stenosis_mask == 1] = base_color
    glow_color = [255, 255, 100]  # Bright yellow-gold glow
    colored_mask[edges > 0] = glow_color
    
    dist_transform = cv2.distanceTransform(stenosis_mask, cv2.DIST_L2, 5)
    gradient_factor = np.clip(1 - (dist_transform / np.max(dist_transform)), 0.7, 1)
    for i in range(3):
        colored_mask[:, :, i] = (colored_mask[:, :, i] * gradient_factor).astype(np.uint8)
    
    return colored_mask

def overlay_mask_on_image(image, mask, alpha=0.5, enhance_contrast=True):
    colored_mask = color_code_segmentation(mask, [0, 1], color='gold', smooth=True, edge_thickness=2)
    blend_mask = (mask == 1).astype(np.float32)
    rgb_overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    hsv_image = cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2HSV)
    hsv_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2HSV)
    
    hsv_image[:, :, 0] = hsv_image[:, :, 0] * (1 - alpha * 0.1 * blend_mask) + hsv_mask[:, :, 0] * (alpha * 0.1 * blend_mask)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * (1 - alpha * blend_mask) + hsv_mask[:, :, 1] * (alpha * 1.2 * blend_mask)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * (1 - alpha * blend_mask) + hsv_mask[:, :, 2] * (alpha * 1.1 * blend_mask), 0, 255)
    
    blended_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    if enhance_contrast:
        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        blended_image = cv2.LUT(blended_image, table)
        
        lab = cv2.cvtColor(blended_image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        blended_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    blurred_edges = cv2.GaussianBlur(colored_mask, (15, 15), 0)
    glow_layer = cv2.addWeighted(blended_image, 0.85, blurred_edges, 0.15, 0)
    
    return glow_layer.astype(np.uint8)

# Load segmentation model (existing)
def load_segmentation_model(model_path="model_mit_b5_256_best_dice.pth"):
    try:
        model = AngioUnet()
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint) #, strict=False
        print("Segmentation model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load segmentation model: {e}")

# Global models (loaded once)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model = None  # Segmentation
grade_model = None  # Grading

# Google Drive download function (unchanged)
def download_model_from_gdrive(gdrive_url, output_path):
    if '/file/d/' in gdrive_url:
        file_id = gdrive_url.split('/file/d/')[1].split('/')[0]
    else:
        raise ValueError("Invalid Google Drive URL.")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    print(f"Model downloaded to {output_path}")

# Load all models at startup
# Segmentation model path/URL (existing)
seg_model_path = "model_mit_b5_256_best_dice.pth"
seg_gdrive_url = "https://drive.google.com/file/d/1sytzRSEoSI6T2bKPOrl_iP-FoUmjcCju/view?usp=sharing"  # Your existing

if not os.path.exists(seg_model_path):
    print("Downloading segmentation model from Google Drive...")
    download_model_from_gdrive(seg_gdrive_url, seg_model_path)

print("Loading segmentation model...")
seg_model = load_segmentation_model(seg_model_path)
seg_model.to(device)
seg_model.eval()



# Grading model path/URL (NEW - replace with your actual classifier .pth URL)
grade_model_path = "unet_bottleneck_classifier_new.pth"
grade_gdrive_url = "https://drive.google.com/file/d/1R49zTOrS76ghkkcvgHflgW7Tm9S0nszW/view?usp=sharing" #"https://drive.google.com/file/d/1g-3Hg28glo6_UIO4sTdlh_FucclxXQSz/view?usp=sharing"  # UPDATE THIS! #

if not os.path.exists(grade_model_path):
    print("Downloading grading classifier from Google Drive...")
    download_model_from_gdrive(grade_gdrive_url, grade_model_path)

print("Loading grading model...")
grade_model = load_grading_model(grade_model_path)
grade_model.to(device)
grade_model.eval()

print(f"All models loaded on device: {device}")

# # Preprocessing transform (shared for both models)
# transform = A.Compose([
#     A.Resize(height=256, width=256), #, interpolation=cv2.INTER_LINEAR
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet for mit_b5
#     ToTensorV2(),
# ])
transform = A.Compose([
    A.Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ToTensorV2(), # Converts image HWC->CHW, scales to [0,1] THEN normalizes; converts mask HWC->CHW or HW->HW
])

# Unified predict: Segmentation overlay + Grading
def predict(image, alpha=0.5):
    global seg_model, grade_model
    if seg_model is None or grade_model is None:
        raise RuntimeError("Models not loaded. Check startup logs.")

    # Preprocess
    orig_image = np.array(image)
    if len(orig_image.shape) == 2:
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2RGB)
    elif orig_image.shape[2] == 4:
        orig_image = orig_image[:, :, :3]

    transformed = transform(image=orig_image)
    input_tensor = transformed["image"].unsqueeze(0)  # [1, 3, 256, 256]

    # Segmentation
    with torch.no_grad():
        seg_output = seg_model(input_tensor)
        seg_probs = torch.softmax(seg_output, dim=1)
        pred_mask = torch.argmax(seg_probs, dim=1).squeeze(0).cpu().numpy()

    if orig_image.shape[:2] != (256, 256):
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (orig_image.shape[1], orig_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlaid_image = overlay_mask_on_image(orig_image, pred_mask, alpha=alpha)

    # Grading (reuse input_tensor)
    grade_str = predict_grade(grade_model, input_tensor, device)

    # Outputs
    return Image.fromarray(overlaid_image), grade_str

# Gradio Interface (updated for Image + Text output)
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Input Angiography Image"),
        gr.Slider(0.1, 1.0, value=0.5, label="Overlay Alpha (Transparency)")
    ],
    outputs=[
        gr.Image(type="pil", label="Golden Segmentation Overlay"),
        gr.Textbox(label="Stenosis Grade Prediction", lines=2)
    ],
    title="Angiography Stenosis Segmentation & Grading",
    description="Upload an image for golden overlay segmentation + automatic stenosis grading (e.g., p70_90: 70-90%). Powered by MiT-B5 U-Net + Bottleneck Classifier."
)


if __name__ == "__main__":
    iface.launch()
