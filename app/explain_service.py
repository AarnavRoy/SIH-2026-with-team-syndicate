import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def generate_vit_saliency_heatmap(model: torch.nn.Module, image_tensor: torch.Tensor, original_size: tuple) -> str:
    """
    Computes visual saliency:
    - If model is ViT (has encoder): tracks attention across patch tokens.
    - If model is EfficientNet (has features): uses Grad-CAM on final convolutional layer.
    Returns: Base64-encoded PNG string of the colorized heatmap overlay.
    """
    model.eval()
    
    # Check if EfficientNet architecture
    if hasattr(model, 'features'):
        return _generate_efficientnet_gradcam(model, image_tensor, original_size)
    
    # ViT Architecture
    activations = []
    def hook_fn(module, input, output):
        activations.append(output)
    
    # Hook the last transformer layer
    target_layer = model.encoder.layers[-1]
    hook = target_layer.register_forward_hook(hook_fn)
    
    try:
        # Clone tensor with gradients enabled
        input_tensor = image_tensor.clone().requires_grad_(True)
        outputs = model(input_tensor)
        
        # Target the top prediction class
        pred_class = torch.argmax(outputs, dim=1).item()
        score = outputs[0, pred_class]
        
        # Backward pass to get gradients
        model.zero_grad()
        score.backward()
        
        # Activations shape: [1, 197, 768] (index 0 is CLS token, 1..196 are patches)
        act = activations[0].detach()
        patch_acts = act[:, 1:, :] # [1, 196, 768]
        
        # Compute patch energy across feature dimensions
        patch_weights = torch.mean(torch.abs(patch_acts), dim=-1) # [1, 196]
        
        # Reshape to 14x14 grid (224 / 16 = 14)
        grid_map = patch_weights.view(1, 1, 14, 14)
        
        # Upsample smoothly to original image size
        orig_w, orig_h = original_size
        heatmap_2d = F.interpolate(grid_map, size=(orig_h, orig_w), mode='bicubic', align_corners=False)
        heatmap_2d = heatmap_2d.squeeze().cpu().numpy()
        
        return _colorize_heatmap(heatmap_2d, orig_w, orig_h)
        
    finally:
        hook.remove()

def _generate_efficientnet_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor, original_size: tuple) -> str:
    """
    Computes Grad-CAM for EfficientNet-B0 at the final feature extraction layer.
    """
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
        
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
        
    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        input_tensor = image_tensor.clone().requires_grad_(True)
        outputs = model(input_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()
        score = outputs[0, pred_class]
        
        model.zero_grad()
        score.backward()
        
        if len(gradients) > 0 and len(activations) > 0:
            grads = gradients[0].detach()
            acts = activations[0].detach()
            pooled_grads = torch.mean(grads, dim=[2, 3], keepdim=True)
            cam = torch.relu(torch.sum(pooled_grads * acts, dim=1, keepdim=True))
        else:
            cam = torch.mean(torch.abs(activations[0].detach()), dim=1, keepdim=True)
            
        orig_w, orig_h = original_size
        cam_up = F.interpolate(cam, size=(orig_h, orig_w), mode='bicubic', align_corners=False)
        heatmap_2d = cam_up.squeeze().cpu().numpy()
        
        return _colorize_heatmap(heatmap_2d, orig_w, orig_h)
    finally:
        h1.remove()
        h2.remove()

def _colorize_heatmap(heatmap_2d: np.ndarray, orig_w: int, orig_h: int) -> str:
    # Normalize between 0.0 and 1.0
    h_min, h_max = heatmap_2d.min(), heatmap_2d.max()
    if h_max > h_min:
        heatmap_norm = (heatmap_2d - h_min) / (h_max - h_min)
    else:
        heatmap_norm = np.zeros_like(heatmap_2d)
        
    # Colorize using custom Jet/Turbo RGB gradient
    heatmap_rgb = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    r = np.clip(1.5 * heatmap_norm - 0.5, 0, 1)
    g = np.clip(1.0 - 2.0 * np.abs(heatmap_norm - 0.5), 0, 1)
    b = np.clip(0.8 - 1.5 * heatmap_norm, 0, 1)
    alpha = np.clip(heatmap_norm * 1.2, 0.2, 0.9)
    
    heatmap_rgb[..., 0] = (r * 255).astype(np.uint8)
    heatmap_rgb[..., 1] = (g * 255).astype(np.uint8)
    heatmap_rgb[..., 2] = (b * 255).astype(np.uint8)
    heatmap_rgb[..., 3] = (alpha * 255).astype(np.uint8)
    
    overlay_img = Image.fromarray(heatmap_rgb, mode="RGBA")
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{b64_str}"
