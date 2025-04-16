import os
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from einops import rearrange, repeat
from io import BytesIO

# Import warping function from your project
from src.model.encoder.costvolume.depth_predictor_multiview import warp_with_pose_depth_candidates

# ✅ Load dataset
file_path = Path("/storage/user/sade/mvsplat/datasets/hypersim/train/single.torch")
tensor = torch.load(file_path)

# ✅ Extract all 97 images and their corresponding camera poses
images = tensor[0]["images"]  # Shape: [97, C, H, W]
poses = tensor[0]["cameras"]  # Shape: [97, 18]

# ✅ Convert poses to intrinsics & extrinsics (batch-wise)
def convert_poses(poses: torch.Tensor):
    """Converts Hypersim pose format to OpenCV-style intrinsics and extrinsics."""
    b, _ = poses.shape

    # Convert to 3x3 intrinsic matrix
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(b, 1, 1)
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert to 4x4 world-to-camera (W2C) extrinsics
    w2c = torch.eye(4, dtype=torch.float32).repeat(b, 1, 1)
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)

    return w2c.inverse(), intrinsics  # Convert W2C to C2W

# ✅ Convert all images into PyTorch tensors
def convert_images(images: list[torch.Tensor]):
    """Converts raw images into PyTorch tensors."""
    torch_images = []
    for image in images:
        image = Image.open(BytesIO(image.numpy().tobytes())).convert("RGB")  # Ensure RGB
        image = transforms.ToTensor()(image)
        torch_images.append(image)
    return torch.stack(torch_images)

# ✅ Convert dataset
extrinsics, intrinsics = convert_poses(poses)  # Get 97 extrinsics & intrinsics
torch_images = convert_images(images)  # Get 97 images as tensors

# ✅ Get batch size (should be 97)
b, c, h, w = torch_images.shape  # b = 97

# ✅ Fix intrinsics shape to be [97, 3, 3]
intr_curr = intrinsics.clone().detach().reshape(b, 3, 3)

# ✅ Fix extrinsics shape to be [97, 4, 4]
pose_curr = extrinsics.clone().detach().reshape(b, 4, 4)

# ✅ Print debugging info
print(f"Images Shape: {torch_images.shape}")  # Should be [97, 3, H, W]
print(f"Intrinsics Shape: {intr_curr.shape}")  # Should be [97, 3, 3]
print(f"Extrinsics Shape: {pose_curr.shape}")  # Should be [97, 4, 4]

# ✅ Set initial near & far values (modify as needed)
NEAR_VALUES = [5, 10, 20]  # Try different near values
FAR_VALUES = [100, 130, 170, 190]  # Try different far values

# ✅ Function to save images
def save_images(images, path):
    """Save tensor images as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = (images * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)

# ✅ Function to analyze black regions in warped images
def analyze_black_regions(image_path):
    """Detects percentage of black pixels in the warped image."""
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_np = np.array(img)
    black_pixels = np.sum(img_np == 0) / img_np.size
    return black_pixels

# ✅ Warping and near/far tuning function
def run_warping_and_tune_near_far():
    """Runs warping for different near/far values and evaluates best settings."""
    best_near, best_far, min_black_ratio = None, None, float("inf")

    for near_value in NEAR_VALUES:
        for far_value in FAR_VALUES:
            print(f"Testing Near: {near_value}, Far: {far_value}")

            # Convert near/far to tensors
            near = torch.tensor([[near_value]])
            far = torch.tensor([[far_value]])

            # ✅ Select two views for warping (e.g., first two images in batch)
            image01 = torch_images[:2]  # First two images
            image10 = image01.flip(0)  # Swap order

            # ✅ Call warping function
            image01_warped = warp_with_pose_depth_candidates(
                image10,  
                intr_curr[:2],  # First two intrinsics
                pose_curr[:2],  # First two extrinsics
                1.0 / far.unsqueeze(0).repeat([2, 1, *image10.shape[-2:]]),
                warp_padding_mode="zeros",
            )

            # ✅ Save original and warped images
            out_dir = f"warp_images/near_{near_value}_far_{far_value}/scene_001/"
            save_images(image01[0], f"{out_dir}/0_ori.png")
            save_images(image01_warped[0, :, 0], f"{out_dir}/0_warped_0.png")

            # ✅ Analyze black pixel percentage
            black_ratio = analyze_black_regions(f"{out_dir}/0_warped_0.png")
            print(f"Black %: {black_ratio:.4f}")

            # ✅ Update best near/far values
            if black_ratio < min_black_ratio:
                min_black_ratio = black_ratio
                best_near, best_far = near_value, far_value

    print(f"\n✅ Best Near/Far: Near = {best_near}, Far = {best_far}, Black % = {min_black_ratio:.4f}")
    return best_near, best_far

# ✅ Run the tuning process
best_near, best_far = run_warping_and_tune_near_far()

# ✅ Save visualization as a PNG (Cluster-Compatible)
scene_path = f"warp_images/near_{best_near}_far_{best_far}/scene_001/"
ori_image = Image.open(scene_path + "0_ori.png")
warped_image = Image.open(scene_path + "0_warped_0.png")

# ✅ Create and save a comparison plot
output_plot_path = f"warp_images/near_{best_near}_far_{best_far}/scene_001/comparison.png"
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(ori_image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(warped_image)
plt.title(f"Warped Image (Near: {best_near}, Far: {best_far})")

plt.savefig(output_plot_path)  # Save instead of .show()
print(f"✅ Comparison saved at: {output_plot_path}")