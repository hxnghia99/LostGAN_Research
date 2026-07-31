
import cv2
import glob
import os
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

def load_batch_cv2(image_paths, start_idx, batch_size, target_size=(299, 299)):
    """Loads a batch of images using OpenCV, converts to RGB, and formats for PyTorch."""
    batch_paths = image_paths[start_idx : start_idx + batch_size]
    batch_tensors = []
    
    for path in batch_paths:
        # Load image (BGR)
        img = cv2.imread(path)
        if img is None:
            continue  # Skip corrupted or unreadable files
            
        # 1. Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Resize to Inception network requirements
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 3. HWC (Height, Width, Channel) -> CHW (Channel, Height, Width)
        img = img.transpose(2, 0, 1)
        
        batch_tensors.append(torch.from_numpy(img))
        
    if not batch_tensors:
        return None
        
    # Stack into a single batch tensor: (Batch_Size, 3, 299, 299)
    # Ensure it's uint8, which is what torchmetrics FID expects
    return torch.stack(batch_tensors).to(torch.uint8)

def compute_fid_with_cv2(real_dir, fake_dir, batch_size=32, device="cuda"):
    # Gather all image paths from both directories
    # Supports common extensions: jpg, jpeg, png, bmp
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    
    real_paths = []
    fake_paths = []
    for ext in extensions:
        real_paths.extend(glob.glob(os.path.join(real_dir, ext)))
        fake_paths.extend(glob.glob(os.path.join(fake_dir, ext)))
        
    if not real_paths or not fake_paths:
        raise ValueError("Make sure both folders contain valid images and paths are correct.")

    # Initialize the metric
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Process Real Images
    print(f"Processing {len(real_paths)} real images...")
    for i in tqdm(range(0, len(real_paths), batch_size)):
        batch = load_batch_cv2(real_paths, i, batch_size)
        if batch is not None:
            fid.update(batch.to(device), real=True)

    # Process Fake Images
    print(f"Processing {len(fake_paths)} fake images...")
    for i in tqdm(range(0, len(fake_paths), batch_size)):
        batch = load_batch_cv2(fake_paths, i, batch_size)
        if batch is not None:
            fid.update(batch.to(device), real=False)

    # Calculate and return final score
    fid_score = fid.compute()
    return fid_score.item()

# --- Run Script ---
if __name__ == "__main__":
    REAL_FOLDER = "outputs/fid_compute/real"
    FAKE_FOLDER = "outputs/fid_compute/fake"

    score = compute_fid_with_cv2(REAL_FOLDER, FAKE_FOLDER, batch_size=64)
    print(f"\nFinal FID Score: {score:.4f}")