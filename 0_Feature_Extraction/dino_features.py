"""
Extract DINOv2 features from images.
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from transformers import AutoModel


class ImageDataset(Dataset):
    """Dataset for loading and preprocessing images."""

    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = Compose(
            [
                Resize(224, interpolation=Image.BICUBIC),
                CenterCrop(224),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path)
        except Exception:
            print(f"Warning: Could not load image at index {idx}: {image_path}")
            image = Image.new("RGB", (224, 224))
        image = self.transform(image)
        return {"image": image}

    def __len__(self):
        return len(self.image_paths)


def extract_dino_features(
    image_paths, device, model_name="facebook/dinov2-large", batch_size=64, num_workers=4
):
    """
    Extract DINOv2 features from a list of images.

    Args:
        image_paths: List of paths to images
        device: Device to run inference on ('cuda' or 'cpu')
        model_name: HuggingFace model name for DINOv2
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading

    Returns:
        numpy array of shape (num_images, feature_dim)
    """
    print(f"Extracting DINO features from {len(image_paths)} images")

    dataloader = DataLoader(
        ImageDataset(image_paths),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    # Load DINOv2 model
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["image"].to(device)
            if device == "cuda":
                images = images.to(torch.float16)
            features = model(images)["pooler_output"]
            all_features.append(features.cpu().numpy())

    all_features = np.vstack(all_features)
    return all_features


def load_image_paths_from_json(json_path, image_dir, image_key="image"):
    """
    Load image paths from a JSON or JSONL file.

    Args:
        json_path: Path to JSON/JSONL file containing image references
        image_dir: Base directory for images
        image_key: Key in the JSON objects containing the image filename

    Returns:
        List of full image paths
    """
    if json_path.endswith(".jsonl"):
        with open(json_path) as f:
            data = [json.loads(line) for line in f]
    else:
        with open(json_path) as f:
            data = json.load(f)

    image_paths = []
    for item in data:
        if image_key in item:
            image_paths.append(os.path.join(image_dir, item[image_key]))
        else:
            print(f"Warning: No '{image_key}' key found in item")
            image_paths.append(None)

    return image_paths


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load image paths
    image_paths = load_image_paths_from_json(
        args.data_path, args.image_dir, args.image_key
    )

    # Filter out None values
    valid_paths = [p for p in image_paths if p is not None]
    print(f"Found {len(valid_paths)} valid image paths")

    # Extract features
    features = extract_dino_features(
        valid_paths,
        device,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Save features
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, features)
    print(f"Saved features with shape {features.shape} to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DINOv2 features from images")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSON/JSONL file containing image references",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Base directory containing images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save extracted features (.npy)",
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default="image",
        help="Key in JSON objects containing image filename",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov2-large",
        help="HuggingFace model name for DINOv2",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    args = parser.parse_args()
    main(args)
