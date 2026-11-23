"""Image loading and preprocessing for VGGT (same behaviour as the official vggt.utils.load_fn)."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF


def _open_rgb(path):
    img = Image.open(path)
    if img.mode == "RGBA":  # composite transparent images on white
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    return img.convert("RGB")


def _pad_to(img, height, width, value=1.0):
    h_pad, w_pad = height - img.shape[1], width - img.shape[2]
    if h_pad > 0 or w_pad > 0:
        top, left = h_pad // 2, w_pad // 2
        img = torch.nn.functional.pad(img, (left, w_pad - left, top, h_pad - top), mode="constant", value=value)
    return img


def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """Pad each image to a black square, resize to target_size. Returns images (N, 3, s, s) and, per image,
    [x1, y1, x2, y2, width, height]: where the original image lies in the square and its original size."""
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    to_tensor = TF.ToTensor()
    images, original_coords = [], []
    for path in image_path_list:
        img = _open_rgb(path)
        width, height = img.size
        max_dim = max(width, height)
        left, top = (max_dim - width) // 2, (max_dim - height) // 2
        scale = target_size / max_dim
        original_coords.append(np.array([left * scale, top * scale, (left + width) * scale, (top + height) * scale,
                                         width, height]))
        square = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square.paste(img, (left, top))
        images.append(to_tensor(square.resize((target_size, target_size), Image.Resampling.BICUBIC)))
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()
    if len(image_path_list) == 1 and images.dim() == 3:
        images, original_coords = images.unsqueeze(0), original_coords.unsqueeze(0)
    return images, original_coords


def load_and_preprocess_images(image_path_list, mode="crop"):
    """Load images as (N, 3, H, W) in [0, 1] with sides that are multiples of 14.

    mode="crop": width 518, height scaled and rounded to a multiple of 14, centre-cropped to at most 518.
    mode="pad": the longer side becomes 518, the shorter side is rounded to a multiple of 14 and padded with white
    to a 518 x 518 square. Images of different shapes are white-padded to a common size.
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    to_tensor = TF.ToTensor()
    target_size = 518
    images, shapes = [], set()
    for path in image_path_list:
        img = _open_rgb(path)
        width, height = img.size
        if mode == "pad" and width < height:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        img = to_tensor(img.resize((new_width, new_height), Image.Resampling.BICUBIC))
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]
        if mode == "pad":
            img = _pad_to(img, target_size, target_size)
        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        images = [_pad_to(img, max_h, max_w) for img in images]

    images = torch.stack(images)
    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)
    return images
