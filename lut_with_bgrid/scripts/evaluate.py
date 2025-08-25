#!/usr/bin/env python3
"""
Evaluation script for LUT with Bilateral Grid image enhancement.
Usage: lut-evaluate --dataset_name fivek --input_color_space sRGB
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from lut_with_bgrid import LUTwithBGrid
from lut_with_bgrid.datasets import (ImageDataset_PPR10k, ImageDataset_sRGB,
                                     ImageDataset_XYZ)


def main():
    """Main function for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluation script for LUT with Bilateral Grid")
    parser.add_argument("--dataset_name", type=str, default="fivek",
                       help="name of the dataset: fivek or ppr10k")
    parser.add_argument("--input_color_space", type=str, default="sRGB",
                       help="input color space: sRGB or XYZ")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained/FiveK_sRGB.pth",
                       help="path of pretrained model")
    parser.add_argument("--lut_interpolation", type=str, default="tetra",
                       help="method of LUT grid interpolation: tri or tetra")
    parser.add_argument("--lut_n_vertices", type=int, default=17,
                       help="number of LUT vertices")
    parser.add_argument("--grid_n_vertices", type=int, default=17,
                       help="number of GRID vertices")
    parser.add_argument("--grid_n_ranks", type=int, default=8,
                       help="number of GRID generator ranks")
    parser.add_argument("--grid_interpolation", type=str, default="tri",
                       help="method of GRID grid interpolation: tri or tetra")
    parser.add_argument("--n_grids", type=int, default=6,
                       help="number of GRID generator output channel")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="directory to save enhanced images")

    opt = parser.parse_args()

    # use gpu when detect cuda
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")

    if opt.dataset_name == "ppr10k":
        backbone_type = 'resnet'
        lut_n_ranks = 10
    else:
        backbone_type = 'cnn'
        lut_n_ranks = 8

    print("Creating LUT with Bilateral Grid model...")
    lut_bgrid_inst = LUTwithBGrid(
        backbone_type=backbone_type,
        lut_n_vertices=opt.lut_n_vertices,
        lut_n_ranks=lut_n_ranks,
        lut_interpolation=opt.lut_interpolation,
        grid_n_vertices=opt.grid_n_vertices,
        grid_n_ranks=opt.grid_n_ranks,
        grid_interpolation=opt.grid_interpolation,
        n_grids=opt.n_grids
    )

    device = torch.device('cuda' if cuda else 'cpu')
    lut_bgrid_inst = lut_bgrid_inst.to(device)

    # Load pretrained models
    print(f"Loading pretrained model from {opt.pretrained_path}")
    lut_bgrid_inst.load_state_dict(torch.load(opt.pretrained_path, map_location=device))
    lut_bgrid_inst.eval()

    # Create dataset based on parameters
    if opt.dataset_name == "fivek":
        if opt.input_color_space == "sRGB":
            dataset = ImageDataset_sRGB(root="./FiveK_dataset", mode="test")
        else:
            dataset = ImageDataset_XYZ(root="./FiveK_dataset", mode="test")
    else:  # ppr10k
        dataset = ImageDataset_PPR10k(root="./PPR10K_dataset", mode="test")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # Create output directory
    os.makedirs(opt.output_dir, exist_ok=True)

    # Process all images
    total_images = len(dataloader)
    print(f"Processing {total_images} images...")

    for i, batch in enumerate(dataloader):
        real_A = batch["input"]
        real_B = batch["target"]
        
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        with torch.no_grad():
            fake_B, _, _, _, _ = lut_bgrid_inst(real_A)

        # Save the enhanced image
        output_filename = f"enhanced_{i:04d}.png"
        output_path = os.path.join(opt.output_dir, output_filename)
        save_image(fake_B.data, output_path)

        if (i + 1) % 10 == 0 or i == total_images - 1:
            print(f"Processed {i + 1}/{total_images} images")

    print("✅ Evaluation completed!")
    print(f"Enhanced images saved to: {opt.output_dir}")
    print("You can now run MATLAB evaluation scripts on these results.")


if __name__ == "__main__":
    main()