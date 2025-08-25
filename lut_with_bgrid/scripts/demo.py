"""
Demo script for LUT with Bilateral Grid image enhancement.
Usage: lut-demo path/to/image.jpg 
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

from lut_with_bgrid import LUTwithBGrid
from PIL import Image
import cv2


def main():
    """Main function for the demo script."""
    parser = argparse.ArgumentParser(description="Demo script for LUT with Bilateral Grid")
    parser.add_argument("input_path", type=Path, help="path to input image")



    parser.add_argument("--input_color_space", type=str, default="sRGB",
                       help="input color space: sRGB or XYZ")
    parser.add_argument("--lut_interpolation", type=str, default="tetra",
                       help="method of LUT grid interpolation: tri or tetra")
    parser.add_argument("--pretrained_path", type=Path, default=Path("./pretrained/FiveK_sRGB.pth"),
                       help="path of pretrained model")
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

    opt = parser.parse_args()

    # use gpu when detect cuda
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")

    

    if "PPR10K" in opt.pretrained_path.name:
        backbone_type = 'resnet'
        lut_n_ranks = 10
    elif "FiveK" in opt.pretrained_path.name:
        backbone_type = 'cnn'
        lut_n_ranks = 8
    else:
        raise ValueError(f"Unknown model {opt.pretrained_path}")

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

    # Process image
    print(f"Processing image: {opt.input_path}")
    img = cv2.imread(opt.input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Remove alpha channel if needed
    if img.shape[2] == 4:
        img = img[:, :, :3]

    real_A = TF.to_tensor(img).to(device)
    real_A = real_A.unsqueeze(0)

    with torch.no_grad():
        result, _, _, _, _ = lut_bgrid_inst(real_A)

        
    # Convert to image format and save
    result = result.squeeze().mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 1000, 1000)
    cv2.imshow("result", result)
    
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()