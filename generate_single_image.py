#!/usr/bin/env python3
import argparse
from torch.utils.data import DataLoader
from vae_swin_unet import ImageDataset, get_transforms
from image_generate import ImageGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct a single image using SwinUNetVAEHRSkip")
    parser.add_argument("--data_dir", type=str, default="v60/val", help="Directory with images to sample from")
    parser.add_argument("--model_path", type=str, default="vae_unet_hrskip_final_model.pth", help="Path to model checkpoint")
    parser.add_argument("--img_size", type=int, default=224, help="Target image size (HxW)")
    parser.add_argument("--latent_dim", type=int, default=512, help="Latent dimension size")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--save_path", type=str, default="single_image.png", help="Output image path")
    parser.add_argument("--show", action="store_true", help="Show matplotlib window")
    return parser.parse_args()


def main():
    args = parse_args()

    # build one-sample dataloader
    transform = get_transforms(img_size=args.img_size)
    dataset = ImageDataset(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    generator = ImageGenerator(
        model_path=args.model_path,
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        device=args.device,
    )

    generator.reconstruct_single_image(
        data_loader=loader,
        num_samples=2,
        save_path=args.save_path,
        show=args.show,
    )


if __name__ == "__main__":
    main()