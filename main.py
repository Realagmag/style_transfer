#!/usr/bin/env python3

import argparse
import gdown
import os
import sys
import torch
import torch.optim as optim
from autoencoder import StrongUNetNoSkips
from helper_functions import load_image, tensor_to_image, gram_matrix, get_features
from torchvision import models


def transfer_style_autoencoder(args):
    if args.style == "colorful_glass":
        model_path = "colorful_glass_autoencoder.pth"
        url = 'https://drive.google.com/file/d/18Kmbmg48W-rEqTk3RGF48sJaml3EzUtQ/view?usp=sharing'
    elif args.style == "van_gogh":
        model_path = "van_gogh_autoencoder.pth"
        url = 'https://drive.google.com/file/d/1AF908RR-osRQ27Hq1IUygKDUuRKEkdGh/view?usp=sharing'
        
    if not os.path.exists(model_path):
        print("\nDownloading model params...'")
        gdown.download(url, model_path, quiet=False)
        print("Model downloaded successfully! Proceeding.\n")
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = StrongUNetNoSkips()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image = load_image(args.input)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    output_image = tensor_to_image(output)
    output_image.save(args.output)
    print(f"✅ Image saved: {args.output}\n")


def transfer_style_vgg(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    content_img = load_image(args.content).to(device)
    style_img = load_image(args.style).to(device)
    target = content_img.clone().requires_grad_(True).to(device)

    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False

    content_layers = ["16", "21"]
    style_layers = ["0", "5", "10", "19", "28"]
    layers = content_layers + style_layers

    content_feats = get_features(content_img, vgg, content_layers)
    style_feats = get_features(style_img, vgg, style_layers)
    style_grams = {l: gram_matrix(style_feats[l]) for l in style_layers}

    optimizer = optim.Adam([target], lr=args.lr)

    print(f"\nStarting VGG Style Transfer with {args.steps} steps...")

    try:
        for i in range(1, args.steps + 1):
            optimizer.zero_grad()
            feats_t = get_features(target, vgg, layers)

            c_loss = torch.mean(
                (feats_t[content_layers[0]] - content_feats[content_layers[0]]) ** 2
            )

            s_loss = 0
            for l in style_layers:
                Gt = gram_matrix(feats_t[l])
                A = style_grams[l]
                s_loss += torch.mean((Gt - A) ** 2)

            loss = args.alpha * c_loss + args.beta * s_loss
            loss.backward()
            optimizer.step()

            if i % 50 == 0 or i == 1:
                print(
                    f"   Iter {i:4d}/{args.steps} | Loss: {loss.item():8.2f} | "
                    f"Content: {c_loss.item():8.2f} | Style: {s_loss.item():8.8f}"
                )

    except KeyboardInterrupt:
        print("\nStopped by user. Saving current output...")

    output = tensor_to_image(target)
    output.save(args.output)
    print(f"✅ Image saved: {args.output}\n")


def main():
    parser = argparse.ArgumentParser(
        description="🎨 Style Transfer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How to use:

  Autoencoder (fast style transfer):
    %(prog)s autoencoder -i content.jpg -s colorful_glass -o output.jpg
    %(prog)s autoencoder -i content.jpg -s van_gogh -o output.jpg

  VGG (classic style transfer):
    %(prog)s vgg -c content.jpg -s style.jpg -o output.jpg
    %(prog)s vgg -c content.jpg -s style.jpg -o output.jpg --steps 2000 --alpha 1e5

        """,
    )

    subparsers = parser.add_subparsers(
        dest="method", help="Choose your style transfer method"
    )

    # ========== Autoencoder Parser ==========
    autoencoder_parser = subparsers.add_parser("autoencoder")
    autoencoder_parser.add_argument(
        "-i", "--input", required=True, help="Path to input image (content)"
    )
    autoencoder_parser.add_argument(
        "-s",
        "--style",
        required=True,
        choices=["colorful_glass", "van_gogh"],
        help="What style to use: colorful_glass or van_gogh",
    )
    autoencoder_parser.add_argument(
        "-o",
        "--output",
        default="output.jpg",
        help="Where to save output image: output.jpg as default",
    )

    # ========== VGG Parser ==========
    vgg_parser = subparsers.add_parser("vgg")
    vgg_parser.add_argument(
        "-c", "--content", required=True, help="Path to input image (content)"
    )
    vgg_parser.add_argument(
        "-s", "--style", required=True, help="Path to input image (style)"
    )
    vgg_parser.add_argument(
        "-o",
        "--output",
        default="vgg_output.jpg",
        help="Where to save output image: vgg_output.jpg as default",
    )
    vgg_parser.add_argument(
        "--alpha", type=float, default=1, help="Content weight (default: 1)"
    )
    vgg_parser.add_argument(
        "--beta", type=float, default=1e12, help="Style weight (default: 1e12)"
    )
    vgg_parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of picture optimisation iterations (default: 1000)",
    )
    vgg_parser.add_argument(
        "--lr", type=float, default=0.02, help="Learning rate (default: 0.02)"
    )
    # Parsowanie argumentów
    args = parser.parse_args()

    if args.method is None:
        parser.print_help()
        sys.exit(1)

    # Sprawdzanie plików wejściowych
    if args.method == "autoencoder":
        if not os.path.exists(args.input):
            print(f"❌ Error: Missing file '{args.input}'")
            sys.exit(1)
    elif args.method == "vgg":
        if not os.path.exists(args.content):
            print(f"❌ Error: Missing file '{args.content}'")
            sys.exit(1)
        if not os.path.exists(args.style):
            print(f"❌ Error: Missing file '{args.style}'")
            sys.exit(1)

    # Wykonanie transferu stylu
    print("\n" + "=" * 60)
    print("🎨 STYLE TRANSFER CLI")
    print("=" * 60)

    if args.method == "autoencoder":
        transfer_style_autoencoder(args)
    elif args.method == "vgg":
        transfer_style_vgg(args)


if __name__ == "__main__":
    main()
