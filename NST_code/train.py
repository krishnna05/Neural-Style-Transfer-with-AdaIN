import argparse
import torch
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default='C:\Users\Krishna Vishwakarma\OneDrive\Desktop\AI-NST\NST_code\content_data',
                        help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default='C:\Users\Krishna Vishwakarma\OneDrive\Desktop\AI-NST\NST_code\style_data',
                        help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default='C:\Users\Krishna Vishwakarma\OneDrive\Desktop\AI-NST\NST_code\vgg19-d01eb7cb.pth',
                        help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Name of experiment')

    

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Create experiment directory
    experiment_dir = Path(args.experiment)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-trained VGG model
    vgg_model_path = Path(args.vgg)
    if not vgg_model_path.is_file():
        raise FileNotFoundError(f"VGG model not found at {vgg_model_path}")
    
    vgg_model = torch.load(vgg_model_path)
    print(f"Loaded VGG model from {vgg_model_path}")

    # Load content and style datasets
    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)

    if not content_dir.is_dir():
        raise FileNotFoundError(f"Content directory not found at {content_dir}")
    
    if not style_dir.is_dir():
        raise FileNotFoundError(f"Style directory not found at {style_dir}")

    print(f"Content dataset located at: {content_dir}")
    print(f"Style dataset located at: {style_dir}")