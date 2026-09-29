import argparse
import torch
from pathlib import Path
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *

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

    parser.add_argument('--final_size', type=int, default=256,
                        help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512,
                        help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512,
                        help='Size of style image')
    parser.add_argument('--crop', action='store_true', default=True,
                        help='Crop image')

    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    #Save argument values
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')
    
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)
    
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset,
                                    batch_size=args.batch_size,
                                    shuffle = True,
                                    pin_memory=True,
                                    drop_last=True)
    style_dataloader = DataLoader(style_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    
    print('Number of batches in content dataset: ', len(content_dataloader))
    print('Number of batches in style dataset: ', len(style_dataloader))
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch:  1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    print('Training...')