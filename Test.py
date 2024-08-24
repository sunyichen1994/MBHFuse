import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn

from data_loader.msrs_data import MSRS_data
from models.Common import YCrCb2RGB, clamp
from models.Fusion import MBHFuse
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def init_seeds(seed=0):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MBHFuse')
    parser.add_argument('--dataset_path', metavar='DIR', default=r'./test_data/TNO',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='results/TNO')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--fusion_pretrained', default='pretrained/fusion_model_epoch_59.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=3407, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)


    test_dataset = MSRS_data(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.arch == 'fusion_model':
        model = MBHFuse()
        model = model.cuda()

        if args.cuda and torch.cuda.device_count() > 1:
             model = nn.DataParallel(model)

        model.load_state_dict(torch.load(args.fusion_pretrained))
        model.eval()

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")

        test_tqdm = tqdm(test_loader, total=len(test_loader))
        with torch.no_grad():
            for _, vis_y_image, cb, cr, inf_image, name in test_tqdm:
                vis_y_image = vis_y_image.cuda()
                cb = cb.cuda()
                cr = cr.cuda()
                inf_image = inf_image.cuda()

                fused_image = model(vis_y_image, inf_image)
                fused_image = clamp(fused_image)

                rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
                rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
                rgb_fused_image.save(f'{args.save_path}/{name[0]}')
