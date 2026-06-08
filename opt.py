import argparse

parser = argparse.ArgumentParser(description="CRSDUN")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='fvgnet/', help='dataset directory')
parser.add_argument('--mask_path', type=str, default='mask/mask512x512.mat')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/CRSDUN/', help='saving_path')
parser.add_argument('--name', type=str, default='xxx_xxx', help='project name')

# Model specifications
parser.add_argument('--method', type=str, default='CRSDUN', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='Y',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='SSR',
                    help='the input mask of the network: SSR, Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument('--batch_size', type=int, default=4, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=500, help='total epoch')

parser.add_argument("--learning_rate", type=float, default=0.0004)

opt = parser.parse_args()

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False