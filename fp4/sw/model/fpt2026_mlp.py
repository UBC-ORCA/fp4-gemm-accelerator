import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from mptorch import FloatingPoint, FixedPoint
import mptorch.quant as qpt
from mptorch.quant import compute_bias, scale, unscale
from mptorch.optim import OptimMP
from mptorch.utils import trainer
import random
import numpy as np
import argparse
from utils import *

parser = argparse.ArgumentParser(description="MLP MNIST Example")
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    metavar="S",
    help="random seed (default: 0)"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.0008,
    metavar="N",
    help="initial learning rate (default: 0.05)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="N",
    help="momentum value to be used by the optimizer (default: 0.9)",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0,
    metavar="N",
    help="weight decay value to be used by the optimizer (default: 0.0)",
)
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="disables CUDA training"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = "cuda" if args.cuda else "cpu"

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

"""
Dataloaders and reshaping
"""

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = torchvision.datasets.MNIST(
    "./data", train=True, transform=transform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(
    "./data", train=False, transform=transform, download=False
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 28 * 28)

"""
Model Quantization and Setup
"""

rounding = "nearest"

# FIXED POINT TRAINING
# wl_val = 16
# fl_val = 8
# mac_format = FixedPoint(wl=32, fl=16) 
# w_format   = FixedPoint(wl=wl_val, fl=fl_val)

# quant_w = lambda x: qpt.fixed_point_quantize(x, wl=wl_val, fl=fl_val, rounding="nearest")
# quant_g = lambda x: qpt.fixed_point_quantize(x, wl=wl_val, fl=fl_val, rounding="nearest")
# acc_q   = lambda x: qpt.fixed_point_quantize(x, wl=32, fl=16, rounding="stochastic")

# FP TRAINING
expMac = 8
manMac = 23

expWeight = 2
manWeight = 1

expGrad = 8
manGrad = 23

expAcc = 8
manAcc = 23

mac_format = FloatingPoint(exp=expMac, man=manMac, subnormals=True, saturate=False)
w_format   = FloatingPoint(exp=expWeight, man=manWeight, subnormals=True, saturate=False)
g_format   = FloatingPoint(exp=expGrad, man=manGrad, subnormals=True, saturate=False)
i_format   = FloatingPoint(exp=expWeight, man=manWeight, subnormals=True, saturate=False)

quant_g = lambda x: qpt.float_quantize(x, exp=expGrad, man=manGrad, rounding=rounding, saturate=False, subnormals=True)
quant_w = lambda x: qpt.float_quantize(x, exp=expWeight, man=manWeight, rounding=rounding, saturate=False, subnormals=True)
acc_q   = lambda x: qpt.float_quantize(x, exp=expAcc, man=manAcc, rounding="stochastic", subnormals=True, saturate=False)

layer_formats = qpt.QAffineFormats(
    fwd_mac=(mac_format),
    fwd_rnd=rounding,
    bwd_mac=(mac_format),
    bwd_rnd=rounding,
    use_scaling= True,
    grad_scaled_format=g_format,
    weight_scaled_format=w_format,
    input_scaled_format=i_format,
    weight_gemm_quant=quant_w,
    input_gemm_quant=quant_w,
    grad_gemm_quant=quant_g,
)

model = nn.Sequential(
    Reshape(),
    qpt.QLinear(784, 128, formats=layer_formats),
    nn.Hardtanh(min_val=-1.0, max_val=1.0),
    qpt.QLinear(128, 96, formats=layer_formats),
    nn.Hardtanh(min_val=-1.0, max_val=1.0),
    qpt.QLinear(96, 10, formats=layer_formats)
).to(device)

base_optimizer = SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)

optimizer = OptimMP(
    base_optimizer,
    acc_quant=acc_q,
    momentum_quant=acc_q,
)

trainer(
    model,
    train_loader,
    test_loader,
    num_epochs=args.epochs,
    lr=args.lr_init,
    batch_size=args.batch_size,
    optimizer=optimizer,
    device=device
)

print_ranges(model, test_loader, device)

export_weights_header(model)              # FP32 export
# export_weights_header_fixed(model)      # fixed-point Qm.n export
export_weights_header_fp4_packed(model)   # packed FP4
export_weights_header_fp4_int16(model)    # one FP4 per int16

# Test data
export_test_data_header(test_dataset, n_samples=10)       
