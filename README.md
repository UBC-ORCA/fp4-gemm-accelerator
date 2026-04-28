# fp4-gemm-accelerator
4-Bit Floating Point Accelerator for Dot Product and Outer Products

## Starting Point
Please make directories for int8 and fp4-old. We will use these files to populate a new directory, fp4.

- fp4/rtl should have the Verilog code
- fp4/sw should have the software to run

Our objective is to create a complete EV2 system (with inner + outer product FP4 accelerator) to compute inferences on the MNIST dataset.

The software will be in plain C and/or assembly, as needed.

Note that FP4 is only for the matrix multiply inputs. The outputs will be in BFLOAT16. The outer product accelerator will compute 64 MACs (BFLOAT16 = BFLOAT16 + FP4 * FP4) in parallel.

The outer product accelerator used for INT8 is a great starting point -- it even has dedicated load and store operations for 32b words into the MAC array. Twice as much data is used in the MAC array:

INT8: 4*4*32b = 16*32b = 16 * 32b words
FP4: 8*8*16b = 64*16b = 32 * 32b words (packed 2 x BFLOAT16 in every 32b word)

However, to finish off the inference computation, we need some other things:
- adding BIAS terms
 HardTanh activation function

Robert, we may need some other changes:
- remove BatchNorm()
- go back to pixels as primary inputs (no sin/cos) since accuracy is not the objective
- tensor_scaling is typically dynamic during training; during inference, is it possible for us to do a sweep and make it static ?

Robert/Steven,
- somehow we need to support BFLOAT16 operations, or we go to binary32, possibly using SoftFloat library, or something much smaller like tinyfloat modified for bfloat16:
https://github.com/ssloy/tinyfloat
- which operations do we really need?

I have written a skeleton of what it looks like to break down a simple NN with a few layers into function calls to do the MAC accelerator instruction. It's far from optimal.

You may find it instructive to also look at this very simple C code to train a small MLP network. At a later stage in the project, we (read: Steven) may wish to add a stretch goal of doing training, not just inference.
https://github.com/djbyrne/mlp.c/blob/main/mlp_simple.c
