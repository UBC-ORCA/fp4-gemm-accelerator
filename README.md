# fp4-gemm-accelerator
4-Bit Floating Point Accelerator for Dot Product and Outer Products

## Starting Point
Please make directories for int8 and fp4-old. We will use these files to populate a new directory, fp4.

- fp4/rtl should have the Verilog code
- fp4/sw should have the software to run

Our objective is to create a complete EV2 system (with inner + outer product FP4 accelerator) to compute INFERENCES on the MNIST dataset.

The software will be in plain C and/or assembly, as needed.

Note that FP4 is only for the matrix multiply inputs. The outputs format is yet TBD -- originally, I had written BFLOAT16 here, but I've come to a different realization after studying the problem in a bit more detail. Here are my thoughts so far:

- convert FP4 inputs to int5 (perfect/lossless), where the integer counts the number of "quanta" in the float (one quanta is the smallest value). for FP4 in OCP, the quanta is 0.5, and the valueset of FP4 (for OCP) is the set of 8 values +/-{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 }. This can be represented by integers +/-{ 0, 1, 2, 3, 4, 6, 8, 12 }. Note that even if the valueset changes to IEEE P3109, the integer values and the FP4-to-int logic remains the same!

- with values maxing out at +/-12, we can accumulate about 10 MACs (10*12=120) before overflowing an int8. It is tempting to stop accumulating after computing a full 8 * 8 tile of A * B = C, to avoid overflows. However, additional tiles from a strip of A and a stripe of B eventually need to be added together into the same C tile.

- accumulating into int16 would allow A and B matrix dimension K = 256*10 = 2560 before overflowing. This is "pretty large" for a tiny inference engine, so let's go with it.

- let's read out the C matrix in two different ways, for software flexibility. 1) read out INT16 values, 2) read out INT16 converted to BINARY16 (or BFLOAT16). The readout will be 32b at a time, so at most you will need two int-to-float conversion modules. The range of BINARY16 is about +/- 60,000, so it should easily handle the INT16 range (it will end up rounding many data values into the buckets it has; also, BINARY16 can represent many many values < 0.25, which are impossible to produce while accumulating FP4 * FP4 values (their smallest value would be 0.5 * 0.5 = 0.25).

- eventually, we need to end up with FP4 values again after any bias and activation functions are applied
 
The outer product accelerator used for INT8 is a great starting point -- it even has dedicated load and store operations for 32b words into the MAC array. It computed INT8 * INT8 = INT32 and did INT32 accumulation. The MAC array then holds this much data:

INT8: 4 * 4 * 32b = 16 * 32b = 16 * 32b words

For FP4, it will be computing FP4 * FP4 = INT16 / BINARY16, which will hold this much data:

FP4: 8 * 8 * 16b = 64 * 16b = 32 * 32b words (packed 2 x INT16 in every 32b word)

However, to finish off the inference computation, we need some other things:
- adding BIAS terms
- HardTanh (or similar) activation function


Robert, we may need some other changes:
- remove BatchNorm(); if it is necessary during training, it can be removed when converting to inference by folding https://medium.com/data-science/speed-up-inference-with-batch-normalization-folding-8a45a83a89d8
- go back to pixels as primary inputs (no sin/cos) since accuracy is not the objective
- tensor_scaling is typically dynamic during training; during inference, is it possible for us to do a sweep and make it static ?
- we need to consider how to finish computations in packed 2 x INT16 and ultimately convert data to 8 x FP4
- working with INT16 is far easier than BFLOAT16, and more accurate, as long as you take care to avoid overflows (maybe use INT32 as intermediate values)

  
Robert/Steven,
- NO LONGER NEEDED: somehow we need to support BFLOAT16 operations, or we go to binary32, possibly using SoftFloat library, or something much smaller like tinyfloat modified for bfloat16:
https://github.com/ssloy/tinyfloat
- INT16: which operations do we really need? tensor scaling and conversion to FP4 need to happen efficiently

I have written a skeleton of what it looks like to break down a simple NN with a few layers into function calls to do the MAC accelerator instruction. It's far from optimal. I'll upload it here into github when it's ready and name it `tilemm.c`.

You may find it instructive to also look at this very simple C code to train a small MLP network. At a later stage in the project, we (read: Steven) may wish to add a stretch goal of doing training, not just inference.
https://github.com/djbyrne/mlp.c/blob/main/mlp_simple.c
