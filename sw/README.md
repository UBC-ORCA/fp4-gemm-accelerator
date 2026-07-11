# Summary
This directory contains the TB used by verilator for inference, headers for the weights, as well as three versions of the inference software:
* inference_baseline: FP4 are decoded as signed int4 and directly multiplied without the use of any special instructions. This is used to get a baseline speed, and makes inaccurate predictions.
* inference_software: All hardware insturctions used are actually emulated in software for this version, so it matches the hardware accelerated code just without those hardware instructions.
* inference_hardware: The hardware accelerated version of the inference code that utilizes the insturctions vmac64, vle32, zzMAC64, and mv.

Each directory contains the .c file itself and make file.

# Headers & Test Images (headers/)
## Test Images
The test images can be found as .bin files. There are two seperate .bin files: test_10k and test_80. These contain the full 10k images from the MNIST test dataset, and the first 80 images from the same dataset.

## Headers
AS OF JULY 10TH, ONLY A BLOCK SIZE OF 8 WORKS WITH THE HARDWARE/VMAC64 VERSION OF THE SOFTWARE.

The header files are only for weights and have a common naming scheme:
```
weights_blk[BLOCK_SIZE/K1_STEP_SIZE]_pkg[PACKAGE_FORMAT]_scale[SCALE_FACTOR_FORMAT].h
```
* BLOCK_SIZE/K1_STEP_SIZE: Block szie for scaling. Can either be 8, 16 (NV), or 32 (MX).
* PACKAGE_FORMAT: Format used for packing the FP4 weights. This will either be INT16 (1xFP4) and UINT32 (8xFP4).
* SCALE_FACTOR_FORMAT: Format of the scaling factors. Options are E8M0 for MX scaling, and E4M3 for NV scaling.

Do note that due to training limitations, the same scale factor is reused per block.


# Building
To generate the .elf, .map, and .hex files, CD to the directory of the version you want to build and run:
```
make -f inference.mk
```

To run the software in verilator, please CD to the rtl/ directory and run `./run_inference.sh --help`.


