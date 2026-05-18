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

- with values maxing out at +/-144, we can accumulate about 227 MACs (32767/144=227) before overflowing an INT16. It is tempting to stop accumulating after computing a full 8 * 8 tile of A * B = C, to avoid overflows. However, additional tiles from a strip of A and a stripe of B eventually need to be added together into the same C tile.

- practically, accumulating into int16 would allow A and B matrix dimension K = 256 before overflowing. This is "pretty large" for a tiny inference engine, so let's go with it.

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


## Hardware MAC64 Instruction Set for Outer Product with FP4

**NOTE:** the old versions of these instructions are at the end of this document (for archival purposes)

### GEN3: instructions for 8 * 8 tile only, computing MAC with two VECTOR operands of 16 * 32b, each 32b holding 8 * FP4 values

This design will accumulate up to K=256 efficiently, in blocks of 16, using tiles of 8 * 8. For K>256, some loading/saving of an accumulator BRAM `U` will be required.

```
// v0-v15 hold weights
// v16-v31 hold activations
// we will re-use the activations in the I dimension
for J = 0 to 1023 step 8 // proceed across activation matrix
  for I = 0 to 1023 step 8
    // zero memory storage used to save U BRAM  (8 columns x I rows in total)
    // better yet, load BIAS values here into memory
  end I
  for K = 0 to 1023 step 256 // do the rest of the dot product dimension here, reducing 256 at a time in the innermost loop
    // 16 vector loads (into v0-v15), each loading 8 columns x 16 rows of activations
    // clear v16, so all entries are 0
    for I = 0 to 1023 step 8 // proceed down weight matrix
      // load U BRAM from position I in memory (first iteration of K it will load BIAS values)

      // COMPUTE: 1st set of 16 rows of activations:
      lw t0, 0(a0) // load activation scaling factors
      lw t1, 4(a0)
      lw t2, 0(a1) // load weight scaling factors
      lw t3, 4(a1)
      vhwMAC v0, 0(a2) // takes 16 cycles to compute: loads weights starting at 0(a2), writes to tile T 
      hwASCALE x0, t0, t1 // apply activation scaling factors (two 32b words = 8 scales * E4M3 each), executed while vhwMAC is underway
      hwWSCALE x0, t2, t3 // apply weight scaling factors, executed while vhwMAC is underway
      //
      // COMPUTE: 2nd set of 16 rows of activations:
      // load scaling factors (4 loads, hwASCALE, hwWSCALE)
      lw t0, 8(a0) // load activation scaling factors
      lw t1, 12(a0)
      lw t2, 8(a1) // load weight scaling factors
      lw t3, 12(a1)
      vhwMAC v1, 64(a2) // next set of weights is 4B * 16 = 64B offset
      hwASCALE x0, t0, t1
      hwWSCALE x0, t2, t3
      ...
      // COMPUTE: 16th set of 16 rows of activations:
      // load scaling factors (4 loads, hwASCALE, hwWSCALE)
      vhwMAC v15, 960(a2)

      // wait 16 cycles, do a dummy vhwMAC with weights of 0 and activations of 0
      vhwMAC v16, 0(a2) // dummy NOPs, time delay
      // BRAM now holds all 8 * 8 accumulated values from T in bfloat16 format
      // save U BRAM to memory at position I

      addi a2, a2, 1024 // jump to next set of weights
      addi to a0 and a1 as appropriate

    end I go to next 8 rows

  end K // go to next 256 group
  // apply ACTIVATION function and convert bfloat16 to FP4 on all U BRAM copies (all rows I x 8 columns of J)
end J // go to next 8 columns
```
The **vhwMAC** instruction computes the outer product of weights * v16, summing along the vector length of 16. Note that the weights are read from memory, while the activations come from a vector register.  After 16 cycles of accumulating products, it must snapshot all 8 * 8 x int16 values (total 1024 bits) from T so they are ready for post-processing over the next 16 clock cycles. In addition, it must also snapshot the 8 * E4M3 A scales and the 8 * E4M3 W scales (total 128 bits). This post-processing will overlap with the next vhwMAC (and any other instructions in between). After the post-processing, the fully accumulated tile will be held in a BRAM which we'll call `U`.

This design will be tiny because the 8 * 8 = 64 MAC units are tiny (int16, or even int13 would work) plus a set of snapshot registers. Each product and int16 sum should only take about 30 LUTs.

If this operates at 200MHz, then peak computation rate is 2 * 64 * 200M operations per cycle = 25.6 GOP/s.

In the code above, the innermost loop has been unrolled into 16 distinct **vhwMAC** instructions. Each **vhwMAC** instruction takes 16 cycles to execute, during which the scalar processor may continue executing other instructions in preparation for the next **vhwMAC** operation. Thus, the loading of activation/weight scaling factors is fully overlapped with the MAC computations, and the unrolled loop accumulates a total of 256 products along the k dimension.

Each of the clock cycles, 64 FP4 products are computed in parallel, converted to int9, and summed into an int13 register. After 16 cycles, the 64 int13 registers are all `full' (at risk of overflow) and need to be accumulated into a larger format. We don't want to convert all 64 values and accumulate them in a single clock cycle, because that would require 64 copies of area-intensive hardware.

Instead, we will snapshot the 64 int13 values (832 bits) and the 16 E4M3 scale factors (128 bits) at the end of the 16 cycles. Then, we will use the next 16 cycles (while a subsequent vhwMAC instruction is running) to convert the 64 values into bfloat16, scale them, and accumulate them into the `U` BRAM as bfloat16. This can be done at a minimal rate of 4 values per cycle, requiring only 4 bfloat16 accumulators and 4 or 8 multipliers. This is the post-processing step.

Post-processing each element (done 4 at a time, in parallel) consists of (a) converting int13 into a float -- this requires normalizing the value by counting leading 0s and shifting it left into a format like E4M12 if we want it to be lossless, (b) in parallel, computing the product of the two E4M3 scale factors, one for each row (from weights) and another for each column (from activations), to produce an E5M7 value, and (c) computing the product of the E4M12 accumulation and E5M7 scales, producing an E6M20 value which should then be truncated or rounded to E6M7 and accumulated with the bfloat16 value stored in the `U` BRAM. The `U` BRAM should read out 4 bfloat16 values every cycle and write 4 values back. This post-procesing has some pipeline latency, so it will require 16 + pipeline depth cycles in total to execute. This is OK, because any attempt to read the BRAM will start with the first element written, and take 16 cycles until it reaches the last value written.

After this, we will have accumulated 16*16 = 256 dot products along the K dimension into bfloat16. Going further than this is possible, but then we would not be able to re-use the A matrix which has been conveniently preloaded into vector registers outside of the for I loop (we want to keep re-using the A matrix strip, loading it only once then discarding it). Hence, to go beyond 256, we need a way to write the `U` BRAM to memory, and also to load `U` from memory -- a modest 128 bytes is required to hold each copy of `U`.

This would all be easist if the `U` memory is visible to the vector instructions as a vector. So far, a vector register is used to hold an entire `A` tile consisting of 16 * 32 bits, for a vector VLEN of 512 bits. The `U` memory needs a VLEN of 2048 bits, and must be accessed 64b per cycle (4 bfloat16 at a time), so it would make sense if a `U` BRAM appears to be two adjacent vector registers, eg v30-v31 with a VLEN of 1024 bits, or as four adjacent vector registers v28-v31 while maintaining a VLEN of 512.  (For reference, a VLEN=1024 means total vector register size is 32Kib.)

Also, vector instructions will be needed to convert bfloat16 into packed FP4. Ideally this would be done by writing 8 FP4 values (32b) per cycle into a vector register. This would require reading 8 bfloat16 values per cycle -- likely coming from v28-v31 in parallel every cycle. However, we only have 4 bfloat accumulators (adders).


## Software Needed

- MPTorch used to train basic MNIST using MLP and 3 * 3 CNNs
- avoid hardware complex things like batchnorm, if possible
- think hard about tensor scaling and activation function, and range limits during training
- software on CVE2 to run MLP and CNN models without acceleration
- without acceleration means extracting FP4 bit patterns, treating them as INT4, and computing directly with regular C code; the results will be incorrect, but this is only to set runtime baseline
- software on CVE2 to run MLP and CNN models with acceleration
- accelerated version needs to tile the larger matrices, probably batch the models so matrix * matrix multiplication is used (not matrix * vector), and worry about the tile-based transpose problem between C outputs becoming the A inputs for the next layer

### Thoughts on post-processing and tensor scaling on INT16 values (no hw assist)

The hwMAC64 accelerator produces pairs of INT16 values, either moving them into integer registers using instructions like **mv2MAC64**, or first writing them to memory using **st2MAC64**. If written to memory, they can be reloaded again using regular C data types (int16_t) without the need to extract halfwords from words. However, directly moving the results into the register file may be more efficient.

The problem then becomes final processing of INT16 results and packing them into UINT32 words holding 8 * FP4 results. This will be done during the tensor scaling and quantize steps.

For tensor scaling, I think this is the process:
- scan entire output matrix for its max value
- compute the power of 2 needed to scale this max value to 16384 or larger; call this value SCALE
- multiply all elements by SCALE so they are left-justified in the INT16 format
- convert each matrix element into FP4 (many ways, including cheating by conversion to INT4 then to <sign,UINT3> signed-magnitude format)
- alternate way: take the top 5-6 MSB of INT16 value and use a 32 or 64 entry lookup table to convert it into FP4 (may need rounding to consider the lower truncated INT16 bits)
- pack eight FP4 values into words

The lookup table would be the reverse of FP4 to INT5 (INT5 because the sign bit still needs to be acounted for):
- (b000) 0.0 = 0
- (b001) 0.5 = 1
- (b010) 1.0 = 2
- (b011) 1.5 = 3
- (b100) 2.0 = 4
- (b101) 3.0 = 6
- (b110) 4.0 = 8
- (b111) 6.0 = 12

In reverse, the missing integer values become (using round nearest, ties to even -- please verify):
- 5 would become 2.0 
- 7 would become 4.0
- 9 and 10 would become 4.0
- 11 or larger would become 6.0



## REFERENCE: Tiled Matrix Multiply

The naive version of matrix multiply uses three nested loops, in order of `i`, `j`, and `k`, where the innermost `k` loop is used to compute a dot product, also known as the **inner product**, and the `i` loop traverses rows of `W` and `F` while the `j` loop traverses columns of `A` and `F`. The core computation is `F[i][j] += W[i][k] * A[k][j]` where W are the weights, A are the activations, and F are the computed feature outputs:
```
for i = 0 to 1023
  for j = 0 to 1023
    for k = 0 to 1023
      F[i][j] += W[i][k] * A[k][j]
```
Visually, it can be drawn something like this:
```
k:
  0      AAA
   1     AAA
    2    AAA
     3   AAA
       + ---  i:
  WWWW | FFF  0
  WWWW | FFF  1
      j: 012
```

To avoid reading/writing memory for `F` each time, we can do this:
```
for i = 0 to 1023
  for j = 0 to 1023
    t = 0
    for k = 0 to 1023
      t += W[i][k] * A[k][j]
    F[i][j] += t
```
This `t` term stays in a register, acting a bit like a cache. Later, we will transform `t` into `T`, an entire 8 x 8 tile of temporary values, and operate on all of them in parallel with the **hwMAC** instruction. The last line, `F[i][j] += t` ensures new results are added to any initial value in the matrix, just like the original algorithm which also added to the initial value of `F`.

While the **inner product** works well to cache the value of `t`, it can only compute one output element at a time (the dot product). For example, suppose we use sub-word SIMD and group eight FP4 values from `W` into `rs1` and eight FP4 values from `A` into `rs2`. An inner product between W and A would compute eight multiplications and seven additions; once added to the original value of `F[i][j]`, an 8th addition appears. In other words, this inner product computes at most 8 multiply-accumulate operations (MACs).

However, an **outer product** can compute even more MACs in parallel, as long as it has enough storage to hold the intermediate sums (several values of `t`). For this, it needs a tile of values `T` which will be written (?added?) into `F` one tile at a time. To start this , we transform the algorithm loop order so the `k` loop is now the outermost loop, and the `i` and `j` loops are the innermost loops:
```
for k = 0 to 1023
  for i = 0 to 1023
    for j = 0 to 1023
      F[i][j] += W[i][k] * A[k][j]
```
If `i` and `j` are fully unrolled, and we use a wildcard `i*` and `j*` for those positions and `:*:` to denote an outer product, then we would get:
```
for k = 0 to 1023
  F[*][*] += W[i*][k] :*: A[k][j*]
```
As you can see, this reads rows of `A` and columns of `W`, producing all `i*j` products and accumulating them into the target matrix `F` over `k` iterations.

For large matrices, we must break `F` down into smaller 8 * 8 tiles `T`:
```
for I = 0 to 1023 step 8
  for J = 0 to 1023 step 8
    T = 0 // 8*8 tile, read F[i][j] if unfinished
    for k = 0 to 1023
      for i = 0 to 7 // UNROLL
        for j = 0 to 7 // UNROLL
          T[i][j] += W[I+i][k] * A[k][J+j]
    F[I+i][J+j] = T[i][j] // copy entire tile back (or accumulate using += if `T` started with unfinished `F`).
```
Here, you'll see the innermost `i` and `j` loops have just 8 iterations. This is the size of the tile `T`, and it covers just an 8 * 8 patch of the larger 1024 * 1024 `F` matrix. These two innermost loops can be fully unrolled, yielding a total of 64 MAC operations taken from the core computation, `T[i][j] += W[I+i][k] * A[k][J+j]`. This core computation uses a strip of the `W` matrix that is 8 rows tall, and a strip of the `A` matrix that is 8 columns wide; both strips are of length 1024, which is iterated over by the `k` dimension (the middle loop). These 3 inner loops only compute the results for one tile, so outer loops are added for `I` and `J` to iterate over all possible 8 * 8 tiles within `F`; those outer loops go in increments of 8, of course, to step along by one tile at a time. The code now looks like this:
```
for I = 0 to 1023 step 8
  for J = 0 to 1023 step 8
    T = 8*8 tile starting at F[i][j]
    for k = 0 to 1023
      T[0][0] += W[I+0][k] * A[k][J+0]
      T[0][1] += W[I+0][k] * A[k][J+1]
      T[0][2] += W[I+0][k] * A[k][J+2]
      ...
      T[7][5] += W[I+7][k] * A[k][J+5]
      T[7][6] += W[I+7][k] * A[k][J+6]
      T[7][7] += W[I+7][k] * A[k][J+7]
    F[I+i][J+j] = T[i][j] // copy entire 8*8 tile back
```
The code above reads partial columns of `W`, from `W[I+0]` to `W[I+7]`. It will be more performance oriented if we transpose W (either the entire matrix, or every 8 * 8 block). Suppose the entire matrix is transposed as `W' = transpose(W)`, then the inner loops become:
```
    for k = 0 to 1023
      T[0][0] += W'[k][I+0] * A[k][J+0]
      T[0][1] += W'[k][I+0] * A[k][J+1]
      T[0][2] += W'[k][I+0] * A[k][J+2]
      ...
      T[7][5] += W'[k][I+7] * A[k][J+5]
      T[7][6] += W'[k][I+7] * A[k][J+6]
      T[7][7] += W'[k][I+7] * A[k][J+7]
    F[I+i][J+j] = T[i][j] // copy entire 8*8 tile back
```
and this enables efficient reading of the 8 column values needed of `W` as 8 row values in `W'`.

Rewriting this into the OLD ISA:
```
for I = 0 to 1023 step 8
  for J = 0 to 1023 step 8
    zzMAC64; // clears entire tile; alternatively, could load tile from `F`
    for k = 0 to 1023
      WW = W'[k][I+0] to W'[k][I+7] // reads 8 elements from a row
      AA = A[k][J+0] to A[k][J+7] // reads 8 elements from a row
      hwMAC64 WW, AA
    maxMAC64 x0 // computes ReLU on entire tile
    stMAC64 x0, 0(rs1) // stores tile, 2 elements at a time; also clears tile entries
    stMAC64 x1, 4(rs1)
    stMAC64 x2, 8(rs1)
    stMAC64 x3, 12(rs1)
    ...
    stMAC64 x31, 124(rs1)
```
where `x0` to `x31` represent the contents of the tile `{ T[i][2*j], T[i][2*j+1] }`, that is `xN` where `N=4*i+j`.

The only problem with this code is the k dimension is 1024. Any value over 227 can potentially overflow the 16b integers. In that case, the tile `T` could be read out, one 16b entry at a time, converted to 32b, and added to a 32b integer matrix. This is one use-case for the **mvoMAC** and **mveMAC** instructions.

Overall, I see code looking something like this:

```
for I = 0 to 1023 step 8 begin
  for J = 0 to 1023 step 8 begin

    // the i/j loops below can be fully unrolled
    for i = 0 to 7
      for j = 0 to 7
        T[i][j] = 0 // using **zzMAC** instruction

    for K = 0 to 1023 begin // CAREFUL: may cause saturation of 16b values
      // the i/j loops below can be fully unrolled
      for i = 0 to 7
        for j = 0 to 7
          T[i][j] += W'[K][I+i] * A[K][J+j] // using **hwMAC**
    end // K

// EITHER: write out tile into the matrix
    // this needs a "store C[i][j]" instruction
    for i = 0 to 7
      for j = 0 to 7
        F[I+i][J+j] = T[i][j] // using **st2MAC** intructions
        // don't forget to load F and post-process it with bias + activation
// OR: immediately apply bias and activation
    // this needs a "rd = T[i][j]" instruction
    for i = 0 to 7
      for j = 0 to 7 begin
        r = max( T[i][j] + bias[I+i][J+j], 0 ) // integer instructions, including **mvoMAC** and **mveMAC**, and may be **maxMAC**
        F[I+i][J+j] = r // regular store instruction
      end // j

  end // J
end // I
```

Rewriting this into the ISA:
```
for I = 0 to 1023 step TS
  for J = 0 to 1023 step 2*TS
    zzMAC; // clears entire tile; alternatively, could load tile from `F`
    for k = 0 to 1023
      for z = 0 to TS-1 step 8 // step 8 because 8 * FP4 elements fit into 32b words
        rs1 = W'[k][I+z+0] to W'[k][I+z+7] // reads 8 elements from a row
        setMACW rd=[z/8], rs1
      for z = 0 to TS-1
        rs1 = A[k][J+0] to A[k][J+7] // reads 2 elements from a row
        rs2[IMM5] = shift amount from tensor scaling max int16 determination 
        setMACA rd=[z], rs1, rs2
      hwMAC
    for i = 0 to TS-1
      addMAC rs1[i], rs2 // rs2 = bias value (broadcast to entire row of T)
    maxMAC x0 // computes ReLU on entire tile
    st2MAC x0, 0(rs2)  // rs2 is destination address for F, writes out 2 values of T[0][1:0]
    st2MAC x0, 4(rs2)  // writes out 2 values of T[0][3:2]
    ...
    st2MAC x0, 124(rs2)  // writes out 2 values of T[0][63:62]
    ...
    st2MAC x31, 0(rs2)  // rs2 is destination address for F, writes out 2 values of T[31][1:0]
    st2MAC x31, 4(rs2)  // writes out 2 values of T[31][3:2]
    ...
    st2MAC x31, 124(rs2)  // writes out 2 values of T[31][63:62]
```
where `r0` to `r31` represent the contents of the tile `{ T[i][2*j], T[i][2*j+1] }`, that is `rX` where `X=4*i+j`.

The only problem with this code is the k dimension is 1024. Any value over 227 can potentially overflow the 16b integers. In that case, the tile `T` could be read out, one 16b entry at a time, converted to 32b, and added to a 32b integer matrix. This is one use-case for the **mvoMAC** and **mveMAC** instructions.

Overall, I see code looking something like this:

```
for I = 0 to 1023 step TS begin
  for J = 0 to 1023 step 2*TS begin

    // the i/j loops below can be fully unrolled
    for i = 0 to TS-1
      for j = 0 to 2*TS-1
        T[i][j] = 0 // using **zzMAC**

    for K = 0 to 1023 begin // CAREFUL: may cause saturation of 16b values
      // the i/j loops below can be fully unrolled
      for i = 0 to TS-1
        for j = 0 to 2*TS-1
          T[i][j] += W'[K][I+i] * A[K][J+j] // using **hwMAC**
    end // K

// EITHER: write out tile into the matrix
    // this needs a "store C[i][j]" instruction
    for i = 0 to TS-1
      for j = 0 to 2*TS-1
        F[I+i][J+j] = T[i][j] // using **st2MAC** intructions
        // don't forget to load F and post-process it with bias + activation
// OR: use matrix-update instructions to apply bias and activation
    // this needs a "rd = T[i][j]" instruction
    for i = 0 to TS-1
      for j = 0 to 2*TS-1 begin
        r = max( T[i][j] + bias[I+i][J+j], 0 ) // integer instructions, including **mvoMAC** and **mveMAC**, and may be **maxMAC**
        F[I+i][J+j] = r // regular store instruction
      end // j

  end // J
end // I
```

## Tensor and Block Scaling

**NOTE: PLEASE IGNORE, THIS DOES NOT CORRECTLY PERFORM BLOCK SCALING**

Tensor/block scaling prenormalizes a tensor or block to scale its maximum value to the maximal or near-maximal value in the target format when quantizing. For simplicity, we will only consider scaling by powers of 2 which require shifting and not multiplying or dividing. For now, shifting will involve simple truncation. Below, we will consider block scaling in the inference design -- however, for now, training will be done only with tensor scaling to produce a single global scaling constant that will be applied to all blocks at the inference level for that layer.

The matrix multiplication being done is `F = W * A = (c_w * W_n) * (c_a * A_n) = (c_w * c_a) * (W_n * A_n)`, where `c_w` and `c_a` are constants extracted from the tensors `W` and `A`, and `W_n` and `A_n` are normalized versions of the weight and activation matrices.

In the case of inference, `c_w` and `W_n` are fixed and known in advance. To simplify the outer product generation, we should consider the transposed version `W' = c_w * W_n'`, where the constant `c_w` is produced across columns of `W` (rows of `W_n`). We can think of `c_w` being applied at different scales -- as being unique/constant across the entire tensor, or unique across an entire column of `W`, or unique across a partial column of `W` (up to 16/32 elements per block) which is approximately equal to the height of the tile `T` (i.e., equal to `TS`). As noted in the first paragraph of this section, we will consider `c_w` a constant across the entire `W`, but inference software should be written such that it reloads `c_w` into the MAC engine each time a new tile is computed.

For scaling `A`, the scaling factor `c_a` can also be computed in advance based upon the data distribution of activations observed during training. This is valid because, during inference, we expect a similar data distribution if it is encountering input data that is similar to its training set. As the case of the weights, we will extract a single constant `c_a` during training, which is based upon the maximal value of `c_a` observed in the last few epoch in training (reset `c_a` each epoch, remember the maximum value used). Why the maximum? Each time we quantize `A`, we are looking for the scaling factor that shrinks its maximum value down to maxfloat of the target format. If `A` is examined across different batches, then one of those batches will have some larger element in `A` than a prior batch and require a larger scaling factor to bring it down to maxfloat.

Similarly, `c_a` can be applied to rows of `A`, or to partial rows of `A` up to 16/32 elements at a time. The width or the tile `T` can be up to 64, or double 32 / quadruple 16.

To apply scaling, the hardware will compute `T = W_n * A_n`, where the elements in `W_n` and `A_n` have all been prenormalized to fit the maxfloat of the format. Outside of this, for the entire tile `T`, software must compute `c_w * c_a` outside of this. The largest `T` that we can support in hardware is 32 * 64 (due to limitations of the current ISA, where we reuse the 5 bits in a register specifier to indicate row `T[rs1]` or element-pair `{ T[rs1][2*rs2], T[rs1][2*rs2+1] }', for example). Within that `T`, we may have 1 or 2 different values of `c_w` for block scaling on the weights (eg, block size 16), and likewise have 1/2/4 different values of `c_a`. At most, then, we have `{c_w1, c_w2} x {c_a1, c_a2, c_a3}' different scaling factors being applied, yielding up to 6 different minitiles in T.

For our purposes, where we are assuming all `c_w` are the same and all `c_a` are the same and apply to the entire tensor, this is not hard to track. However, during inference, we want to mimic the smaller block-size scaling factors being used. This means we need to compute (up to 6) scaling factors for the minitiles.

After matrix multiply, we need to apply the bias terms. If this is done inside the hardware tile T, the bias term needs to be prenormalized by `c_w * c_a` before it is added. If it is done outside of the hardware thile T, it can probably be done after restoring `T` to the full version when restoring it into the output matrix `F[I][J] = c_w * c_a * T`. Don't forget to properly apply the activation function as well (with the correct scale, if using hardtanh).

Finally, let's think of going to the next layer. Suppose, at first, that the larger matrix F[I][J] is computed in int32 rather than int16. This would probably allow us to compute `c_w * c_a * T` as int32 (hopefully without overflowing). However, when moving to the next layer, the matrix `F` will be tiled again, and also have block scaling applied again, so layer F_i becomes A_i+1 as follows `A_i+1 = (c_wi * c_ai) * T_i` which is then prenormalized to `A_i+1 = c_a{i+1} * A_n{i+1}`, yielding `A_n{i+1} = A_i+1 / c_a{i+1} =  ((c_wi * c_ai)/c_a{i+1}) * T_i = cc_{i+1} * T_i` where `cc_{i+1} = (c_wi * c_ai)/c_a{i+1)`. Again, since these are powers of 2, it can be done by adding/subtracting their respective exponent values in logspace. Also, since this is inference, we can potentially treat all of these as *constants* based upon the maximal values seen during training. These scaling factors would be applied at a block level in the inference code, computing `A_n{i+1} = cc_{i+1} * T_i` (again, where T_i is post-activation, assuming bias and activations were properly applied to the prenormalized values).

This means that while loading the int16 values of `A` into our MAC engine, we need to do an arithmetic-shift by some power of 2 indicated by the `cc+{i+1}` factor. This power of 2 factor may change every 8/16/32 elements being loaded into T. We can do this shifting ahead of time in software (easiest). Alternatively, we could alter the MAC engine so that it gets shifted as the raw values of `A` are being loaded using `setAMAC` instructions -- this is the purpose of the `rs2` specifier, which assumes we will always be shifting-right (if shift-left is needed, then it will have to be done in software, or we will need to extend the definition of the instruction to shift in both directions). You should be able to determine all of the `c_w` and `c_a` constants from training, then we can check their range to make sure they will fit the shift amounts of the `setAMAC` instruction and probably add support for shifting in either direction.

Alternatively, if we want to treat the `c_a` values as dynamic, we would want to compute them based on the values coming out of `T`. For this purpose, we may wish to have some hardware that tracks the maximum values as they are read out of `T`, then add an instruction to copy those values into a scalar CPU register. This requires only a small amount of extra hardware.


# APPENDIX MATERIAL

### References

Jerry's software for writing custom instructions can be found here:
https://github.com/JerryYun2004/RISC-V-RVV-Lite/tree/LUTRAM-VRF/sw/benchmarks

Jerry's hardware is here:
https://github.com/JerryYun2004/CVe2-RVV-Lite-A1-Extension

INT8 repository:
https://github.com/ruwayd99/RISC_V_Small_Integer_Accelerator_for_DNN_Inference



### GEN1: instructions for 8 * 8 tile only, computing MAC with two operands of 32b holding 8 * FP4 values

On an RV32 processor, most instructions combine two source integer registers `rs1` and `rs2` into a third destination `rd`.
We will produce an instruction to compute 64 MACs in parallel as a way of building the outer product in matrix multiply.

Accumulation will be done with 16b signed integers using saturating arithmetic (cannot exceed 32767, or go below -32768).

Multiply operations are done with 8 * FP4 values packed into 32b integers.

Compute operations:
- **zzMAC64** resets entire tile, `T[i][j] = 0` for all i x j in 0..7 x 0..7
- **maxMAC64 rs1** computes `T = max(T, rs1)` on all tile entries; `rs1` holds a signed 16-b integer (use `0` for ReLU)
- **hwMAC64 rs1, rs2** computes   `T[i][j] += rs1[i] * rs2[j]` for all i x j in 0..7 x 0..7, where `rs1` and `rs2` are 32b integer registers each holding 8 x FP4 values
- **addMAC64 rs1, rs2** computes saturating add using 16b integer from `rs2` to all activations in row in `rs1` specifier, `T[rs1][*] += rs2[15:0]`
- these instructions do not modify any integer registers
- for **addMAC64**, `rs1` is a 5b register specifier that does not indicate an integer register; instead, the 5 bits indicate row `i`

Move operations:
- moves from T to integer register file (or to vector register file, if it exists)
- **mvoMAC64 rd, rs1, rs2** moves single odd  tile entry `T[rs1][2*rs2+1]` indicated by 5b specifiers `rs1 and `rs2` (not register contents) to register `rd`, **clearing the tile entry to zero**
- **mveMAC64 rd, rs1, rs2** moves single even tile entry `T[rs1][2*rs2]`   indicated by 5b specifiers `rs1 and `rs2` (not register contents) to register `rd`, **clearing the tile entry to zero**
- **mv2MAC64 rd, rs1, rs2** moves concatenated **pair** of tile entries `{ T[rs1][2*rs2+1], T[rs1][2*rs2] }`, indicated by 5b specifiers `rs1 and `rs2` (not register contents) to  register `rd`, **clearing the tile entries to zero**
- `rd` is a destination in the integer register file
- `rs1` is a 5b register specifier that does not indicate an integer register; instead, the 5 bits indicate row `i`
- `rs2` is a 5b register specifier that does not indicate an integer register; instead, the 5 bits indicate column `j` as `2*rs2` and/or `2*rs2+1`

Memory operations:
- FLAWED: **ld2MAC64 rs2, IMM12(rs1)** reads 32b memory from effective address `rs1+IMM12`, writing to concatenated pair of tile entries `{T[i][2*j+1],T[i][2*j]}` indicated by field `rs2`
- FLAWED: **st2MAC64 rs2, IMM12(rs1)** writes 32b from concatenated **pair** of tile entries `{ T[i][2*j+1], T[i][2*j] }` indicated by field `rs2` to effective address `rs1+IMM12`, **clearing the tile entry to zero**
- the `rs2` field is a 5b register specifier that does not indicate an integer register; instead, the 2 LSB indicate `2*j` and the next 3 bits indicate `i`

The **zzMAC64**, **maxMAC64**, and **hwMAC64** instructions operate on all 64 tile entries.

The **addMAC64** instruction operates on an entire row, allowing a bias term to be added (with saturation).

In **hwMAC64**, the results of every FP4 * FP4 operation will be accumulated into a 16b integer. Each FP4 value itself can fit into an INT5, but not all INT5 values are used, making it a bad idea to get lazy and compute FP4 * FP4 in the INT5 * INT5 space. Instead, stay in the FP4 space and remove the sign bit so each multiplier operand is only 3b. This makes a product easy to compute using 6-input LUTs and summed into 16b integer accumulator. Each product spans the range from +/-0 to +/-144. A 16b accumulator can accumulate up to 227 of the largest products (144) before overflowing. This would be extremely rare, as there would normally be many small values and negative values as well. I think it is safe to assume up to 256 terms can be accumulated before overflow becomes a concern. To protect against overflow, make the accumulators **saturating** (wraparound would likely yield unpredictable results, but saturating at 32767 or -32768 should be OK).

Curiously, there are only 18 unique products out of the 64 combinations from multiplying two FP4 values: it consists of the FP4 valueset {0, 1, 2, 3, 4, 6, 8, 12}, but it also consists of {9, 16, 18, 24, 32, 36, 48, 64, 72, 96, 144 }. (Valuesets expressed as their integer equivalent when used in accumulation.)

### GEN2: with instructions to accept int16 values directly (converts to FP4 in hardware)

Compute tile up to 32 * 64:
- use Verilog parameter for tilesize is TS = 8 (8 is the default value)
- actual tile dimensions are rows=TS, columns=2*TS, up to TS=32
- maximum dimensions are T[32][64], for a maximum size of 2048 elements

MAC-oriented compute operations:
- **zzMAC** resets entire tile, `T[i][j] = 0` for all i x j in 0 .. TS-1 x 0 .. 2 * TS-1
- **maxMAC rs1** computes `T = max(T, rs1)` on all tile entries; `rs1` holds a signed 16-b integer (use `0` for ReLU; it's actually an int32 but kept in range of int16)
- **hwMAC** computes   `T[i][j] += W[i] * A[j]` for all i x j in 0 .. TS-1 x 0 .. 2 * TS-1, where `W` and `A` are the outer product registers each holding TS or 2 * TS FP4 values
- **setWMAC rd, rs1** sets `W[rd] = rs1`, where rs1 holds 8 * FP4 values; there will be at most 8 * 8 = 64 FP4 values
- **setAMAC rd, rs1, rs2** sets `{ A[2*rd], A[2*rd+1] } = { rs1[15:0]>>>rs2, rs1[31:16]>>>rs2 }`, where the `rd` specifier is in the range 0-31 and `rs1` holds 2 * INT16 values, for a maximum total of 64 positions in `A`, and `rs2` is a 5b specifier for the arithmetic shift-right amount in the range of 0-10. These INT16 values will be immediately converted to and stored as FP4 in hardware registers. The shifting (INT16 >> 10) yields an int6 value (at most 6 bits), in fixed-point Q4.2 format ranging from -8.0 to +7.75, to be converted into FP4 in the range of -6 to 6. The shifting may or may not round or jam the LSBs (truncate for now; we can talk about this rounding/jamming later.)

Matrix update operations (much faster than vector unit):
- **addMAC rs1, rs2** computes saturating add using 16b integer from `rs2` to all activations in the row indicated by the `rs1` specifier, `T[rs1][*] += rs2[15:0]`
- **maxMAC rs1** computes `T = max(T, rs1)` on all tile entries; `rs1` holds 32b value treated as INT16 (use `0` for ReLU)
- for **addMAC**, `rs1` is a 5b register specifier that does not indicate an integer register; instead, the 5 bits indicate row `i`
- IGNORE FOR NOW **setDMAC rs1, rs2** sets dimensions for `W` in rs1 and `A` in `rs2`, MACs outside of these dimensions are idle (only for power savings)

Memory operations:
- **st2MAC rs1, IMM12(rs2)** writes 32b from concatenated **pair** of tile entries `{ T[rs1][IMM12[6:1]+1], T[rs1][IMM12[6:1]] }` indicated by 5b specifier `rs1` (not register contents) and 6b specifier `IMM12[6:1]` to effective address `rs2+IMM12`, **clearing the tile entry to zero**
- `rs1` is a 5b register specifier that does not indicate an integer register; instead, the 5 bits indicate row `i`
- `IMM12` is an address offset, but it also indicates which pair of column values to take from `T`. Hence, the `F` matrix rows must be 128-byte aligned
- the **st2MAC** instruction is essential to write out data after Matrix update operations

Move operations:
- moves from T to integer register file
- maximum dimensions are T[32][64], for a maximum MAC size of 2048 elements
- these moves are needed if further post-processing (other than adding bias and applying ReLU) is required
- **mvoMAC rd, rs1, rs2** moves single odd  tile entry `T[rs1][2*rs2+1]` indicated by 5b specifiers `rs1 and `rs2` (not register contents) to register `rd`, **clearing the tile entry to zero**
- **mveMAC rd, rs1, rs2** moves single even tile entry `T[rs1][2*rs2]`   indicated by 5b specifiers `rs1 and `rs2` (not register contents) to register `rd`, **clearing the tile entry to zero**
- **mv2MAC rd, rs1, rs2** moves concatenated **pair** of tile entries `{ T[rs1][2*rs2+1], T[rs1][2*rs2] }`, indicated by 5b specifiers `rs1 and `rs2` (not register contents) to  register `rd`, **clearing the tile entries to zero**
- `rs1` is a 5b register specifier that does not indicate an integer register; instead, the 5 bits indicate row `i`
- `rs2` is a 5b register specifier that does not indicate an integer register; instead, the 5 bits indicate column `j` as `2*rs2` and/or `2*rs2+1`
- `rd` is a destination in the integer register file

Vector operations (OPTIONAL):
- **mvvMAC vd, rs1** single instruction to copy an entire row, `v[rd] = T[rs1][*]`
- row length determined by vector length register using `setvl` instructions
- each vector register needs to hold `VLEN=2*TS*16`, or up to 1024 to hold up to 64 x 16b halfwords

- 

## Hardware MAC64 Instruction Set for Convolutions with FP4 (may be OBSOLETE -- based on GEN1 MAC64 design)

In an effort to add further novelty to the design, I'm considering adding instructions to assist with accelerating convolutional neural networks.

This convolution accelerator should work for 1 * 1, 3 * 3, 5 * 5 and 7 * 7 kernels. It requires the same `MAC64' array used for matrix multiply, consisting of FP4 multipliers and int16 accumulators, plus some extra logic to hold 8 * 8 tile of image pixels in FP4, called `A`, and the ability to rotate the pixels in that image tile them up/down/left/right.

The 8 * 8 input tile of `A` are FP4 registers that are preloaded with the image pixels forming one operand of the multiplication. The second operand comes from broadcasting a single FP4 value from a 32b integer register. To efficiently iterate over multiple filter coefficients, an immediate constant 0..7 will determine one of 8 FP4 values to extract from the second operand. Also, rather than moving the location of the filter, the image pixels will rotate by 1 column for one instruction, or 1 row for another instruction; a snaking path will be taken to cover the entire 2D window of the filter. The instruction set and code below shows how this works.

Move operations:
- **mvA IMM2, rs1, rs2** loads two rows of pixels in the 8 * 8 `A` tile, where the `IMM2` specifier indicates a value for `r` from 0..3. The values of `rs1` and `rs2` each hold 8 FP4 values for rows `2*r` and `2*r+1` in `A`.
- - **conv rs1, IMM3** computes `T += f(rs1) * A`, where `T` is the 8 * 8 accumulator tile, `rs1` holds 8 FP4 values, and IMM3 indicates which FP4 value to extract from `rs1`.
- **convLC rs1, IMM3** like **conv**, but this also concurrently computes `A = rotlc(A)`, where `rs1` holds 8 FP4 values, and IMM3 indicates which FP4 value to extract from `rs1`. The `rotlc()` function moves all FP4 values in A to the left by 1 column, with the leftmost pixels moving to the rightmost column.
- **convRC rs1, IMM3** similar to **convLC** but computes `A = rotrc(A)` which rotates A to the right by 1 column, with the rightmost column moving to the leftmost position.
- **convUR rs1, IMM3** similar to **convLC** but computes `A = rotur(A)` which rotates A up by 1 row, with the topmost column moving to the bottom-most position. 
- **convDR rs1, IMM3** similar to **convLC** but computes `A = rotdr(A)` which rotates A down by 1 row, with the bottom-most column moving to the topmost position.
- instructions needed from the outer product accelerator include t**zzMAC64** and **st2MAC64**

The idea is to use these instructions to snake through all positions within the K * K filter kernel. For example, this sequence of instructions convolves `A` with a 3 x 3 filter kernel:
```
mvA 0, x1, x2 // pre-load 8 rows of tile A with 4 instructions
mvA 1, x3, x4
mvA 2, x5, x6
mvA 3, x7, x8
// do a 3x3 convolution with the first 8 filter values from x9, and the 9th from x10
zzMA64  // clear all 8 * 8 entries of T, erase all inactive rows/columns
convLC  x9, 0 // convolve A with  x9[ 3: 0], A = rotlc(A) moves A left   (after: 1 inactive column on right)
convLC  x9, 1 // convolve A with  x9[ 7: 4], A = rotlc(A) moves A left   (after: 2 inactive columns on right)
convUR  x9, 2 // convolve A with  x9[11: 8], A = rotur(A) moves A up     (after: 2 inactive columns on right, 1 inactive row on bottom)
convRC  x9, 3 // convolve A with  x9[15:12], A = rotrc(A) moves A right  (after: 1 inactive columns on right, 1 inactive row on bottom)
convRC  x9, 4 // convolve A with  x9[19:16], A = rotrc(A) moves A right  (after: 0 inactive columns on right, 1 inactive row on bottom)
convUR  x9, 5 // convolve A with  x9[23:20], A = rotur(A) moves A up     (after: 0 inactive columns on right, 2 inactive rows on bottom)
convLC  x9, 6 // convolve A with  x9[27:24], A = rotrc(A) moves A left   (after: 1 inactive columns on right, 2 inactive rows on bottom)
convLC  x9, 7 // convolve A with  x9[31:28], A = rotrc(A) moves A left   (after: 2 inactive columns on right, 2 inactive rows on bottom)
conv   x10, 0 // convolve A with x10[ 3: 0]                              (after: 2 inactive columns on right, 2 inactive rows on bottom)
// write out T which holds the convolution results
// 3 * 3 tile produces 6 * 6 results (results outside of this region are gibberish)
// 5 * 5 tile produces 4 * 4 results (results outside of this region are gibberish)
// 7 * 7 tile produces 2 * 2 results (results outside of this region are gibberish)
// these results must be saved to memory using st2MAC64 instructions
// or they can be moved to integer registers using mvoMAC64, mveMAC64, mv2MAC64 instructions

// one 3 * 3 kernel requires: 4 x mvA, 1 x zzMAC64, 18 x st2MAC64, plus 9 conv* instructions = 32 instructions total
// in contrast, one 3 * 3 kernel would require 18 * 6 * 6 = 32 * 81 individual multiply or add instructions, plus 36 store instructions and many (repeated) loads and loop overhead

// OPTIONAL THOUGHT:
// if re-using the image tile with another convolution kernel (different 3x3 filter), it is possible to work backwards from the above
// clear T, start next convolution while restoring A to proper position by snaking the reverse of the above. does this make sense?
zzMA64  // clear all 8 * 8 entries of T
mvA // load new filter kernel
convRC x10, 0 // convolve A with x10[ 3: 0], A = rotdr(A) moves A down  (after: 1 inactive columns on right, 2 inactive rows on bottom)
convRC  x9, 7 // convolve A with  x9[31:28], A = rotrc(A) moves A left  (after: 0 inactive columns on right, 2 inactive rows on bottom)
convDR  x9, 6 // convolve A with  x9[27:24], A = rotrc(A) moves A left  (after: 0 inactive columns on right, 1 inactive rows on bottom)
convLC  x9, 5 // convolve A with  x9[23:20], A = rotur(A) moves A up    (after: 1 inactive columns on right, 1 inactive rows on bottom)
convLC  x9, 4 // convolve A with  x9[19:16], A = rotrc(A) moves A right (after: 2 inactive columns on right, 1 inactive row on bottom)
convDR  x9, 3 // convolve A with  x9[15:12], A = rotrc(A) moves A right (after: 2 inactive columns on right, 0 inactive row on bottom)
convRC  x9, 2 // convolve A with  x9[11: 8], A = rotur(A) moves A up    (after: 1 inactive columns on right, 0 inactive row on bottom)
convRC  x9, 1 // convolve A with  x9[ 7: 4], A = rotlc(A) moves A left  (after: all rows/columns active)
conv    x9, 0 // convolve A with  x9[ 3: 0]                             (after: all rows/columns active)
// now write out tile T as the solution
```

The extra hardware required to support these instructions is:
- an 8 * 8 array of FP4 values
- the ability to shift values in the 8 * 8 array in 4 directions (4:1 mux), from U/D/L/R positions
- the ability to extract a single FP4 value from a 32 operand using IMM3, and broadcast this to all 8 * 8 multipliers

To see an example of this being computed in C, please look at the [examples/conv.c](examples/conv.c) source code.
