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


# Hardware MAC64 Instruction Set for Outer Product with FP4

On an RV32 processor, most instructions combine two source integer registers `rs1` and `rs2` into a third destination `rd`.
We will produce an instruction to compute 64 MACs in parallel as a way of building the outer product in matrix multiply.

Accumulation will be done with 16b signed integers using saturating arithmetic (cannot exceed 32767, or go below -32768).

Multiply operations are done with 8 * FP4 values packed into 32b integers.

Compute operations:
- **zzMAC64** resets entire tile, `T[i][j] = 0` for all i x j in 0..7 x 0..7
- **maxMAC64 rs1** computes `T = max(T, rs1)` on all tile entries; `rs1` holds a signed 16-b integer (use `0` for ReLU)
- **hwMAC64 rs1, rs2** computes   `T[i][j] += rs1[i] * rs2[j]` for all i x j in 0..7 x 0..7, where `rs1` and `rs2` are 32b integer registers each holding 8 x FP4 values
- **ad2MAC64 rs1, rs2** computes saturating add using **pair** of 16b integers from `rs1`, `T[i][2*j] += rs1[15:0]` and `T[i][2*j+1] += rs1[31:16]`
- these instructions do not modify any integer registers
- for **ad2MAC64**, `rs2` is a 5b register specifier that does not indicate an integer register; instead, the 2 LSB indicate `2*j` and the next 3 bits indicate `i`

Move operations:
- always moves from T to integer register file
- **mvoMAC64 rd,  rs2** moves single odd  tile entry `T[i][2*j+1]` indicated by `rs2` to integer register `rd`, **clearing the tile entry to zero**
- **mveMAC64 rd,  rs2** moves single even tile entry `T[i][2*j]`   indicated by `rs2` to integer register `rd`, **clearing the tile entry to zero**
- **mv2MAC64 rd,  rs2** moves concatenated **pair** of tile entries `{ T[i][2*j+1], T[i][2*j] }`, where `i` and `j` are taken from the `rs2` specifier (not register contents), to integer register `rd`, **clearing the tile entries to zero**
- `rd` is a destination in the integer register file
- `rs2` is a 5b register specifier that does not indicate an integer register; instead, the 2 LSB indicate `2*j` and the next 3 bits indicate `i`

Memory operations:
- **ld2MAC64 rs2, IMM12(rs1)** reads 32b memory from effective address `rs1+IMM12`, writing to concatenated pair of tile entries `{T[i][2*j+1],T[i][2*j]}` indicated by field `rs2`
- **st2MAC64 rs2, IMM12(rs1)** writes 32b from concatenated **pair** of tile entries `{ T[i][2*j+1], T[i][2*j] }` indicated by field `rs2` to effective address `rs1+IMM12`, **clearing the tile entry to zero**
- the `rs2` field is a 5b register specifier that does not indicate an integer register; instead, the 2 LSB indicate `2*j` and the next 3 bits indicate `i`

The **zzMAC64**, **maxMAC64**, and **hwMAC64** instructions operate on all 64 tile entries.

The **ad2MAC64** instruction operates on two tile entries, allowing 2 bias terms to be added (with saturation).

In **hwMAC64**, the results of every FP4 * FP4 operation will be accumulated into a 16b integer. Each FP4 value itself can fit into an INT5, but not all INT5 values are used, making it a bad idea to get lazy and compute FP4 * FP4 in the INT5 * INT5 space. Instead, stay in the FP4 space and remove the sign bit so each multiplier operand is only 3b. This makes a product easy to compute using 6-input LUTs and summed into 16b integer accumulator. Each product spans the range from +/-0 to +/-144. A 16b accumulator can accumulate up to 227 of the largest products (144) before overflowing. This would be extremely rare, as there would normally be many small values and negative values as well. I think it is safe to assume up to 256 terms can be accumulated before overflow becomes a concern. To protect against overflow, make the accumulators **saturating** (wraparound would likely yield unpredictable results, but saturating at 32767 or -32768 should be OK).

Curiously, there are only 18 unique products out of the 64 combinations from multiplying two FP4 values: it consists of the FP4 valueset {0, 1, 2, 3, 4, 6, 8, 12}, but it also consists of {9, 16, 18, 24, 32, 36, 48, 64, 72, 96, 144 }. (Valuesets expressed as their integer equivalent when used in accumulation.)



# Hardware MAC64 Instruction Set for Convolutions with FP4

In an effort to add further novelty to the design, I'm considering adding instructions to assist with accelerating convolutional neural networks.

This convolution accelerator should work for 1 * 1, 3 * 3, 5 * 5 and 7 * 7 kernels. It requires the same `MAC64' array used for matrix multiply, consisting of FP4 multipliers and int16 accumulators, plus some extra logic to hold 8 * 8 tile of image pixels in FP4, called `A`, and the ability to rotate the pixels in that image tile them up/down/left/right.

The 8 * 8 input tile of `A` are FP4 registers that are preloaded with the image pixels forming one operand of the multiplication. The second operand comes from broadcasting a single FP4 value from a 32b integer register. To efficiently iterate over multiple filter coefficients, an immediate constant 0..7 will determine one of 8 FP4 values to extract from the second operand. Also, rather than moving the location of the filter, the image pixels will rotate by 1 column for one instruction, or 1 row for another instruction; a snaking path will be taken to cover the entire 2D window of the filter. The instruction set and code below shows how this works.

Move operations:
- **mvA IMM2, rs1, rs2** loads two rows of pixels in the 8 * 8 `A` tile, where the `IMM2` specifier indicates a value for `r` from 0..3. The values of `rs1` and `rs2` each hold 8 FP4 values for rows `2*r` and `2*r+1` in `A`.
- - **conv rs1, IMM3** computes `T += f(rs1) * A`, where `T` is the 8 * 8 accumulator tile, `rs1` holds 8 FP4 values, and IMM3 indicates which FP4 value to extract from `rs1`. The rightmost/bottom-most columns/rows of `A` that are considered **inactive** do not update `T` (i.e., are treated as zeros).
- **convLC rs1, IMM3** like **conv**, but this also concurrently computes `A = rotlc(A)`, where `rs1` holds 8 FP4 values, and IMM3 indicates which FP4 value to extract from `rs1`. The `rotlc()` function moves all FP4 values in A to the left by 1 column, with the leftmost pixels moving to the rightmost column. An **inactive** region at the far right grows by 1 column. Inactive pixels are treated as 0 in future instruction invocations.
- **convRC rs1, IMM3** similar to **convLC** but computes `A = rotrc(A)` which rotates A to the right by 1 column, with the rightmost column moving to the leftmost position. The **inactive** region at the far right shrinks by 1 column.
- **convUR rs1, IMM3** similar to **convLC** but computes `A = rotur(A)` which rotates A up by 1 row, with the topmost column moving to the bottom-most position. The **inactive** region at the bottom grows by 1 row.
- **convDR rs1, IMM3** similar to **convLC** but computes `A = rotdr(A)` which rotates A down by 1 row, with the bottom-most column moving to the topmost position. The **inactive** region at the bottom shrinks by 1 row.
- as above, the **zzMAC64** instruction is needed. This instruction should make all rows/columns of **active** to start the convolution correctly.

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
// using up to 64 mvoMAC64, mveMAC64 instructions
// or up to 32 mv2MAC64, or st2MAC64 instructions
// NOTE: fewer instructions are needed if the last few rows/columns of T can be skipped
//       e.g., only 6 * 6 results are valid, requiring 18 st2MAC64 instructions, when computing a 3 * 3 filter on an 8 * 8 tile

// 1 kernel requires: 4 x mvA, 1 x zzMAC64, 18 x st2MAC64, plus 9 conv* instructions

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
- 14 flip-flops that track up to 7 inactive rows and 7 inactive columns, and prevent those rows/columns from participating in the tile update (either force the multiplicand to 0, or prohibit writing to the relevant accumulators; latter probably uses less logic); these flip-flops must be reset on a **zzMAC64** instruction.

To see an example of this being computed in C, please look at the [examples/conv.c](examples/conv.c) source code.


# Tiled Matrix Multiply

## WARNING: TEXT BELOW HAS NOT BEEN CAREFULLY EDITTED. IT WILL PROBABLY BE HEAVILY REWRITTEN.

The naive version of matrix multiply uses three nested loops, in order of `i`, `j`, and `k`, where the innermost `k` loop is used to compute a dot product, also known as the **inner product**, and the `i` loop traverses rows of `A` and `C` while the `j` loop traverses columns of `B` and `C`. The core computation is `C[i][j] += A[i][k] * B[k][j]`:
```
for i = 0 to 1023
  for j = 0 to 1023
    for k = 0 to 1023
      C[i][j] += A[i][k] * B[k][j]
```
To avoid reading/writing memory for `C` each time, we can do this:
```
for i = 0 to 1023
  for j = 0 to 1023
    t = 0
    for k = 0 to 1023
      t += A[i][k] * B[k][j]
    C[i][j] += t
```
This `t` term stays in a register, acting a bit like a cache. Later, we will transform `t` into `T`, an entire 8 x 8 tile of temporary values, and operate on all of them in parallel with the **hwMAC64** instruction. The last line, `C[i][j] += t` ensures new results are added to any initial value in the matrix, just like the original algorithm which also added to the initial value of `C`.

While the **inner product** works well to cache the value of `t`, it can only compute one output element at a time (the dot product). For example, suppose we use sub-word SIMD and group eight FP4 values from `A` into `rs1` and eight FP4 values from `B` into `rs2`. An inner product between A and B would compute eight multiplications and seven additions; once added to the original value of `C[i][j]`, an 8th addition appears. In other words, this inner product computes at most 8 multiply-accumulate operations (MACs).

However, an **outer product** can compute even more MACs in parallel, as long as it has enough storage to hold the intermediate sums (several values of `t`).
 For this, it needs a tile of values `T` which will be written (?added?) into `C` one tile at a time. To start this , we transform the algorithm loop order so the `k` loop is now the outermost loop, and the `i` and `j` loops are the innermost loops:
```
for k = 0 to 1023
  for i = 0 to 1023
    for j = 0 to 1023
      C[i][j] += A[i][k] * B[k][j]
```
For large matrices like these, we must break this down into smaller 8 * 8 tiles:
```
for I = 0 to 1023 step 8
  for J = 0 to 1023 step 8
    T = 8*8 tile starting at C[i][j]
    for k = 0 to 1023
      for i = 0 to 7 // UNROLL
        for j = 0 to 7 // UNROLL
          T[i][j] += A[I+i][k] * B[k][J+j]
    C[i][j] = T // copy entire tile back
```
Here, you'll see the innermost `i` and `j` loops have just 8 iterations. This is the size of the tile `T`, and it covers just an 8 * 8 patch of the larger 1024 * 1024 `C` matrix. These two innermost loops can be fully unrolled, yielding a total of 64 MAC operations taken from the core computation, `T[i][j] += A[I+i][k] * B[k][J+j]`. This core computation uses a strip of the `A` matrix that is 8 rows tall, and a strip of the `B` matrix that is 8 columns wide; both strips are of length 1024, which is iterated over by the `k` dimension (the middle loop). These 3 inner loops only compute the results for one tile, so outer loops are added for `I` and `J` to iterate over all possible 8 * 8 tiles within `C`; those outer loops go in increments of 8, of course, to step along by one tile at a time. The code now looks like this:
```
for I = 0 to 1023 step 8
  for J = 0 to 1023 step 8
    T = 8*8 tile starting at C[i][j]
    for k = 0 to 1023
      T[0][0] += A[I+0][k] * B[k][J+0]
      T[0][1] += A[I+0][k] * B[k][J+1]
      T[0][2] += A[I+0][k] * B[k][J+2]
      ...
      T[7][5] += A[I+7][k] * B[k][J+5]
      T[7][6] += A[I+7][k] * B[k][J+6]
      T[7][7] += A[I+7][k] * B[k][J+7]
    C[i][j] = T // copy entire 8*8 tile back
```
Rewriting this into the ISA:
```
for I = 0 to 1023 step 8
  for J = 0 to 1023 step 8
    zzMAC64; // clears entire tile; alternatively, could load tile from `C`
    for k = 0 to 1023
      AA = A[I+0][k] to A[I+7][k] // reads 8 elements from a column
      BB = B[k][J+0] to B[k][J+7] // reads 8 elements from a row
      hwMAC64 AA, BB
    maxMAC64 x0 // computes ReLU on entire tile
    stMAC64 r0, 0(rs1) // stores tile, 2 elements at a time; also clears tile entries
    stMAC64 r1, 4(rs1)
    stMAC64 r2, 8(rs1)
    stMAC64 r3, 12(rs1)
    ...
    stMAC64 r31, 124(rs1)
```
where `r0` to `r31` represent the contents of the tile `{ T[i][2*j], T[i][2*j+1] }`, that is `rX` where `X=4*i+j`.

The only problem with this code is the k dimension is 1024. Any value over 227 can potentially overflow the 16b integers. In that case, the tile `T` could be read out, one 16b entry at a time, converted to 32b, and added to a 32b integer matrix. This is one use-case for the **mvoMAC64** and **mveMAC64** instructions.

Overall, I see code looking something like this:

```
for I = 0 to 1023 step 8 begin
  for J = 0 to 1023 step 8 begin

    // the i/j loops below can be fully unrolled
    for i = 0 to 7
      for j = 0 to 7
        T[i][j] = 0 or T[i][j] = C[I+i][J+j]; // using **zzMAC64** or **ld2MAC64** instructions

    for K = 0 to 1023 begin // CAREFUL: may cause saturation of 16b values
      // the i/j loops below can be fully unrolled
      for i = 0 to 7
        for j = 0 to 7
          T[i][j] += A[I+i][K] * B[K][J+j] // using **hwMAC64**
    end // K

// EITHER: write out tile into the matrix
    // this needs a "store C[i][j]" instruction
    for i = 0 to 7
      for j = 0 to 7
        C[I+i][J+j] = T[i][j] // using **st2MAC64** intructions
// OR: immediately apply bias and activation
    // this needs a "rd = T[i][j]" instruction
    for i = 0 to 7
      for j = 0 to 7 begin
        r = max( T[i][j] + bias[I+i][J+j], 0 ) // integer instructions, including **mvoMAC64** and **mveMAC64**, and may be **maxMAC64**
        C[I+i][J+j] = r // regular store instruction
      end // j

  end // J
end // I
```

