    .section .text
    .globl matmul8_vec
    .align 2

#define OPV      0x57
#define LOADFP   0x07
#define STOREFP  0x27

/* vsetvli rd, rs1, e32,m1,ta,ma  -> vtypei = 0x0D0 */
#define RVV_VSETVLI(rd, rs1, vtypei) \
  .word (((vtypei) << 20) | ((rs1) << 15) | (7 << 12) | ((rd) << 7) | OPV)

/* vle32.v vd, (rs1) */
#define RVV_VLE32(vd, rs1) \
  .word ((1 << 25) | ((rs1) << 15) | (6 << 12) | ((vd) << 7) | LOADFP)

/* vse32.v vs3, (rs1) */
#define RVV_VSE32(vs3, rs1) \
  .word ((1 << 25) | ((rs1) << 15) | (6 << 12) | ((vs3) << 7) | STOREFP)

#define F3_OPIVV 0
#define F3_OPIVX 4
#define F6_VADD  0x00
#define F6_VMUL  0x25

/* opivv: vd = vs2 <op> vs1  (for vadd.vv, order does not matter) */
#define RVV_OPIVV(f6, vd, vs1, vs2) \
  .word (((f6) << 26) | (1 << 25) | ((vs2) << 20) | ((vs1) << 15) | (F3_OPIVV << 12) | ((vd) << 7) | OPV)

/* opivx: vd = vs2 <op> rs1 */
#define RVV_OPIVX(f6, vd, rs1, vs2) \
  .word (((f6) << 26) | (1 << 25) | ((vs2) << 20) | ((rs1) << 15) | (F3_OPIVX << 12) | ((vd) << 7) | OPV)

//[stev]
#define CUSTOM2 0x5b

#define ZZMAC64 \
.word (((0x00)<<25) | (0<<20) | (0<<15) | (0<<12) | (0<<7) | CUSTOM2)

#define HWMAC64(rs1,rs2) \
.word (((0x02)<<25) | ((rs2)<<20) | ((rs1)<<15) | (0<<12) | (0<<7) | CUSTOM2)

// -----------------------------------------------------------------------------
// Tile move instructions
// rs1 = row specifier (instruction field)
// rs2 = pair specifier (instruction field)
// rd  = integer destination register
// -----------------------------------------------------------------------------

#define MVEMAC64(rd,row,pair) \
.word (((0x05)<<25) | ((pair)<<20) | ((row)<<15) | (0<<12) | ((rd)<<7) | CUSTOM2)

#define MVOMAC64(rd,row,pair) \
.word (((0x04)<<25) | ((pair)<<20) | ((row)<<15) | (0<<12) | ((rd)<<7) | CUSTOM2)

#define MV2MAC64(rd,row,pair) \
.word (((0x06)<<25) | ((pair)<<20) | ((row)<<15) | (0<<12) | ((rd)<<7) | CUSTOM2)

#define MAXMAC64(rs1) \
.word (((0x01)<<25) | (0<<20) | ((rs1)<<15) | (0<<12) | (0<<7) | CUSTOM2)


#define ADDMAC64(row, rs2) \
.word (((0x03)<<25) | ((rs2)<<20) | ((row)<<15) | (0<<12) | (0<<7) | CUSTOM2)

#define LD2MAC64(tile, base, imm5) \
.word (((0x07)<<25) | \
       ((tile)<<20) | \
       ((base)<<15) | \
       (0<<12)      | \
       (((imm5)&0x1f)<<7) | \
       CUSTOM2)

#define ST2MAC64(tile, base, imm5) \
.word (((0x08)<<25) | \
       ((tile)<<20) | \
       ((base)<<15) | \
       (0<<12)      | \
       (((imm5)&0x1f)<<7) | \
       CUSTOM2)

.globl mac_zz
mac_zz:
    ZZMAC64
    ret

.globl mac_hw
mac_hw:
    HWMAC64(10,11)
    ret

.globl mac_out_even
mac_out_even:
    MVEMAC64(10,0,0)      # returns T[0][0] in a0
    ret

.globl mac_out_odd
mac_out_odd:
    MVOMAC64(10,0,0)      # returns T[0][1] in a0
    ret

.globl mac_out_pair
mac_out_pair:
    MV2MAC64(10,0,0)      # returns {T[0][1],T[0][0]} in a0
    ret


.globl mac_max
mac_max:
    MAXMAC64(10)      # uses a0 (x10)
    ret

.globl mac_add_row
mac_add_row:
    ADDMAC64(3, 10)   # row=3, value in a0
    ret

.globl mac_ld2
mac_ld2:
    # a0 (x10) = base address
    LD2MAC64(0, 10, 0)      # tile=(row=0,pair=0), base=x10, imm=0
    ret

.globl mac_st2
mac_st2:
    # a0 (x10) = base address
    ST2MAC64(0, 10, 0)      # tile=(row=0,pair=0), base=x10, imm=0
    ret

/*
 * mac_add_bias(bias=a0)
 *
 * Adds the layer bias to the tile: T[i][all] += bias[i]<<2, for rows i=0..7.
 */
.globl mac_add_bias
mac_add_bias:
    lh   t0,  0(a0)
    slli t0, t0, 2
    ADDMAC64(0, 5)         # T[0][all] += t0
    lh   t0,  2(a0)
    slli t0, t0, 2
    ADDMAC64(1, 5)
    lh   t0,  4(a0)
    slli t0, t0, 2
    ADDMAC64(2, 5)
    lh   t0,  6(a0)
    slli t0, t0, 2
    ADDMAC64(3, 5)
    lh   t0,  8(a0)
    slli t0, t0, 2
    ADDMAC64(4, 5)
    lh   t0, 10(a0)
    slli t0, t0, 2
    ADDMAC64(5, 5)
    lh   t0, 12(a0)
    slli t0, t0, 2
    ADDMAC64(6, 5)
    lh   t0, 14(a0)
    slli t0, t0, 2
    ADDMAC64(7, 5)
    ret

/*
 * mac_st2_readback(out=a0)
 *
 * Reads the 8x8 INT16 tile into out (64 entries, row-major), clearing the tile.
 */
.globl mac_st2_readback
mac_st2_readback:
    MV2MAC64(5, 0, 0)      # {T[0][1],T[0][0]}
    sw   t0,   0(a0)
    MV2MAC64(5, 0, 1)      # {T[0][3],T[0][2]}
    sw   t0,   4(a0)
    MV2MAC64(5, 0, 2)      # {T[0][5],T[0][4]}
    sw   t0,   8(a0)
    MV2MAC64(5, 0, 3)      # {T[0][7],T[0][6]}
    sw   t0,  12(a0)

    MV2MAC64(5, 1, 0)      # {T[1][1],T[1][0]}
    sw   t0,  16(a0)
    MV2MAC64(5, 1, 1)
    sw   t0,  20(a0)
    MV2MAC64(5, 1, 2)
    sw   t0,  24(a0)
    MV2MAC64(5, 1, 3)
    sw   t0,  28(a0)

    MV2MAC64(5, 2, 0)      # {T[2][1],T[2][0]}
    sw   t0,  32(a0)
    MV2MAC64(5, 2, 1)
    sw   t0,  36(a0)
    MV2MAC64(5, 2, 2)
    sw   t0,  40(a0)
    MV2MAC64(5, 2, 3)
    sw   t0,  44(a0)

    MV2MAC64(5, 3, 0)      # {T[3][1],T[3][0]}
    sw   t0,  48(a0)
    MV2MAC64(5, 3, 1)
    sw   t0,  52(a0)
    MV2MAC64(5, 3, 2)
    sw   t0,  56(a0)
    MV2MAC64(5, 3, 3)
    sw   t0,  60(a0)

    MV2MAC64(5, 4, 0)      # {T[4][1],T[4][0]}
    sw   t0,  64(a0)
    MV2MAC64(5, 4, 1)
    sw   t0,  68(a0)
    MV2MAC64(5, 4, 2)
    sw   t0,  72(a0)
    MV2MAC64(5, 4, 3)
    sw   t0,  76(a0)

    MV2MAC64(5, 5, 0)      # {T[5][1],T[5][0]}
    sw   t0,  80(a0)
    MV2MAC64(5, 5, 1)
    sw   t0,  84(a0)
    MV2MAC64(5, 5, 2)
    sw   t0,  88(a0)
    MV2MAC64(5, 5, 3)
    sw   t0,  92(a0)

    MV2MAC64(5, 6, 0)      # {T[6][1],T[6][0]}
    sw   t0,  96(a0)
    MV2MAC64(5, 6, 1)
    sw   t0, 100(a0)
    MV2MAC64(5, 6, 2)
    sw   t0, 104(a0)
    MV2MAC64(5, 6, 3)
    sw   t0, 108(a0)

    MV2MAC64(5, 7, 0)      # {T[7][1],T[7][0]}
    sw   t0, 112(a0)
    MV2MAC64(5, 7, 1)
    sw   t0, 116(a0)
    MV2MAC64(5, 7, 2)
    sw   t0, 120(a0)
    MV2MAC64(5, 7, 3)      # {T[7][7],T[7][6]}
    sw   t0, 124(a0)
    ret

/*
 * matmul8_vec(a=a0, b=a1, c=a2, tmp=a3)
 *
 * A   is 8x8 row-major.
 * B   is 8x8 row-major.
 * C   is 8x8 row-major and is preloaded to zero by the testbench.
 * tmp is unused here, kept only to match the C prototype.
 *
 * Mathematical workload matches the scalar benchmark:
 *   C[i,j] = C[i,j] + sum_k A[i,k] * B[k,j]
 *
 * Implementation:
 *   preload all 8 rows of B into v0..v7 once,
 *   then for each output row i:
 *     acc = C[i,*]
 *     acc += A[i,0] * B[0,*]
 *     ...
 *     acc += A[i,7] * B[7,*]
 *     store acc to C[i,*]
 *
 * The vector unit is expected to handle vector instruction dependencies
 * through its busy/done handshake and synchronous-VRF read state.
 */

matmul8_vec:
    addi sp, sp, -16
    sw   ra, 12(sp)
    sw   s0,  8(sp)
    sw   s1,  4(sp)
    sw   s2,  0(sp)

    mv   s0, a0              /* base A  */
    mv   s1, a1              /* base B  */
    mv   s2, a2              /* base C  */

    li   t0, 8
    RVV_VSETVLI(5, 5, 0x0D0) /* t0=x5 -> vl=8, e32,m1,ta,ma */

    /* Load all 8 rows of B one time: v0..v7 */
    RVV_VLE32(0, 9)          /* v0 <- B[0,*], rs1=s1=x9 */
    addi s1, s1, 32
    RVV_VLE32(1, 9)          /* v1 <- B[1,*] */
    addi s1, s1, 32
    RVV_VLE32(2, 9)          /* v2 <- B[2,*] */
    addi s1, s1, 32
    RVV_VLE32(3, 9)          /* v3 <- B[3,*] */
    addi s1, s1, 32
    RVV_VLE32(4, 9)          /* v4 <- B[4,*] */
    addi s1, s1, 32
    RVV_VLE32(5, 9)          /* v5 <- B[5,*] */
    addi s1, s1, 32
    RVV_VLE32(6, 9)          /* v6 <- B[6,*] */
    addi s1, s1, 32
    RVV_VLE32(7, 9)          /* v7 <- B[7,*] */

    li   t0, 8               /* outer loop count: 8 rows */

1:
    /* Load current C row into accumulator v8. C is preloaded to zero. */
    RVV_VLE32(8, 18)         /* v8 <- C[i,*], rs1=s2=x18 */

    lw      t1,  0(s0)
    RVV_OPIVX(F6_VMUL, 9, 6, 0)  /* v9 = v0 * t1, t1=x6 */
    RVV_OPIVV(F6_VADD, 8, 9, 8)  /* v8 = v8 + v9 */

    lw      t1,  4(s0)
    RVV_OPIVX(F6_VMUL, 9, 6, 1)  /* v9 = v1 * t1 */
    RVV_OPIVV(F6_VADD, 8, 9, 8)

    lw      t1,  8(s0)
    RVV_OPIVX(F6_VMUL, 9, 6, 2)  /* v9 = v2 * t1 */
    RVV_OPIVV(F6_VADD, 8, 9, 8)

    lw      t1, 12(s0)
    RVV_OPIVX(F6_VMUL, 9, 6, 3)  /* v9 = v3 * t1 */
    RVV_OPIVV(F6_VADD, 8, 9, 8)

    lw      t1, 16(s0)
    RVV_OPIVX(F6_VMUL, 9, 6, 4)  /* v9 = v4 * t1 */
    RVV_OPIVV(F6_VADD, 8, 9, 8)

    lw      t1, 20(s0)
    RVV_OPIVX(F6_VMUL, 9, 6, 5)  /* v9 = v5 * t1 */
    RVV_OPIVV(F6_VADD, 8, 9, 8)

    lw      t1, 24(s0)
    RVV_OPIVX(F6_VMUL, 9, 6, 6)  /* v9 = v6 * t1 */
    RVV_OPIVV(F6_VADD, 8, 9, 8)

    lw      t1, 28(s0)
    RVV_OPIVX(F6_VMUL, 9, 6, 7)  /* v9 = v7 * t1 */
    RVV_OPIVV(F6_VADD, 8, 9, 8)

    RVV_VSE32(8, 18)         /* store C[i,*] */

    addi    s0, s0, 32       /* next row of A */
    addi    s2, s2, 32       /* next row of C */
    addi    t0, t0, -1
    bnez    t0, 1b

    lw   ra, 12(sp)
    lw   s0,  8(sp)
    lw   s1,  4(sp)
    lw   s2,  0(sp)
    addi sp, sp, 16
    ret
