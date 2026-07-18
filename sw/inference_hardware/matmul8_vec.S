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
#define CUSTOM1 0x2b    // 0b0101011
#define CUSTOM2 0x5b    // 0b1011011

#define ZZMAC64 \
.word (((0x00)<<25) | (0<<20) | (0<<15) | (0<<12) | (0<<7) | CUSTOM2)

#define HWMAC64(rs1,rs2) \
.word (((0x02)<<25) | ((rs2)<<20) | ((rs1)<<15) | (0<<12) | (0<<7) | CUSTOM2)

//MEM
/*
31                20 19:15 14:12 11:7 6:0
+-------------------+------+------+------+-------+
|   imm[11:0]       | rs1  |000   | vs1  |CUSTOM1|
+-------------------+------+------+------+-------+
*/

#define VMAC64(vs1, offset, base) \
.word ((((offset) & 0xFFF) << 20) | \
       ((base) << 15)            | \
       (0 << 12)                 | \
       ((vs1) << 7)              | \
       CUSTOM1)

//MEM_end

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

/*
.globl mac_add_row
mac_add_row:
    ADDMAC64(3, 10)   # row=3, value in a0
    ret
*/

.globl mac_add_row
mac_add_row:
    # a0 = row (0..7)
    # a1 = value

    li      t2, 8
    bgeu    a0, t2, 1f          # optional bounds check

    slli    t0, a0, 2           # sizeof(pointer)=4
    la      t1, add_table
    add     t1, t1, t0
    lw      t1, 0(t1)
    jr      t1

1:
    ret                         # invalid row

.align 2
add_table:
    .word add_row0
    .word add_row1
    .word add_row2
    .word add_row3
    .word add_row4
    .word add_row5
    .word add_row6
    .word add_row7

add_row0:
    ADDMAC64(0, 11)
    ret

add_row1:
    ADDMAC64(1, 11)
    ret

add_row2:
    ADDMAC64(2, 11)
    ret

add_row3:
    ADDMAC64(3, 11)
    ret

add_row4:
    ADDMAC64(4, 11)
    ret

add_row5:
    ADDMAC64(5, 11)
    ret

add_row6:
    ADDMAC64(6, 11)
    ret

add_row7:
    ADDMAC64(7, 11)
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


//mv handler
.globl mac_out
mac_out:
    # a0 = row (0..7)
    # a1 = pair (0..3)
    # a2 = mode
    # returns a0

    li      t2, 8
    bgeu    a0, t2, 1f

    li      t2, 4
    bgeu    a1, t2, 1f

    li      t2, 3
    bgeu    a2, t2, 1f

    beqz    a2, even_dispatch

    li      t2, 1
    beq     a2, t2, odd_dispatch

    j       pair_dispatch

1:
    li      a0, 0
    ret

even_dispatch:
    slli    t0, a0, 2      # row * 4
    add     t0, t0, a1     # + pair
    slli    t0, t0, 2      # *4

    la      t1, even_table
    add     t1, t1, t0
    lw      t1, 0(t1)
    jr      t1

.align 2
even_table:
    .word even00
    .word even01
    .word even02
    .word even03

    .word even10
    .word even11
    .word even12
    .word even13

    .word even20
    .word even21
    .word even22
    .word even23

    .word even30
    .word even31
    .word even32
    .word even33

    .word even40
    .word even41
    .word even42
    .word even43

    .word even50
    .word even51
    .word even52
    .word even53

    .word even60
    .word even61
    .word even62
    .word even63

    .word even70
    .word even71
    .word even72
    .word even73

even00:
    MVEMAC64(10,0,0)
    ret

even01:
    MVEMAC64(10,0,1)
    ret

even02:
    MVEMAC64(10,0,2)
    ret

even03:
    MVEMAC64(10,0,3)
    ret


even10:
    MVEMAC64(10,1,0)
    ret

even11:
    MVEMAC64(10,1,1)
    ret

even12:
    MVEMAC64(10,1,2)
    ret

even13:
    MVEMAC64(10,1,3)
    ret

even20:
    MVEMAC64(10,2,0)
    ret

even21:
    MVEMAC64(10,2,1)
    ret

even22:
    MVEMAC64(10,2,2)
    ret

even23:
    MVEMAC64(10,2,3)
    ret

even30:
    MVEMAC64(10,3,0)
    ret

even31:
    MVEMAC64(10,3,1)
    ret

even32:
    MVEMAC64(10,3,2)
    ret

even33:
    MVEMAC64(10,3,3)
    ret

even40:
    MVEMAC64(10,4,0)
    ret

even41:
    MVEMAC64(10,4,1)
    ret

even42:
    MVEMAC64(10,4,2)
    ret

even43:
    MVEMAC64(10,4,3)
    ret

even50:
    MVEMAC64(10,5,0)
    ret

even51:
    MVEMAC64(10,5,1)
    ret

even52:
    MVEMAC64(10,5,2)
    ret

even53:
    MVEMAC64(10,5,3)
    ret

even60:
    MVEMAC64(10,6,0)
    ret

even61:
    MVEMAC64(10,6,1)
    ret

even62:
    MVEMAC64(10,6,2)
    ret

even63:
    MVEMAC64(10,6,3)
    ret

even70:
    MVEMAC64(10,7,0)
    ret

even71:
    MVEMAC64(10,7,1)
    ret

even72:
    MVEMAC64(10,7,2)
    ret

even73:
    MVEMAC64(10,7,3)
    ret


odd_dispatch:
    slli    t0, a0, 2      # row * 4
    add     t0, t0, a1     # + pair
    slli    t0, t0, 2      # *4

    la      t1, odd_table
    add     t1, t1, t0
    lw      t1, 0(t1)
    jr      t1

.align 2
odd_table:
    .word odd00
    .word odd01
    .word odd02
    .word odd03

    .word odd10
    .word odd11
    .word odd12
    .word odd13

    .word odd20
    .word odd21
    .word odd22
    .word odd23

    .word odd30
    .word odd31
    .word odd32
    .word odd33

    .word odd40
    .word odd41
    .word odd42
    .word odd43

    .word odd50
    .word odd51
    .word odd52
    .word odd53

    .word odd60
    .word odd61
    .word odd62
    .word odd63

    .word odd70
    .word odd71
    .word odd72
    .word odd73

odd00:
    MVOMAC64(10,0,0)
    ret

odd01:
    MVOMAC64(10,0,1)
    ret

odd02:
    MVOMAC64(10,0,2)
    ret

odd03:
    MVOMAC64(10,0,3)
    ret

odd10:
    MVOMAC64(10,1,0)
    ret

odd11:
    MVOMAC64(10,1,1)
    ret

odd12:
    MVOMAC64(10,1,2)
    ret

odd13:
    MVOMAC64(10,1,3)
    ret

odd20:
    MVOMAC64(10,2,0)
    ret

odd21:
    MVOMAC64(10,2,1)
    ret

odd22:
    MVOMAC64(10,2,2)
    ret

odd23:
    MVOMAC64(10,2,3)
    ret

odd30:
    MVOMAC64(10,3,0)
    ret

odd31:
    MVOMAC64(10,3,1)
    ret

odd32:
    MVOMAC64(10,3,2)
    ret

odd33:
    MVOMAC64(10,3,3)
    ret

odd40:
    MVOMAC64(10,4,0)
    ret

odd41:
    MVOMAC64(10,4,1)
    ret

odd42:
    MVOMAC64(10,4,2)
    ret

odd43:
    MVOMAC64(10,4,3)
    ret

odd50:
    MVOMAC64(10,5,0)
    ret

odd51:
    MVOMAC64(10,5,1)
    ret

odd52:
    MVOMAC64(10,5,2)
    ret

odd53:
    MVOMAC64(10,5,3)
    ret

odd60:
    MVOMAC64(10,6,0)
    ret

odd61:
    MVOMAC64(10,6,1)
    ret

odd62:
    MVOMAC64(10,6,2)
    ret

odd63:
    MVOMAC64(10,6,3)
    ret

odd70:
    MVOMAC64(10,7,0)
    ret

odd71:
    MVOMAC64(10,7,1)
    ret

odd72:
    MVOMAC64(10,7,2)
    ret

odd73:
    MVOMAC64(10,7,3)
    ret

pair_dispatch:
    slli    t0, a0, 2      # row * 4
    add     t0, t0, a1     # + pair
    slli    t0, t0, 2      # *4

    la      t1, pair_table
    add     t1, t1, t0
    lw      t1, 0(t1)
    jr      t1

.align 2
pair_table:
    .word pair00
    .word pair01
    .word pair02
    .word pair03

    .word pair10
    .word pair11
    .word pair12
    .word pair13

    .word pair20
    .word pair21
    .word pair22
    .word pair23

    .word pair30
    .word pair31
    .word pair32
    .word pair33

    .word pair40
    .word pair41
    .word pair42
    .word pair43

    .word pair50
    .word pair51
    .word pair52
    .word pair53

    .word pair60
    .word pair61
    .word pair62
    .word pair63

    .word pair70
    .word pair71
    .word pair72
    .word pair73

pair00:
    MV2MAC64(10,0,0)
    ret

pair01:
    MV2MAC64(10,0,1)
    ret

pair02:
    MV2MAC64(10,0,2)
    ret

pair03:
    MV2MAC64(10,0,3)
    ret

pair10:
    MV2MAC64(10,1,0)
    ret

pair11:
    MV2MAC64(10,1,1)
    ret

pair12:
    MV2MAC64(10,1,2)
    ret

pair13:
    MV2MAC64(10,1,3)
    ret

pair20:
    MV2MAC64(10,2,0)
    ret

pair21:
    MV2MAC64(10,2,1)
    ret

pair22:
    MV2MAC64(10,2,2)
    ret

pair23:
    MV2MAC64(10,2,3)
    ret

pair30:
    MV2MAC64(10,3,0)
    ret

pair31:
    MV2MAC64(10,3,1)
    ret

pair32:
    MV2MAC64(10,3,2)
    ret

pair33:
    MV2MAC64(10,3,3)
    ret

pair40:
    MV2MAC64(10,4,0)
    ret

pair41:
    MV2MAC64(10,4,1)
    ret

pair42:
    MV2MAC64(10,4,2)
    ret

pair43:
    MV2MAC64(10,4,3)
    ret

pair50:
    MV2MAC64(10,5,0)
    ret

pair51:
    MV2MAC64(10,5,1)
    ret

pair52:
    MV2MAC64(10,5,2)
    ret

pair53:
    MV2MAC64(10,5,3)
    ret

pair60:
    MV2MAC64(10,6,0)
    ret

pair61:
    MV2MAC64(10,6,1)
    ret

pair62:
    MV2MAC64(10,6,2)
    ret

pair63:
    MV2MAC64(10,6,3)
    ret

pair70:
    MV2MAC64(10,7,0)
    ret

pair71:
    MV2MAC64(10,7,1)
    ret

pair72:
    MV2MAC64(10,7,2)
    ret

pair73:
    MV2MAC64(10,7,3)
    ret


/*
31        25 24       20 19       15 14 12 11       7 6       0
+-----------+-----------+-----------+-----+-----------+---------+
| funct7=09 |   blk     |   rs1     |     |   vs1     | CUSTOM2 |
+-----------+-----------+-----------+-----+-----------+---------+
*/


// -----------------------------------------------------------------------------
// v0
// -----------------------------------------------------------------------------
.globl load_v0
load_v0:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(0,10)
    ret

.globl vmac64_v0
vmac64_v0:
    VMAC64(0,0,10)
    ret

// -----------------------------------------------------------------------------
// v1
// -----------------------------------------------------------------------------
.globl load_v1
load_v1:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(1,10)
    ret

.globl vmac64_v1
vmac64_v1:
    VMAC64(1,128,10)
    ret

// -----------------------------------------------------------------------------
// v2
// -----------------------------------------------------------------------------
.globl load_v2
load_v2:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(2,10)
    ret

.globl vmac64_v2
vmac64_v2:
    VMAC64(2,256,10)
    ret

// -----------------------------------------------------------------------------
// v3
// -----------------------------------------------------------------------------
.globl load_v3
load_v3:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(3,10)
    ret

.globl vmac64_v3
vmac64_v3:
    VMAC64(3,384,10)
    ret

// -----------------------------------------------------------------------------
// v4
// -----------------------------------------------------------------------------
.globl load_v4
load_v4:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(4,10)
    ret

.globl vmac64_v4
vmac64_v4:
    VMAC64(4,512,10)
    ret

// -----------------------------------------------------------------------------
// v5
// -----------------------------------------------------------------------------
.globl load_v5
load_v5:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(5,10)
    ret

.globl vmac64_v5
vmac64_v5:
    VMAC64(5,640,10)
    ret

// -----------------------------------------------------------------------------
// v6
// -----------------------------------------------------------------------------
.globl load_v6
load_v6:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(6,10)
    ret

.globl vmac64_v6
vmac64_v6:
    VMAC64(6,768,10)
    ret

// -----------------------------------------------------------------------------
// v7
// -----------------------------------------------------------------------------
.globl load_v7
load_v7:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(7,10)
    ret

.globl vmac64_v7
vmac64_v7:
    VMAC64(7,896,10)
    ret

// -----------------------------------------------------------------------------
// v8
// -----------------------------------------------------------------------------
.globl load_v8
load_v8:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(8,10)
    ret

.globl vmac64_v8
vmac64_v8:
    VMAC64(8,1024,10)
    ret

// -----------------------------------------------------------------------------
// v9
// -----------------------------------------------------------------------------
.globl load_v9
load_v9:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(9,10)
    ret

.globl vmac64_v9
vmac64_v9:
    VMAC64(9,1152,10)
    ret

// -----------------------------------------------------------------------------
// v10
// -----------------------------------------------------------------------------
.globl load_v10
load_v10:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(10,10)
    ret

.globl vmac64_v10
vmac64_v10:
    VMAC64(10,1280,10)
    ret

// -----------------------------------------------------------------------------
// v11
// -----------------------------------------------------------------------------
.globl load_v11
load_v11:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(11,10)
    ret

.globl vmac64_v11
vmac64_v11:
    VMAC64(11,1408,10)
    ret

// -----------------------------------------------------------------------------
// v12
// -----------------------------------------------------------------------------
.globl load_v12
load_v12:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(12,10)
    ret

.globl vmac64_v12
vmac64_v12:
    VMAC64(12,1536,10)
    ret

// -----------------------------------------------------------------------------
// v13
// -----------------------------------------------------------------------------
.globl load_v13
load_v13:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(13,10)
    ret

.globl vmac64_v13
vmac64_v13:
    VMAC64(13,1664,10)
    ret

// -----------------------------------------------------------------------------
// v14
// -----------------------------------------------------------------------------
.globl load_v14
load_v14:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(14,10)
    ret

.globl vmac64_v14
vmac64_v14:
    VMAC64(14,1792,10)
    ret

// -----------------------------------------------------------------------------
// v15
// -----------------------------------------------------------------------------
.globl load_v15
load_v15:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(15,10)
    ret

.globl vmac64_v15
vmac64_v15:
    VMAC64(15,1920,10)
    ret

// -----------------------------------------------------------------------------
// v16
// -----------------------------------------------------------------------------
.globl load_v16
load_v16:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(16,10)
    ret

.globl vmac64_v16
vmac64_v16:
    VMAC64(16,2048,10)
    ret

// -----------------------------------------------------------------------------
// v17
// -----------------------------------------------------------------------------
.globl load_v17
load_v17:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(17,10)
    ret

.globl vmac64_v17
vmac64_v17:
    VMAC64(17,2176,10)
    ret

// -----------------------------------------------------------------------------
// v18
// -----------------------------------------------------------------------------
.globl load_v18
load_v18:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(18,10)
    ret

.globl vmac64_v18
vmac64_v18:
    VMAC64(18,2304,10)
    ret

// -----------------------------------------------------------------------------
// v19
// -----------------------------------------------------------------------------
.globl load_v19
load_v19:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(19,10)
    ret

.globl vmac64_v19
vmac64_v19:
    VMAC64(19,2432,10)
    ret

// -----------------------------------------------------------------------------
// v20
// -----------------------------------------------------------------------------
.globl load_v20
load_v20:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(20,10)
    ret

.globl vmac64_v20
vmac64_v20:
    VMAC64(20,2560,10)
    ret

// -----------------------------------------------------------------------------
// v21
// -----------------------------------------------------------------------------
.globl load_v21
load_v21:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(21,10)
    ret

.globl vmac64_v21
vmac64_v21:
    VMAC64(21,2688,10)
    ret

// -----------------------------------------------------------------------------
// v22
// -----------------------------------------------------------------------------
.globl load_v22
load_v22:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(22,10)
    ret

.globl vmac64_v22
vmac64_v22:
    VMAC64(22,2816,10)
    ret

// -----------------------------------------------------------------------------
// v23
// -----------------------------------------------------------------------------
.globl load_v23
load_v23:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(23,10)
    ret

.globl vmac64_v23
vmac64_v23:
    VMAC64(23,2944,10)
    ret

// -----------------------------------------------------------------------------
// v24
// -----------------------------------------------------------------------------
.globl load_v24
load_v24:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(24,10)
    ret

.globl vmac64_v24
vmac64_v24:
    VMAC64(24,3072,10)
    ret

// -----------------------------------------------------------------------------
// v25
// -----------------------------------------------------------------------------
.globl load_v25
load_v25:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(25,10)
    ret

.globl vmac64_v25
vmac64_v25:
    VMAC64(25,3200,10)
    ret

// -----------------------------------------------------------------------------
// v26
// -----------------------------------------------------------------------------
.globl load_v26
load_v26:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(26,10)
    ret

.globl vmac64_v26
vmac64_v26:
    VMAC64(26,3328,10)
    ret

// -----------------------------------------------------------------------------
// v27
// -----------------------------------------------------------------------------
.globl load_v27
load_v27:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(27,10)
    ret

.globl vmac64_v27
vmac64_v27:
    VMAC64(27,3456,10)
    ret

// -----------------------------------------------------------------------------
// v28
// -----------------------------------------------------------------------------
.globl load_v28
load_v28:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(28,10)
    ret

.globl vmac64_v28
vmac64_v28:
    VMAC64(28,3584,10)
    ret

// -----------------------------------------------------------------------------
// v29
// -----------------------------------------------------------------------------
.globl load_v29
load_v29:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(29,10)
    ret

.globl vmac64_v29
vmac64_v29:
    VMAC64(29,3712,10)
    ret

// -----------------------------------------------------------------------------
// v30
// -----------------------------------------------------------------------------
.globl load_v30
load_v30:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(30,10)
    ret

.globl vmac64_v30
vmac64_v30:
    VMAC64(30,3840,10)
    ret

// -----------------------------------------------------------------------------
// v31
// -----------------------------------------------------------------------------
.globl load_v31
load_v31:
    li      t0,32
    RVV_VSETVLI(0,5,0x0D0)
    RVV_VLE32(31,10)
    ret

.globl vmac64_v31
vmac64_v31:
    VMAC64(31,3968,10)
    ret





// void load_act_scales(const uint8_t *base);
// a0 = base address

.globl load_act_scales
load_act_scales:
    lw  t1, 0(a0) # x6  = act_scale[0]
    lw  t2, 4(a0)  # x7  = act_scale[1]
    ret

.globl load_w_scales
load_w_scales:
    lw  t3, 0(a0) # x28 = w_scale[0]
    lw  t4, 4(a0) # x29 = w_scale[1]
    ret


/*
31      30 29           25 24      20 19      15 14    12 11         7 6      0
+---------+---------------+----------+----------+--------+------------+--------+
| unused  | t3[4:0]       | t1       | t0       |cf_id   | t2         |1011011 |
| [1:0]   |               |          |          | [2:0]  |            |        |
+---------+---------------+----------+----------+--------+------------+--------+

Bits:
[31:30] = unused[1:0]
[29:25] = t3[4:0]
[24:20] = t1
[19:15] = t0
[14:12] = cf_id[2:0]
[11:7] = t2
[6:0] = 1011011 (custom-2 opcode)
*/


/*
31        25 24    20 19    15 14   12 11        7 6      0
+------------+--------+--------+-------+-----------+--------+
| cf_id[9:3] |  rs2   |  rs1   |cf_id  |  custom   | 1011011
|            |        |        | [2:0] |           |        |
+------------+--------+--------+-------+-----------+--------+

Bits:
[31:25] = cf_id[9:3]
[24:20] = rs2
[19:15] = rs1
[14:12] = cf_id[2:0]
[11:7]  = custom
[6:0]   = 1011011   (custom-2 opcode)

s0 = x8
s1 = x9
s2 = x18
s3 = x19

t1 = x6 
t2 = x7 

*/

#define MAC_AS(rs1, rs2) \
.word (((0x0A & 0x7F) << 25) | \
       (((rs2) & 0x1F) << 20) | \
       (((rs1) & 0x1F) << 15) | \
       (0 << 12) | \
       (0 << 7) | \
       CUSTOM2)

#define MAC_WS(rs1, rs2) \
.word (((0x0B & 0x7F) << 25) | \
       (((rs2) & 0x1F) << 20) | \
       (((rs1) & 0x1F) << 15) | \
       (0 << 12) | \
       (0 << 7) | \
       CUSTOM2)


.globl mac_as
mac_as:
    # MAC_AS s0, s1
    MAC_AS(6,7)
    ret

.globl mac_ws
mac_ws:
    # MAC_WS s2, s3
    MAC_WS(28,29)
    ret

// write bram
/*
MACBIAS
--------

rs1[10:0]

31             11 10       6 5     3 2     0
+----------------+----------+-------+-------+
|    unused      |  tile    |  row  |  col  |
+----------------+----------+-------+-------+
                  5 bits    3 bits  3 bits

rs2
31                          16 15              0
------------------------------------------------
|            unused           |    BF16 value   |
------------------------------------------------


instruction format
-----------------

31        25 24       20 19       15 14:12 11       7 6      0
+------------+-----------+-----------+-------+-----------+--------+
| funct7=0C  |   rs2     |   rs1     | 000   |     0     |CUSTOM2 |
+------------+-----------+-----------+-------+-----------+--------+

 assign bias_col   = rs1_i[2:0];
    assign bias_row   = rs1_i[5:3];
    assign bias_tile  = rs1_i[10:6];
    assign bias_value = rs2_i[15:0];

*/

#define MACBIAS(rs1, rs2) \
.word (((0x0C & 0x7F) << 25) | \
       (((rs2) & 0x1F) << 20) | \
       (((rs1) & 0x1F) << 15) | \
       (0 << 12)              | \
       (0 << 7)               | \
       CUSTOM2)

// accBank(tile): select the current accumulator bank (funct7=0x0D, CUSTOM2)
// rs1[4:0] = tile
#define ACCBANK(rs1) \
.word (((0x0D & 0x7F) << 25) | \
       (0 << 20)              | \
       (((rs1) & 0x1F) << 15) | \
       (0 << 12)              | \
       (0 << 7)               | \
       CUSTOM2)

.globl acc_bank
acc_bank:
    # a0 = tile -> rs1[4:0]
    ACCBANK(10)
    ret

// bram_rd(tile,row,col): read the accumulator BRAM pair {row+1,row} at col -> rd (GPR)
// funct7=0x0E, CUSTOM2. rs1 holds {tile[10:6],row[5:3],col[2:0]}, rd = destination.
#define BRAMRD(rd, rs1) \
.word (((0x0E & 0x7F) << 25) | \
       (0 << 20)              | \
       (((rs1) & 0x1F) << 15) | \
       (0 << 12)              | \
       (((rd) & 0x1F) << 7)   | \
       CUSTOM2)

.globl bram_rd
bram_rd:
    # a0 = tile, a1 = row (EVEN), a2 = col  -> returns {row+1,row} pair in a0
    slli    t5, a0, 6       # tile << 6   (t5/t6 scratch: preserve scale regs t1-t4)
    slli    t6, a1, 3       # row << 3
    or      t5, t5, t6
    or      t5, t5, a2      # t5 = {tile[10:6], row[5:3], col[2:0]}
    BRAMRD(10, 30)          # rd = x10 = a0, rs1 = x30 = t5
    ret


#
# a0 = tile
# a1 = row
# a2 = column
# a3 = BF16 value
#
# uses:
#   x6 = rs1 payload
#   x7 = rs2 payload
#

.globl mac_bias
mac_bias:

    #
    # rs1 payload:
    #
    # [10:6] = tile (5 bits)
    # [5:3]  = row  (3 bits)
    # [2:0]  = col  (3 bits)
    #

    # Use t5/t6 as scratch so t1-t4 (the caller's act/weight scales,
    # loaded by load_act_scales/load_w_scales and consumed by mac_as/mac_ws)
    # are preserved across a mac_bias call.
    slli    t5, a0, 6       # tile << 6
    slli    t6, a1, 3       # row << 3

    or      t5, t5, t6      # add row
    or      t5, t5, a2      # add column

    #
    # rs2 payload:
    #
    # [15:0] = BF16 value
    #

    mv      t6, a3


    #
    # x30 = t5 = rs1 payload
    # x31 = t6 = rs2 payload
    #

    MACBIAS(30,31)

    ret
//[end]


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
