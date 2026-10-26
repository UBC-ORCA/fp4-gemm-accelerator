#pragma once
#include <stdint.h>

#define BRAM_GEN_1 1
#define BRAM_GEN_2 2
#define BRAM_GEN BRAM_GEN_2

// =======================================
// MAC accelerator instructions
// =======================================
// Encodings mirror matmul8_vec.S. CUSTOM1=0x2b, CUSTOM2=0x5b, funct3=0 unless noted
//   VMAC64(N,ptr,off)  VMAC64   CUSTOM1   vN x weight block at ptr+off -> raw tile
//   MAC_AS(a,b)        MAC_AS   f7=0x0A   apply act scale words a,b to the tile
//   MAC_WS(a,b)        MAC_WS   f7=0x0B   apply wgt scale words a,b -> fold into bank
//   ACC_BANK(t)        ACCBANK  f7=0x0D   select accumulator bank t
//   BRAM_RD(rd,p)      BRAMRD   f7=0x0E   read the bram pair at p into register rd
//   VSETVLI(avl)       vsetvli  OPV=0x57  set vl=avl, e32,m1,ta,ma
//   VLE32(N,ptr)       vle32.v  0x07      load vl words at ptr -> vN
// static assert and macro used for VMAC64
// off is unsigned in the unit, the assembler only takes -2048..2047
#define UINT12_TO_INT12(u12) (((u12) & 0x7FF) - ((u12) & 0x800))
#define VMAC64(N,ptr,off)   __asm__ volatile(".insn i 0x2b,0x0,x" #N ",%0,%c1" :: "r"(ptr), "i"(UINT12_TO_INT12(off)))
#define VMAC64L(N,ptr,off)      __asm__ volatile(".insn i 0x2b,0x1,x" #N ",%0,%c1" :: "r"(ptr), "i"(UINT12_TO_INT12(off)))
#define VMAC64H(N,ptr,off)      __asm__ volatile(".insn i 0x2b,0x2,x" #N ",%0,%c1" :: "r"(ptr), "i"(UINT12_TO_INT12(off)))
#define MAC_AS(a,b)         __asm__ volatile(".insn r 0x5b,0x0,0x0a, x0,%0,%1" :: "r"(a),"r"(b))
#define MAC_WS(a,b)         __asm__ volatile(".insn r 0x5b,0x0,0x0b, x0,%0,%1" :: "r"(a),"r"(b))
#define ACC_BANK(t)         __asm__ volatile(".insn r 0x5b,0x0,0x0d, x0,%0,x0" :: "r"(t))
#define BRAM_RD(rd,p)       __asm__ volatile(".insn r 0x5b,0x0,0x0e, %0,%1,x0" : "=r"(rd) : "r"(p))
#define VSETVLI(avl)        __asm__ volatile(".insn i 0x57,0x7,x0,%0,0xD0" :: "r"(avl))
#define VLE32(N,ptr)        __asm__ volatile(".insn i 0x07,0x6,x" #N ",%0,0x20" :: "r"(ptr))

#if (BRAM_GEN == BRAM_GEN_1)
    //   MAC_BIAS(p,v)      MACBIAS  f7=0x0C   seed bf16 v into the cell addressed by p
    #define BRAM_RD_FP4(rd,p)   __asm__ volatile(".insn r 0x5b,0x0,0x08, %0,%1,x0" : "=r"(rd) : "r"(p))
    #define MAC_BIAS(p,v)       __asm__ volatile(".insn r 0x5b,0x0,0x0c, x0,%0,%1" :: "r"(p),"r"(v))
#elif (BRAM_GEN == BRAM_GEN_2)
    /*
        BRAM_RD_FP4(rd,p)  BRAMFP4  f7=0x08    move bram[t, :, c] to rd, and move bias to bram[t, :, c]
        MAC_BIAS(p,v)      BRAMFP4  f7=0x08    same op with rd=x0: seed bias v, discard the readout
    */
    #define BRAM_RD_FP4(rd,p,bias)  __asm__ volatile(".insn r 0x5b,0x0,0x08, %0,%1,%2" : "=r"(rd) : "r"(p), "r"(bias))
    #define MAC_BIAS(p,v)           __asm__ volatile(".insn r 0x5b,0x0,0x08, x0,%0,%1" :: "r"(p),"r"(v))
    #define BRAM_MOV(rd, p, bias)   BRAM_RD_FP4(rd, p, bias)
#endif

// Address payload for a bram cell: {tile[10:6], row[5:3], col[2:0]}
#define BRAM_ADDR(tile, row, col)  (((uint32_t)(tile) << 6) | ((uint32_t)(row) << 3) | (col))
#define MAC_BIAS_TILE(tile, row, col, val)   MAC_BIAS(BRAM_ADDR(tile, row, col), val)

// Broadcast one accumulator cell (tile,row,col) with a bf16 bias value in the row dimension
static inline void mac_bias(uint8_t tile, uint8_t row, uint8_t column, uint16_t bf16) {
    MAC_BIAS(BRAM_ADDR(tile, row, column), (uint32_t)bf16);
}

// Read the accumulator bram pair {row+1,row} at (tile,row,col) into a register
static inline uint32_t bram_rd(uint32_t tile, uint32_t row, uint32_t col) {
    uint32_t result;
    BRAM_RD(result, BRAM_ADDR(tile, row, col));
    return result;
}

// New BRAM MOV Instruction