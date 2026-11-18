// MLP inference on CVE2 with GEN3 spec

#include <assert.h>
#include "weights_blk32_pkgUINT32_scaleE8M0.h"
#include "image.h"

#if defined(DS_CIFAR10) || defined(DS_CIFAR100)
  #define IN_REAL   3072            // 32*32*3 rgb
  #define IN_DIM    3072
  #define L1_DIM     256
  #define L2_DIM      96
  #define OUT_DIM     10
#else
  #define IN_REAL    784            // 28*28 greyscale
  #define IN_DIM     800            // pad K to a BS multiple
  #define L1_DIM     128
  #define L2_DIM      96
  #define OUT_DIM     10
#endif

#define BATCH       8             // Parallel batch size for MNIST samples
#define TT          8             // Tile dimension for 8x8 MACs        
#define NVREG      32             // Number of vector register
#define NTILES     32             // Number of tiles that fit in BRAM
#define AV          8             // Activation matrix vertical size

// Layer 1 input scales per image (one per block)
#define NUM_SCALES (IN_DIM / SCALE_BLK_HDR)

// MAC array is TT x TT cells
#define MAC_UNITS       (TT * TT)
// Useful MACs per image
#define MACS_PER_IMAGE  ((uint64_t)IN_REAL * L1_DIM + \
                         (uint64_t)L1_DIM  * L2_DIM + \
                         (uint64_t)L2_DIM  * OUT_DIM)
// FlOPS is 2 * MAC
#define FLOPS_PER_IMAGE   (2 * MACS_PER_IMAGE)
#define PEAK_FLOPS_CYCLE  (2 * MAC_UNITS)

// Number of K elements per inner block
#define BS  K1_STEP_HDR

// Read-out shift per layer (already in hardware, used for assert)
static const int rdout_shift[3] = RDOUT_SHIFT_HDR;

// Enable performance counters
#define PERF_COUNTERS

// Enable intermediate UART prints (P|T|M results)
// #define PTM_PRINTS

//[PROF]
#define PAIR_PROFILE
#define PAIR_COMPRESS_TEST

static volatile uint32_t *const DONE_MMIO = (volatile uint32_t *)0xFFFF0000u;

static inline void kill_simulation(void) {
    *DONE_MMIO = 0xFF; // Signal simulator testbench to halt
    while (1);         // Catch pipeline flush before termination
}

//[end]

void __assert_func(const char *f,int l,const char *fn,const char *e){
    (void)f;(void)l;(void)fn;(void)e; __builtin_trap();
}

// =======================================
// UART output helpers
// =======================================
extern void putchar_uart(char c);

static void print_str(const char *s) {
  while (*s) putchar_uart(*s++);
}

static void putdec(uint32_t n) {
    char buf[11];
    int i = 0;
    if (n == 0) { 
        putchar_uart('0'); 
        return; 
    }
    while (n > 0) { 
        buf[i++] = '0' + (n % 10); 
        n /= 10; 
    }
    while (i--) putchar_uart(buf[i]);
}

// =======================================
// Performance counters (mcycle)
// =======================================
#ifdef PERF_COUNTERS
// Read the cycle counter by enabling zicsr temporarily for rv32im
static inline uint32_t rdcyc(void) {
    uint32_t c;
    __asm__ volatile (".option push\n\t"
                      ".option arch, +zicsr\n\t"
                      "csrr %0, mcycle\n\t"
                      ".option pop" : "=r"(c));
    return c;
}

static uint64_t pc_imgq;
static uint64_t pc_gemm[3];
static uint64_t pc_qact[2];    // hidden read-out as packed FP4 (readout_fp4)
static uint64_t pc_argmax;     // final read-out + argmax
static uint64_t pc_load[3];    // activation vector loads, inside gemm
static uint64_t pc_ktile[3];   // do_k_tile loop only, inside gemm
static uint64_t pc_total;      // whole sample loop, one span, loop control included
static int      pc_layer;

#define PC_LAYER(n)     (pc_layer = (n))
#define TIME(acc, stmt) do { uint32_t _t = rdcyc(); stmt; (acc) += (uint32_t)(rdcyc() - _t); } while (0)

// Begin/end pair for regions carrying a GCC pragma, which cannot sit in a macro argument
#define TIME_BEG(t)      uint32_t t = rdcyc()
#define TIME_END(acc, t) (acc) += (uint32_t)(rdcyc() - (t))

// Print 64 bit integer since totals can exceed 32 bits
static void putdec64(uint64_t n) {   
    char buf[21]; int i = 0;
    if (n == 0) { 
        putchar_uart('0'); 
        return; 
    }
    while (n > 0) { 
        buf[i++] = '0' + (int)(n % 10); 
        n /= 10; 
    }
    while (i--) putchar_uart(buf[i]);
}

static void pc_line(const char *name, uint64_t v) {
    print_str("  "); 
    print_str(name); 
    print_str(" "); 
    putdec64(v); 
    print_str("\n");
}

// Print scaled/100 with 2 decimals
static void pc_fixed2(const char *name, uint64_t scaled, const char *unit) {
    print_str("  ");
    print_str(name);
    print_str(" ");
    putdec64(scaled / 100);
    putchar_uart('.');
    if (scaled % 100 < 10) putchar_uart('0');
    putdec64(scaled % 100);
    print_str(unit);
    print_str("\n");
}

// part as a percentage of whole
static void pc_percent(const char *name, uint64_t part, uint64_t whole) {
    uint64_t scaled = 0;
    if (whole) {
        scaled = (part * 10000) / whole;
    }
    pc_fixed2(name, scaled, " %");
}

// how much of count happens per cycle, used for flops per cycle
static void pc_rate(const char *name, uint64_t count, uint64_t cycles) {
    uint64_t scaled = 0;
    if (cycles) {
        scaled = (count * 100) / cycles;
    }
    pc_fixed2(name, scaled, "");
}

static void pc_report(void) {
    print_str("\n[PERF] cycles over run\n");
    pc_line("imgq   |", pc_imgq);       // image load and quant

    pc_line("gemm_1 |", pc_gemm[0]);    // gemm compute per layer
    pc_line("gemm_2 |", pc_gemm[1]);
    pc_line("gemm_3 |", pc_gemm[2]);

    pc_line("qact_1 |", pc_qact[0]);    // hidden read-out from banks as packed FP4
    pc_line("qact_2 |", pc_qact[1]);

    pc_line("argmax |", pc_argmax);     // final read-out from banks + argmax

    pc_line("load_1 |", pc_load[0]);    // activation vector loads (subset of gemm)
    pc_line("load_2 |", pc_load[1]);
    pc_line("load_3 |", pc_load[2]);

    pc_line("ktile_1|", pc_ktile[0]);   // do_k_tile loop only (subset of gemm)
    pc_line("ktile_2|", pc_ktile[1]);
    pc_line("ktile_3|", pc_ktile[2]);

    // widest view is the whole gemm function, narrowest is the do_k_tile loop
    uint64_t gemm_t = pc_gemm[0]  + pc_gemm[1]  + pc_gemm[2];
    uint64_t gemm_k = pc_ktile[0] + pc_ktile[1] + pc_ktile[2];
    uint64_t load_t = pc_load[0]  + pc_load[1]  + pc_load[2];
    uint64_t qact_t = pc_qact[0]  + pc_qact[1];
    // end to end inference, imgq left out since real inputs arrive as FP4
    uint64_t infer_t = gemm_t + qact_t + pc_argmax;

    print_str("\n[PERF] total cycles over run\n");
    // gemm() only; read-out and argmax sit outside the function
    pc_line("gemm_T |", gemm_t);
    // the MAC datapath on its own
    pc_line("gemm_K |", gemm_k);
    // activation loads inside gemm
    pc_line("load_T |", load_t);
    // total read-out as packed FP4
    pc_line("qact_T |", qact_t);
    // end to end inference, the runtime every ratio below is taken over
    pc_line("infer_T|", infer_t);
    // whole sample loop for reference, imgq and loop control included
    pc_line("total  |", pc_total);

    // Useful MACs against what the array could have retired in the same cycles
    uint64_t macs = MACS_PER_IMAGE * (uint64_t)N_SAMPLES;

    print_str("\n[PERF] MAC utilization\n");
    pc_percent("usage  |", macs, MAC_UNITS * infer_t);   // end to end inference
    pc_percent("gemm_U |", macs, MAC_UNITS * gemm_k);    // do_k_tile loop only

    // MAC array only, scaling and conversion are not counted
    uint64_t flops = FLOPS_PER_IMAGE * (uint64_t)N_SAMPLES;

    print_str("\n[PERF] flops\n");
    pc_line("flops  |", flops);
    pc_rate("f/cyc  |", flops, infer_t);    // achieved, end to end inference
    pc_rate("f/cyc_g|", flops, gemm_k);     // achieved, do_k_tile loop only
    pc_line("f/cyc_p|", PEAK_FLOPS_CYCLE);  // peak the array can sustain
}
#else
#define PC_LAYER(n)      ((void)0)
#define TIME(acc, stmt)  do { stmt; } while (0)
#define TIME_BEG(t)      ((void)0)
#define TIME_END(acc, t) ((void)0)
#define pc_report()      ((void)0)
#endif

// =======================================
// MAC accelerator instructions
// =======================================
// Encoded as CUSTOM1=0x2b, CUSTOM2=0x5b, funct3=0 unless noted
//   VMAC64(N,ptr,off)          VMAC64   CUSTOM1   vN x weight block at ptr+off -> raw tile
//   MAC_AS(a,b)                MAC_AS   f7=0x0A   apply act scale words a,b to the tile
//   MAC_WS(a,b)                MAC_WS   f7=0x0B   apply wgt scale words a,b -> fold into bank
//   ACC_BANK(t)                ACCBANK  f7=0x0D   select accumulator bank t
//   BRAM_RD(rd,p)              BRAMRD   f7=0x0E   read the bram pair at p into register rd
//   BRAM_RD_FP4(rd,p,bias)     BRAMFP4  f7=0x08   read tile/col at p, 8 samples -> packed fp4 and seed with bf16 bias
//   VSETVLI(avl)               vsetvli  OPV=0x57  set vl=avl, e32,m1,ta,ma
//   VLE32(N,ptr)               vle32.v  0x07      load vl words at ptr -> vN
// static assert and macro used for VMAC64
// off is unsigned in the unit, the assembler only takes -2048..2047
_Static_assert(4 * (NVREG - 1) * BS <= 0xFFF, "block offset does not fit in imm12");
#define UINT12_TO_INT12(u12) (((u12) & 0x7FF) - ((u12) & 0x800))
#define VMAC64(N,ptr,off)       __asm__ volatile(".insn i 0x2b,0x0,x" #N ",%0,%c1" :: "r"(ptr), "i"(UINT12_TO_INT12(off)))
#define MAC_AS(a,b)             __asm__ volatile(".insn r 0x5b,0x0,0x0a, x0,%0,%1" :: "r"(a),"r"(b))
#define MAC_WS(a,b)             __asm__ volatile(".insn r 0x5b,0x0,0x0b, x0,%0,%1" :: "r"(a),"r"(b))
#define ACC_BANK(t)             __asm__ volatile(".insn r 0x5b,0x0,0x0d, x0,%0,x0" :: "r"(t))
#define BRAM_RD(rd,p)           __asm__ volatile(".insn r 0x5b,0x0,0x0e, %0,%1,x0" : "=r"(rd) : "r"(p))
#define BRAM_RD_FP4(rd,p,bias)  __asm__ volatile(".insn r 0x5b,0x0,0x08, %0,%1,%2" : "=r"(rd) : "r"(p), "r"(bias))
#define VSETVLI(avl)            __asm__ volatile(".insn i 0x57,0x7,x0,%0,0xD0" :: "r"(avl))
#define VLE32(N,ptr)            __asm__ volatile(".insn i 0x07,0x6,x" #N ",%0,0x20" :: "r"(ptr))

// Address payload for a bram cell: {tile[10:6], row[5:3], col[2:0]}
#define BRAM_ADDR(tile, row, col)  (((uint32_t)(tile) << 6) | ((uint32_t)(row) << 3) | (col))

// Read the accumulator bram pair {row+1,row} at (tile,row,col) into a register
static inline uint32_t bram_rd(uint32_t tile, uint32_t row, uint32_t col) {
    uint32_t result;
    BRAM_RD(result, BRAM_ADDR(tile, row, col));
    return result;
}

// One K-block reduction
static inline __attribute__((always_inline))
void do_k_tile(int vreg, const uint32_t *As, const uint32_t *Ws, const uint32_t *weights) {
    switch (vreg) {
        case  0: VMAC64(0, weights, 4 * 0 * BS); break;
        case  1: VMAC64(1, weights, 4 * 1 * BS); break;
        case  2: VMAC64(2, weights, 4 * 2 * BS); break;
        case  3: VMAC64(3, weights, 4 * 3 * BS); break;
        case  4: VMAC64(4, weights, 4 * 4 * BS); break;
        case  5: VMAC64(5, weights, 4 * 5 * BS); break;
        case  6: VMAC64(6, weights, 4 * 6 * BS); break;
        case  7: VMAC64(7, weights, 4 * 7 * BS); break;
        case  8: VMAC64(8, weights, 4 * 8 * BS); break;
        case  9: VMAC64(9, weights, 4 * 9 * BS); break;
        case 10: VMAC64(10, weights, 4 * 10 * BS); break;
        case 11: VMAC64(11, weights, 4 * 11 * BS); break;
        case 12: VMAC64(12, weights, 4 * 12 * BS); break;
        case 13: VMAC64(13, weights, 4 * 13 * BS); break;
        case 14: VMAC64(14, weights, 4 * 14 * BS); break;
        case 15: VMAC64(15, weights, 4 * 15 * BS); break;
        case 16: VMAC64(16, weights, 4 * 16 * BS); break;
        case 17: VMAC64(17, weights, 4 * 17 * BS); break;
        case 18: VMAC64(18, weights, 4 * 18 * BS); break;
        case 19: VMAC64(19, weights, 4 * 19 * BS); break;
        case 20: VMAC64(20, weights, 4 * 20 * BS); break;
        case 21: VMAC64(21, weights, 4 * 21 * BS); break;
        case 22: VMAC64(22, weights, 4 * 22 * BS); break;
        case 23: VMAC64(23, weights, 4 * 23 * BS); break;
        case 24: VMAC64(24, weights, 4 * 24 * BS); break;
        case 25: VMAC64(25, weights, 4 * 25 * BS); break;
        case 26: VMAC64(26, weights, 4 * 26 * BS); break;
        case 27: VMAC64(27, weights, 4 * 27 * BS); break;
        case 28: VMAC64(28, weights, 4 * 28 * BS); break;
        case 29: VMAC64(29, weights, 4 * 29 * BS); break;
        case 30: VMAC64(30, weights, 4 * 30 * BS); break;
        case 31: VMAC64(31, weights, 4 * 31 * BS); break;
        default: break;
    }
    MAC_AS(As[0], As[1]);   // apply activation scales
    MAC_WS(Ws[0], Ws[1]);   // apply weight scales
}

// Load one activation block (32 words) into vector register vreg
static inline __attribute__((always_inline))
void load_vreg(int vreg, const uint32_t *ptr) {
    switch (vreg) {
        case  0: VLE32(0,  ptr); break;
        case  1: VLE32(1,  ptr); break;
        case  2: VLE32(2,  ptr); break;
        case  3: VLE32(3,  ptr); break;
        case  4: VLE32(4,  ptr); break;
        case  5: VLE32(5,  ptr); break;
        case  6: VLE32(6,  ptr); break;
        case  7: VLE32(7,  ptr); break;
        case  8: VLE32(8,  ptr); break;
        case  9: VLE32(9,  ptr); break;
        case 10: VLE32(10, ptr); break;
        case 11: VLE32(11, ptr); break;
        case 12: VLE32(12, ptr); break;
        case 13: VLE32(13, ptr); break;
        case 14: VLE32(14, ptr); break;
        case 15: VLE32(15, ptr); break;
        case 16: VLE32(16, ptr); break;
        case 17: VLE32(17, ptr); break;
        case 18: VLE32(18, ptr); break;
        case 19: VLE32(19, ptr); break;
        case 20: VLE32(20, ptr); break;
        case 21: VLE32(21, ptr); break;
        case 22: VLE32(22, ptr); break;
        case 23: VLE32(23, ptr); break;
        case 24: VLE32(24, ptr); break;
        case 25: VLE32(25, ptr); break;
        case 26: VLE32(26, ptr); break;
        case 27: VLE32(27, ptr); break;
        case 28: VLE32(28, ptr); break;
        case 29: VLE32(29, ptr); break;
        case 30: VLE32(30, ptr); break;
        case 31: VLE32(31, ptr); break;
        default: break;
    }
}

// =======================================
// Reading the accumulator banks back out
// =======================================

// Map a bf16 to an unsigned value that compares in the same order
static inline uint16_t bf16_ordered(uint16_t bf16) {
    uint16_t sgn = bf16 & 0x8000;
    uint16_t msk = 0x7FFF + sgn + ((sgn>>15)^1);
    return (uint16_t)bf16 ^ msk;
}

// Predict each sample: the output neuron with the largest logit, read straight from the banks
static void argmax(int *predictions, int WH) {
    int tiles = (WH + TT - 1) / TT;

    uint16_t best[BATCH];
    for (int sample = 0; sample < BATCH; sample++) {
        predictions[sample] = 0;
        best[sample] = 0;
    }

    for (int tile = 0; tile < tiles; tile++) {
        int neuron0 = tile * TT;
        int cols = (WH - neuron0 < TT) ? (WH - neuron0) : TT;   // real neurons in this bank
        for (int row = 0; row < TT/2; row++) {
            uint16_t best1 = best[2*row];
            uint16_t best2 = best[2*row+1];
            int      pred1 = predictions[2*row];
            int      pred2 = predictions[2*row+1];
            for (int col = 0; col < cols; col++) {
                int neuron = neuron0 + col;                              // column picks the neuron
                uint32_t pair = bram_rd(tile, 2 * row, col);
                uint16_t lo = bf16_ordered((uint16_t)(pair & 0xFFFF));   // sample = 2*row
                uint16_t hi = bf16_ordered((uint16_t)(pair >> 16)   );   // sample = 2*row + 1
                if (lo > best1) {
                    best1 = lo;
                    pred1 = neuron;
                }
                if (hi > best2) {
                    best2 = hi;
                    pred2 = neuron;
                }
            }
            best[2*row]      = best1;
            best[2*row+1]    = best2;
            predictions[2*row]      = pred1;
            predictions[2*row+1]    = pred2;
        }
    }
}


// Read banks back out as FP4 activations
static void readout_fp4(uint32_t *z, int WH, int WH_next, const uint32_t* bias_packed) {
    assert(WH_next <= WH);              // readout only sees current layer size, breaks if next layer is larger
    int tiles = (WH + TT - 1) / TT;             // no. of tiles used in readout
    int bias_tiles = (WH_next + TT - 1) / TT;   // no. of tiles used in bias seeding
    for (int tile = 0; tile < tiles; tile++) {
        int idx = tile;             // assuming that J is 0
        int cols = WH - idx*TT;     // real neurons in this tile
        if (cols > TT) cols = TT;
            
        int neuron0 = tile * TT;    // first neuron of this bank
        #pragma GCC unroll 4
        for (int c = 0; c < (cols + 1)/2; c++) {   // column-pairs holding a neuron (0,1,2,3)
            // load it once (r=0) and seed every row with the same value
            uint32_t word = 0;
            if (tile < bias_tiles) 
                word = bias_packed[idx*32 + c*8];
            uint32_t lo = word;            // even col 2c
            uint32_t hi = word >> 16;      // odd  col 2c+1

            BRAM_RD_FP4(z[neuron0+2*c], BRAM_ADDR(tile, 0, 2*c), lo);       // read out even neurons in fp4, seed with bf16 bias
            BRAM_RD_FP4(z[neuron0+2*c+1], BRAM_ADDR(tile, 0, 2*c+1), hi);   // odd neurons, seed with bf16 bias
        }
    }
}

//[PROF]

#ifdef PAIR_PROFILE

#define PAIR_BLOCK 16
#define PAIR_COUNT  (PAIR_BLOCK / 2)

typedef struct {
    uint64_t exp_pair[4][4];  // ordered (EA, EB)
    uint64_t zero_pairs;      // at least one operand is zero
    uint64_t nonzero_pairs;
} pair_profile_t;

static void pair_profile_init(pair_profile_t *p)
{
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++)
            p->exp_pair[a][b] = 0;

    p->zero_pairs = 0;
    p->nonzero_pairs = 0;
}

// Preserve sign and mantissa; replace only the 2-bit exponent.
static inline uint8_t fp4_set_exp(uint8_t code, uint8_t exp)
{
    return (uint8_t)((code & 0x9) | ((exp & 0x3) << 1));
}

// Return 1 if the pair was rewritten.
// weight_mode = 0: activation mapping
// weight_mode = 1: weight mapping
static inline int pair_compress_codes(uint8_t *a,
                                      uint8_t *b,
                                      int weight_mode)
{
/*
    // Do not modify pairs containing zero.
    if (((*a & 0x7) == 0) || ((*b & 0x7) == 0))
        return 0;
*/

    uint8_t ea = (*a >> 1) & 0x3;
    uint8_t eb = (*b >> 1) & 0x3;

    if (!weight_mode) {
        // ACTIVATIONS:
        // (Emax,d) = 000: (00,00) -> (00,01)
        if (ea == 0 && eb == 0) {
            eb = 1;
        }
        // (Emax,d) = 113: (00,11) -> (01,11)
        else if (ea == 0 && eb == 3) {
            ea = 1;
        }
        // (Emax,d) = 113: (11,00) -> (11,01)
        else if (ea == 3 && eb == 0) {
            eb = 1;
        }
        else {
            return 0;
        }
    }
    else {
        // WEIGHTS:
        // (Emax,d) = 110: (11,11) -> (10,11)
        if (ea == 3 && eb == 3) {
            ea = 2;
        }
        // (Emax,d) = 113: (00,11) -> (01,11)
        else if (ea == 0 && eb == 3) {
            ea = 1;
        }
        // (Emax,d) = 113: (11,00) -> (11,01)
        else if (ea == 3 && eb == 0) {
            eb = 1;
        }
        else {
            return 0;
        }
    }

    *a = fp4_set_exp(*a, ea);
    *b = fp4_set_exp(*b, eb);

    return 1;
}

static inline void pair_profile_add(pair_profile_t *p,
                                    uint8_t a, uint8_t b)
{
/*
    // FP4 zero: magnitude bits are both zero.
    if ((a & 0x7) == 0 || (b & 0x7) == 0) {
        p->zero_pairs++;
        return;
    }
*/

    uint8_t ea = (a >> 1) & 0x3;
    uint8_t eb = (b >> 1) & 0x3;

    p->exp_pair[ea][eb]++;
    p->nonzero_pairs++;
}

static void pair_print_percent(uint64_t count, uint64_t total)
{
    uint32_t pct = total ? (uint32_t)((count * 10000) / total) : 0;

    putdec(pct / 100);
    putchar_uart('.');
    if ((pct % 100) < 10) putchar_uart('0');
    putdec(pct % 100);
    putchar_uart('%');
}

// Print ordered exponent-pair composition.
static void pair_profile_report(const char *name,
                                const pair_profile_t *p,
                                int batch_number,
                                uint64_t rewritten_pairs)
{
    print_str("\n[PAIR PROFILE] BATCH ");
    putdec(batch_number);
    print_str(" ");
    print_str(name);
    print_str("\n");

    print_str("Rewritten pairs: ");
    putdec64(rewritten_pairs);
    print_str("\n");

    print_str("Nonzero pairs: ");
    putdec64(p->nonzero_pairs);
    print_str("\nZero-containing pairs: ");
    putdec64(p->zero_pairs);
    print_str("\n");

    print_str("       EB=00    EB=01    EB=10    EB=11\n");

    for (int ea = 0; ea < 4; ea++) {
        print_str("EA=");
        putdec(ea);
        print_str("  ");

        for (int eb = 0; eb < 4; eb++) {
            pair_print_percent(p->exp_pair[ea][eb],
                               p->nonzero_pairs);
            print_str("  ");
        }
        print_str("\n");
    }

    // Distribution by absolute exponent difference.
    uint64_t diff[4] = {0, 0, 0, 0};

    for (int ea = 0; ea < 4; ea++) {
        for (int eb = 0; eb < 4; eb++) {
            int d = (ea > eb) ? (ea - eb) : (eb - ea);
            diff[d] += p->exp_pair[ea][eb];
        }
    }

    print_str("\nExponent difference distribution:\n");
    for (int d = 0; d < 4; d++) {
        print_str("d=");
        putdec(d);
        print_str(": ");
        pair_print_percent(diff[d], p->nonzero_pairs);
        print_str("  count=");
        putdec64(diff[d]);
        print_str("\n");
    }

    // Candidate 3-bit codebook composition.
    // 000-011: d=0
    // 100-110: d=1
    // 111:     d>=2 (large correction)
    uint64_t code_count[8] = {0};

    for (int ea = 0; ea < 4; ea++) {
        for (int eb = 0; eb < 4; eb++) {
            int lo = (ea < eb) ? ea : eb;
            int hi = (ea > eb) ? ea : eb;
            int d = hi - lo;

            if (d == 0) {
                code_count[lo]++;
            } else if (d == 1) {
                // The three d=1 states, independent of orientation.
                code_count[4 + lo]++;
            } else {
                code_count[7]++;
            }

            // Count each ordered pair exactly once.
            // For d=1, both orientations share the same candidate code.
        }
    }
}

#endif


#ifdef PAIR_PROFILE

#define FP4_PER_WORD 8

static void profile_activations(const char *name,
                                uint32_t *A,
                                int KK,
                                int batch_number,
                                int compress)
{
    pair_profile_t p;
    pair_profile_init(&p);

    uint64_t rewritten = 0;
    const int WORDS_PER_BLOCK = PAIR_BLOCK / FP4_PER_WORD;

    for (int block = 0; block < KK; block += WORDS_PER_BLOCK) {
        for (int word = 0; word < WORDS_PER_BLOCK; word++) {
            int k = block + word;

            if (k >= KK)
                continue;

            for (int pair = 0; pair < FP4_PER_WORD / 2; pair++) {
                int s0 = 2 * pair;
                int s1 = s0 + 1;

                uint8_t a = (A[k] >> (4 * s0)) & 0xF;
                uint8_t b = (A[k] >> (4 * s1)) & 0xF;

                pair_profile_add(&p, a, b);

                if (compress && pair_compress_codes(&a, &b, 0)) {
                    uint32_t mask =
                        (0xFu << (4 * s0)) |
                        (0xFu << (4 * s1));

                    A[k] = (A[k] & ~mask) |
                           ((uint32_t)a << (4 * s0)) |
                           ((uint32_t)b << (4 * s1));

                    rewritten++;
                }

                //pair_profile_add(&p, a, b);
            }
        }
    }

    pair_profile_report(name, &p, batch_number, rewritten);
}

#endif


#ifdef PAIR_PROFILE

static void profile_weights(const char *name,
                            uint32_t *W,
                            int KK,
                            int WH,
                            int compress)
{
    pair_profile_t p;
    pair_profile_init(&p);

    uint64_t rewritten = 0;
    int num_blocks = (KK + BS - 1) / BS;
    int num_tiles  = (WH + TT - 1) / TT;

    // Outer J-strip traversal matching GEMM
    for (int J = 0; J < WH; J += NTILES * TT) {
        int tiles_this_strip = num_tiles - J / TT;
        if (tiles_this_strip > NTILES)
            tiles_this_strip = NTILES;

        // K-chunk traversal matching GEMM
        for (int K = 0; K < KK; K += NVREG * BS) {
            int blk0 = K / BS;
            int blks = num_blocks - blk0;
            if (blks > NVREG)
                blks = NVREG;

            uint32_t *weights = &W[(K / BS) * BS];

            // Tile loop matching GEMM
            for (int T = 0; T < tiles_this_strip; T++) {
                int tile = J / TT + T;

                int cols = WH - tile * TT;
                if (cols > TT)
                    cols = TT;

                // Vector register / K-block loop matching GEMM
                for (int vreg = 0; vreg < blks; vreg++) {
                    uint32_t *block_w = weights + vreg * BS;
                    int kbase = K + vreg * BS;

                    for (int col = 0; col < cols; col++) {
                        for (int ptr_k = 0; ptr_k < BS; ptr_k += 2) {
                            int k0 = ptr_k;
                            int k1 = ptr_k + 1;

                            if (kbase + k1 >= KK)
                                continue;

                            uint8_t a = (block_w[k0] >> (4 * col)) & 0xF;
                            uint8_t b = (block_w[k1] >> (4 * col)) & 0xF;

                            pair_profile_add(&p, a, b);

                            if (compress && pair_compress_codes(&a, &b, 1)) {
                                uint32_t mask = 0xFu << (4 * col);

                                uint32_t *word0 = &block_w[k0];
                                uint32_t *word1 = &block_w[k1];

                                *word0 = (*word0 & ~mask) |
                                         ((uint32_t)a << (4 * col));

                                *word1 = (*word1 & ~mask) |
                                         ((uint32_t)b << (4 * col));

                                rewritten++;
                            }

                            //pair_profile_add(&p, a, b);
                        }
                    }
                }

                weights += num_blocks * BS;
            }
        }
    }

    pair_profile_report(name, &p, 1, rewritten);
}

#endif


//[end]


// =======================================
// GEMM
// =======================================
// One layer of the MLP: C = A*W + bias, summed in accumulator banks and read out after each layer
//   A : activations, AV rows x KK columns (KK = this layer's input size)
//   W : weights,     KK rows x WH columns (WH = this layer's output size)
//   C : outputs,     AV rows x WH columns, computed in TTxTT tiles across the banks
// AV is the batch of 8 samples, assume one column strip (WH <= NTILES*TT)
void gemm(const uint32_t* A, const uint32_t* W, int KK, int WH, 
    const uint32_t* wscale_words, const uint32_t* ascale_words) {
    // Results are left in the BRAM accumulator
    assert( AV == TT );          // else BRAM must be saved after each I iteration
    assert( WH <= NTILES * TT ); // else BRAM must be saved+reloaded each J iteration

    int num_blocks = (KK + BS - 1) / BS;   // K split into BS-blocks
    int num_tiles  = (WH + TT - 1) / TT;   // columns split into TTxTT tiles
    
    // For all rows
    for (int I = 0; I < AV; I += TT) 
    {   
        // For all columns 
        for (int J = 0; J < WH; J += (NTILES*TT))
        {
            // tiles in this NTILES*TT column strip
            int tiles_this_strip = num_tiles - J/TT;
            if (tiles_this_strip > NTILES) tiles_this_strip = NTILES;

            for (int K = 0; K < KK; K += (NVREG*BS))
            {
                // number of K-blocks reduced in this chunk
                int blk0 = K / BS;
                int blks = num_blocks - blk0;
                if (blks > NVREG) blks = NVREG;

                // Activation loads into v0..v(blks-1)
                TIME_BEG(_lt);
                VSETVLI(32);
                #pragma GCC unroll 32
                for (int vreg = 0; vreg < blks; vreg++)
                    load_vreg(vreg, &A[K + vreg*BS]);
                TIME_END(pc_load[pc_layer], _lt);

                const uint32_t* Ascales = &ascale_words[(K/BS)*2];
                const uint32_t* Wscales = &wscale_words[(K/BS)*2];
                const uint32_t* weights = &W[(K/BS)*BS];

                // Tile Loop
                for (int T = 0; T < tiles_this_strip; T++)
                {
                    ACC_BANK(T);   // target accumulator bank T for this tile

                    TIME_BEG(_kt);
                    #pragma GCC unroll 32
                    for (int vreg = 0; vreg < blks; vreg++) {
                        do_k_tile(vreg, &Ascales[vreg*2], &Wscales[vreg*2], weights);
                    }
                    TIME_END(pc_ktile[pc_layer], _kt);

                    Wscales += num_blocks*2;   // step one tile-column of Wscales  [(KK/BS)*TT bytes]
                    weights += num_blocks*BS;  // step one tile-column of weights  [(KK/2)*TT bytes]
                }
            }
            // result now lives in the accumulator banks
        }
    }
}

// Run the forward pass on a batch of samples
void inference_batch(uint32_t* inputs,
                     const uint32_t* ascale_in,
                     int* predictions,
                     uint32_t* W1,
                     uint32_t* W2,
                     uint32_t* W3,
                     int batch_number) {
    // Hidden layers read out to FP4 activations; the final layer argmaxes straight from the banks
    static uint32_t z1_packed[L1_DIM], z2_packed[L2_DIM];

    PC_LAYER(0);

    TIME(pc_qact[0],
         readout_fp4(z1_packed, L1_DIM, L1_DIM, bias1_packed));

#ifdef PAIR_PROFILE
    profile_activations("L1 ACTIVATIONS", inputs, IN_DIM,
                        batch_number, 1);

    if (batch_number == 1) {
        profile_weights("L1 WEIGHTS", W1, IN_DIM, L1_DIM, 1);
    }
#endif

    TIME(pc_gemm[0],
         gemm(inputs, W1, IN_DIM, L1_DIM, wscale1, ascale_in));

    TIME(pc_qact[0],
         readout_fp4(z1_packed, L1_DIM, L2_DIM, bias2_packed));

    PC_LAYER(1);

#ifdef PAIR_PROFILE
    profile_activations("L2 ACTIVATIONS", z1_packed, L1_DIM,
                        batch_number, 1);

    if (batch_number == 1) {
        profile_weights("L2 WEIGHTS", W2, L1_DIM, L2_DIM, 1);
    }
#endif

    TIME(pc_gemm[1],
         gemm(z1_packed, W2, L1_DIM, L2_DIM, wscale2, ascale2));

    TIME(pc_qact[1],
         readout_fp4(z2_packed, L2_DIM, OUT_DIM, bias3_packed));

    PC_LAYER(2);

#ifdef PAIR_PROFILE
    profile_activations("L3 ACTIVATIONS", z2_packed, L2_DIM,
                        batch_number, 1);

    if (batch_number == 1) {
        profile_weights("L3 WEIGHTS", W3, L2_DIM, OUT_DIM, 1);
    }
#endif

    TIME(pc_gemm[2],
         gemm(z2_packed, W3, L2_DIM, OUT_DIM, wscale3, ascale3));

    TIME(pc_argmax, argmax(predictions, OUT_DIM));
}

int main(void) {
    // the readout lut is built for one shift, so the header has to agree
    assert(rdout_shift[1] == RDOUT_SHIFT && rdout_shift[2] == RDOUT_SHIFT);
    // Only continue if the number of scales in the TB and here match
    assert(*IMG_NUM_SCALES == NUM_SCALES);
    // Store one word per pixel for all batch lanes
    static uint32_t image_packed[IN_DIM];
    // FP4 codes for the whole batch, 8 codes per word
    static uint32_t stage_buf[BATCH][IN_REAL/8];
    // Layer 1 input scales for the whole batch, one byte per scale block
    static uint8_t scale_buf[BATCH][NUM_SCALES];
    // Layer 1 input scales
    static uint32_t ascale1_batch[2 * NUM_SCALES];

    int predictions[BATCH];

    // Print prediction, truth, and reference format
    #ifdef PTM_PRINTS
    print_str("P|T|M\n");
    #endif

    // Count correct predictions
    int correct = 0;    
    // Count matches against reference
    int match = 0;

    TIME_BEG(_rt0);

    for (int s = 0; s < N_SAMPLES; s += BATCH) {
        int n = N_SAMPLES - s;
        if (n > BATCH) n = BATCH;
        int truth[BATCH], ref[BATCH];

        // Pack the batch of images
        TIME(pc_imgq, {
            // Drain the staging buffer, one image at a time
            for (int j = 0; j < n; j++) {
                image_load(s + j);
                const volatile uint32_t *stage32 = (const volatile uint32_t *)IMG_STAGE;
                for (int w = 0; w < IN_REAL/8; w++) {   // 98 words = 784 pixel codes
                    stage_buf[j][w] = stage32[w];
                }
                truth[j] = *IMG_LABEL;
                ref[j]   = *IMG_PRED;

                for (int b = 0; b < NUM_SCALES; b++) {
                    scale_buf[j][b] = IMG_SCALES[b];
                }
            }

            // One pair of words per scale block, one byte per image
            for (int b = 0; b < NUM_SCALES; b++) {
                uint32_t scale_lo = 0;   // images 0..3
                uint32_t scale_hi = 0;   // images 4..7
                for (int j = 0; j < n; j++) {
                    uint32_t scale = scale_buf[j][b];
                    if (j < 4)
                        scale_lo |= scale << (8 * j);
                    else
                        scale_hi |= scale << (8 * (j - 4));
                }
                ascale1_batch[2 * b]     = scale_lo;
                ascale1_batch[2 * b + 1] = scale_hi;
            }

            // One staged word holds 8 codes, code k in bits 4k..4k+3
            for (int w = 0; w < IN_REAL/8; w++) {
                for (int k = 0; k < 8; k++) {
                    uint32_t a = 0;
                    for (int j = 0; j < n; j++) {
                        uint32_t code = (stage_buf[j][w] >> (4 * k)) & 0xF;
                        a |= code << (4 * j);
                    }
                    image_packed[w * 8 + k] = a;
                }
            }

            // K pad tail (784..799) carries no pixels and must read as zero
            for (int p = IN_REAL; p < IN_DIM; p++) {
                image_packed[p] = 0;
            }
        });

        int batch_number = (s / BATCH) + 1;

        inference_batch(image_packed,
                        ascale1_batch,
                        predictions,
                        w1_fp4,
                        w2_fp4,
                        w3_fp4,
                        batch_number);

        for (int j = 0; j < n; j++) {
            if (predictions[j] == truth[j]) correct++;
            if (predictions[j] == ref[j])   match++;
            
            #ifdef PTM_PRINTS
            putdec(predictions[j]);
            print_str("|");
            putdec(truth[j]);
            print_str("|");
            putdec(ref[j]);
            print_str("\n");
            #endif
        }
    }

    TIME_END(pc_total, _rt0);

    print_str("\nACCURACY: ");
    putdec(correct);
    putchar_uart('/');
    putdec(N_SAMPLES);
    print_str("\n");

    print_str("MATCH (vs mptorch): ");
    putdec(match);
    putchar_uart('/');
    putdec(N_SAMPLES);
    print_str("\n");

    pc_report();

    //kill_simulation();

    return 0;
}
