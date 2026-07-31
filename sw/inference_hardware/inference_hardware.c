// MLP inference on CVE2 with GEN3 spec

#include <assert.h>
#include "../headers/weights_blk32_pkgUINT32_scaleE8M0.h"

// Total number of samples to process
#define N_SAMPLES 80

// Memory addresses for test data loading
#define IMG_LOAD  ((volatile unsigned int  *) 0xFFFF0010)
#define IMG_LABEL ((volatile unsigned int  *) 0xFFFF0014)
#define IMG_PRED  ((volatile unsigned int  *) 0xFFFF0018)
#define IMG_STAGE ((volatile unsigned char *) 0x80070000)

// Network dimensions for the MLP layers
#define IN_REAL   784             // real MNIST pixels
#define IN_DIM    800             // pad K only up to a BS multiple: 25*32=800
#define L1_DIM    128
#define L2_DIM     96
#define OUT_DIM    10

#define BATCH       8             // Parallel batch size for MNIST samples
#define TT          8             // Tile dimension for 8x8 MACs        
#define NVREG      32             // Number of vector register
#define NTILES     32             // Number of tiles that fit in BRAM
#define AV          8             // Activation matrix vertical size
//#define WH  out_dim             // Weight martix horizontal size
//#define KK   in_dim             // Shared activation/weight matrix size

#define BS       K1_STEP_HDR      // Number of K elements per inner block

// Read-out multiply: scale up by 2^RDOUT_SHIFT so [-0.5,0.5] fills the FP4 grid;
// MAC_AS undoes it via ascale = 2^-(bias+2) (should be bias+1 but +2 performs better)
#define RDOUT_SHIFT  3                     // x8 (exact = 2, x4)
#define RDOUT_OFF    (125 - RDOUT_SHIFT)   // LUT exponent offset (122)

// Enable performance counters
#define PERF_COUNTERS

// Enable intermediate UART prints (P|T|M results)
#define PTM_PRINTS

void __assert_func(const char *f,int l,const char *fn,const char *e){
    (void)f;(void)l;(void)fn;(void)e; __builtin_trap();
}

//[stev]
// Define the simulator termination MMIO register
static volatile uint32_t *const DONE_MMIO = (volatile uint32_t *)0xFFFF0000u;

static inline void kill_simulation(void) {
    *DONE_MMIO = 0xFF; // Signal simulator testbench to halt
    while (1);         // Catch pipeline flush before termination
}
//[end]

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

// Helper to print raw bfloat16 hex values
static void puthex16(uint16_t val) {
    const char hex_chars[] = "0123456789ABCDEF";
    for (int i = 12; i >= 0; i -= 4) {
        putchar_uart(hex_chars[(val >> i) & 0xF]);
    }
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
static uint64_t pc_qact[2];    // hidden read-out + FP4 quantize (readout_fp4)
static uint64_t pc_argmax;     // final read-out + argmax
static uint64_t pc_bias[3];
static int      pc_layer;      

#define PC_LAYER(n)     (pc_layer = (n))
#define TIME(acc, stmt) do { uint32_t _t = rdcyc(); stmt; (acc) += (uint32_t)(rdcyc() - _t); } while (0)

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

static void pc_report(void) {
    print_str("\n[PERF] cycles over run\n");
    pc_line("imgq   |", pc_imgq);       // image load and quant

    pc_line("gemm_1 |", pc_gemm[0]);    // gemm compute per layer (incl. bias seed)
    pc_line("gemm_2 |", pc_gemm[1]);
    pc_line("gemm_3 |", pc_gemm[2]);

    pc_line("qact_1 |", pc_qact[0]);    // hidden read-out from banks + FP4 quantize
    pc_line("qact_2 |", pc_qact[1]);

    pc_line("argmax |", pc_argmax);     // final read-out from banks + argmax

    pc_line("bias_1 |", pc_bias[0]);    // bias tile seed (subset of gemm)
    pc_line("bias_2 |", pc_bias[1]);
    pc_line("bias_3 |", pc_bias[2]);

    print_str("\n[PERF] total cycles over run\n");
    // gemm compute (incl. bias) plus every read-out
    pc_line("gemm_T |",
        pc_gemm[0] + pc_gemm[1] + pc_gemm[2] +
        pc_qact[0] + pc_qact[1] + pc_argmax);
    // datapath only: gemm compute minus the bias seed
    pc_line("gemm_X |",
        pc_gemm[0] + pc_gemm[1] + pc_gemm[2] -
        (pc_bias[0] + pc_bias[1] + pc_bias[2]));
    // total read-out + FP4 quantize
    pc_line("qact_T |", pc_qact[0] + pc_qact[1]);
    // total bias seed
    pc_line("bias_T |", pc_bias[0] + pc_bias[1] + pc_bias[2]);
}
#else
#define PC_LAYER(n)     ((void)0)
#define TIME(acc, stmt) do { stmt; } while (0)
#define pc_report()     ((void)0)
#endif

// =======================================
// MAC accelerator instructions
// =======================================
// Encodings mirror matmul8_vec.S. CUSTOM1=0x2b, CUSTOM2=0x5b, funct3=0 unless noted
//   VMAC64(N,ptr)   VMAC64   CUSTOM1   vN x weight block at ptr -> raw tile
//   MAC_AS(a,b)     MAC_AS   f7=0x0A   apply act scale words a,b to the tile
//   MAC_WS(a,b)     MAC_WS   f7=0x0B   apply wgt scale words a,b -> fold into bank
//   MAC_BIAS(p,v)   MACBIAS  f7=0x0C   seed bf16 v into the cell addressed by p
//   ACC_BANK(t)     ACCBANK  f7=0x0D   select accumulator bank t
//   BRAM_RD(rd,p)   BRAMRD   f7=0x0E   read the bram pair at p into register rd
//   VSETVLI(avl)    vsetvli  OPV=0x57  set vl=avl, e32,m1,ta,ma
//   VLE32(N,ptr)    vle32.v  0x07      load 32 words at ptr -> vN
#define MAC_ZZ()       __asm__ volatile(".insn r 0x5b,0x0,0x00, x0,x0,x0") // REMOVE LATER
#define VMAC64(N,ptr)  __asm__ volatile(".insn i 0x2b,0x0,x" #N ",%0,0" :: "r"(ptr))
#define MAC_AS(a,b)    __asm__ volatile(".insn r 0x5b,0x0,0x0a, x0,%0,%1" :: "r"(a),"r"(b))
#define MAC_WS(a,b)    __asm__ volatile(".insn r 0x5b,0x0,0x0b, x0,%0,%1" :: "r"(a),"r"(b))
#define MAC_BIAS(p,v)  __asm__ volatile(".insn r 0x5b,0x0,0x0c, x0,%0,%1" :: "r"(p),"r"(v))
#define ACC_BANK(t)    __asm__ volatile(".insn r 0x5b,0x0,0x0d, x0,%0,x0" :: "r"(t))
#define BRAM_RD(rd,p)  __asm__ volatile(".insn r 0x5b,0x0,0x0e, %0,%1,x0" : "=r"(rd) : "r"(p))
#define VSETVLI(avl)   __asm__ volatile(".insn i 0x57,0x7,x0,%0,0xD0" :: "r"(avl))
#define VLE32(N,ptr)   __asm__ volatile(".insn i 0x07,0x6,x" #N ",%0,0x20" :: "r"(ptr))

// Address payload for a bram cell: {tile[10:6], row[5:3], col[2:0]}
#define BRAM_ADDR(tile, row, col)  (((uint32_t)(tile) << 6) | ((uint32_t)(row) << 3) | (col))

// Seed one accumulator cell (tile,row,col) with a bf16 bias value
static inline void mac_bias(uint8_t tile, uint8_t row, uint8_t column, uint16_t bf16) {
    MAC_BIAS(BRAM_ADDR(tile, row, column), (uint32_t)bf16);
}

// Read the accumulator bram pair {row+1,row} at (tile,row,col) into a register
static inline uint32_t bram_rd(uint32_t tile, uint32_t row, uint32_t col) {
    uint32_t result;
    BRAM_RD(result, BRAM_ADDR(tile, row, col));
    return result;
}

// =======================================
// DEBUG CHECKPOINT FUNCTION
// =======================================
static void dump_bram_checkpoint(int max_tiles) {
    print_str("\n=== BRAM STATE CHECKPOINT DUMP ===\n");
    for (int t = 0; t < max_tiles; t++) {
        print_str("--- TILE ");
        putdec(t);
        print_str(" ---\n");
        for (int r_pair = 0; r_pair < TT / 2; r_pair++) {
            print_str("RowPair ");
            putdec(r_pair);
            print_str(": ");
            for (int c = 0; c < TT; c++) {
                uint32_t pair = bram_rd(t, 2 * r_pair, c);
                uint16_t lo = (uint16_t)(pair & 0xFFFF);
                uint16_t hi = (uint16_t)(pair >> 16);

                print_str("[");
                puthex16(lo);
                print_str(",");
                puthex16(hi);
                print_str("] ");
            }
            print_str("\n");
        }
    }
    print_str("=== END BRAM CHECKPOINT DUMP ===\n\n");
}


// One K-block reduction
static inline __attribute__((always_inline))
void do_k_tile(int vreg, const uint32_t *As, const uint32_t *Ws, const uint32_t *weights) {
    //MAC_ZZ(); // REMOVE LATER
    switch (vreg) {
        case  0: VMAC64(0,  weights +  0*BS); break;
        case  1: VMAC64(1,  weights +  1*BS); break;
        case  2: VMAC64(2,  weights +  2*BS); break;
        case  3: VMAC64(3,  weights +  3*BS); break;
        case  4: VMAC64(4,  weights +  4*BS); break;
        case  5: VMAC64(5,  weights +  5*BS); break;
        case  6: VMAC64(6,  weights +  6*BS); break;
        case  7: VMAC64(7,  weights +  7*BS); break;
        case  8: VMAC64(8,  weights +  8*BS); break;
        case  9: VMAC64(9,  weights +  9*BS); break;
        case 10: VMAC64(10, weights + 10*BS); break;
        case 11: VMAC64(11, weights + 11*BS); break;
        case 12: VMAC64(12, weights + 12*BS); break;
        case 13: VMAC64(13, weights + 13*BS); break;
        case 14: VMAC64(14, weights + 14*BS); break;
        case 15: VMAC64(15, weights + 15*BS); break;
        case 16: VMAC64(16, weights + 16*BS); break;
        case 17: VMAC64(17, weights + 17*BS); break;
        case 18: VMAC64(18, weights + 18*BS); break;
        case 19: VMAC64(19, weights + 19*BS); break;
        case 20: VMAC64(20, weights + 20*BS); break;
        case 21: VMAC64(21, weights + 21*BS); break;
        case 22: VMAC64(22, weights + 22*BS); break;
        case 23: VMAC64(23, weights + 23*BS); break;
        case 24: VMAC64(24, weights + 24*BS); break;
        case 25: VMAC64(25, weights + 25*BS); break;
        case 26: VMAC64(26, weights + 26*BS); break;
        case 27: VMAC64(27, weights + 27*BS); break;
        case 28: VMAC64(28, weights + 28*BS); break;
        case 29: VMAC64(29, weights + 29*BS); break;
        case 30: VMAC64(30, weights + 30*BS); break;
        case 31: VMAC64(31, weights + 31*BS); break;
        default: break;
    }
    MAC_AS(As[0], As[1]);   // apply activation scales
    MAC_WS(Ws[0], Ws[1]);   // apply weight scales

    // [rbs] one-fold BRAM-transaction capture: print a fold marker (readback disabled to
    // avoid cluttering the BRAM_DEBUG log with BRAM_RD reads), halt after fold 3 (first corrupt fold).
    // static int fold_dbg = 0;
    // if (fold_dbg < 8) {
    //     print_str("\n>>> after fold "); putdec((uint32_t)fold_dbg);
    //     print_str(" (vreg "); putdec((uint32_t)vreg); print_str(")\n");
    //     dump_bram_checkpoint(1);   // tile 0 bank state, to compare vs robert-sw
    // }
    // if (++fold_dbg >= 8) kill_simulation();
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
// FP4 Quantization
// =======================================
static const uint8_t fp4_mag_lut[16] = {
    0, 1, 2, 3, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7
};

// Convert input float to the nearest FP4 value
int16_t fp4_quantize(float value) {
    int sign = (value < 0.0f);
    float abs_v = sign ? -value : value;

    int idx = (int)(abs_v * 4.0f + 0.5f);
    if (idx > 15) idx = 15;
    uint8_t mag = fp4_mag_lut[idx];

    if (mag == 0) return 0;
    return (int16_t)(sign ? (0x8 | mag) : mag);
}

// =======================================
// LUT for input pixel quantization
// =======================================
// Precomputed table to convert pixel bytes to FP4
static uint8_t pix_to_fp4[256];

static void build_pix_lut(void) {
    for (int v = 0; v < 256; v++) {
        pix_to_fp4[v] = (uint8_t)(fp4_quantize((float)v / 255.0f) & 0xF);
    }
}

// =======================================
// Reading the accumulator banks back out
// =======================================

// Map a bf16 to an unsigned value that compares in the same order (bf16 is sign-magnitude).
static inline uint16_t bf16_ordered(uint16_t bf16) {
    if (bf16 & 0x8000) {
        return (uint16_t)~bf16;                    // negative: flip all bits
    }
    return (uint16_t)(bf16 | 0x8000);              // positive: set the top bit
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
            for (int col = 0; col < cols; col++) {
                int neuron = neuron0 + col;                              // column picks the neuron
                uint32_t pair = bram_rd(tile, 2 * row, col);

                uint16_t lo = bf16_ordered((uint16_t)(pair & 0xFFFF));   // sample = row
                if (lo > best[row]) {
                    best[row] = lo;
                    predictions[row] = neuron;
                }

                uint16_t hi = bf16_ordered((uint16_t)(pair >> 16));      // sample = row + TT/2
                if (hi > best[row + TT/2]) {
                    best[row + TT/2] = hi;
                    predictions[row + TT/2] = neuron;
                }
            }
        }
    }
}

// FP4 E2M1 magnitude codes:  0=0  1=0.5  2=1.0  3=1.5  4=2.0  5=3.0  6=4.0
enum { FP4_05 = 1, FP4_10 = 2, FP4_15 = 3, FP4_20 = 4, FP4_30 = 5, FP4_40 = 6 };

// Nearest-FP4 rounding grid for a value in [0,4]: index = 2 exp bits << 7 | 7 mantissa.
// fp4_from_bf16 supplies the exponent offset (the read-out multiply); this table is
// just the rounding, so it is independent of the multiply.
static uint8_t bf16_fp4_lut[128 * 4];

static void build_bf16_fp4_lut(void) {
    int code = 0;                                   // start: value below 0.25 -> 0
    for (int i = 0; i < 128 * 4; i++) {
        if (i >= (0 * 128) +  1) { code = FP4_05; } // past 0.25 -> 0.5   (0   / 0.5 boundary)
        if (i >= (1 * 128) + 64) { code = FP4_10; } // past 0.75 -> 1.0   (0.5 / 1.0 boundary)
        if (i >= (2 * 128) + 33) { code = FP4_15; } // past 1.25 -> 1.5   (1.0 / 1.5 boundary)
        if (i >= (2 * 128) + 96) { code = FP4_20; } // past 1.75 -> 2.0   (1.5 / 2.0 boundary)
        if (i >= (3 * 128) + 33) { code = FP4_30; } // past 2.5  -> 3.0   (2.0 / 3.0 boundary)
        if (i >= (3 * 128) + 96) { code = FP4_40; } // past 3.5  -> 4.0   (3.0 / 4.0 boundary)
        bf16_fp4_lut[i] = (uint8_t)code;
    }
}

// bf16 activation -> FP4 code.
static inline uint32_t fp4_from_bf16(uint16_t bf16) {
    int exp  = (bf16 >> 7) & 0xFF;
    int sign = (bf16 >> 15) & 1;
    int code;
    if (exp < RDOUT_OFF) {                 // below LUT floor -> 0
        code = 0;
    } else if (exp > 125) {                // |act| >= 0.5 (Hardtanh) -> clamp to top FP4
        code = FP4_40;
    } else {                               // scale up 2^RDOUT_SHIFT, round via LUT
        code = bf16_fp4_lut[(((exp - RDOUT_OFF) & 3) << 7) | (bf16 & 0x7F)];
    }
    if (code == 0) { return 0; }
    return sign ? (0x8u | code) : (uint32_t)code;
}

// Read a hidden layer's banks and qaunt to FP4 activations
static void readout_fp4(uint32_t *z, int WH) {
    int tiles = (WH + TT - 1) / TT;
    for (int tile = 0; tile < tiles; tile++) {
        int neuron0 = tile * TT;                   // first neuron of this bank
        for (int col = 0; col < TT; col++) {
            z[neuron0 + col] = 0;                  // clear this bank's 8 neurons
        }
        for (int row = 0; row < TT/2; row++) {

            #pragma GCC unroll 8 // TT
            for (int col = 0; col < TT; col++) {
                uint32_t pair   = bram_rd(tile, 2 * row, col);
                int neuron = neuron0 + col;        // column picks the neuron
                // the shift is 4 bits per FP4 code times the sample index
                z[neuron] |= fp4_from_bf16((uint16_t)(pair & 0xFFFF)) << (4 * row);          // sample = row
                z[neuron] |= fp4_from_bf16((uint16_t)(pair >> 16))    << (4 * (row + TT/2)); // sample = row + TT/2
            }
        }
    }
}

// =======================================
// GEMM
// =======================================
// One layer of the MLP: C = A*W + bias, summed in accumulator banks and read out after each layer
//   A : activations, AV rows x KK columns (KK = this layer's input size)
//   W : weights,     KK rows x WH columns (WH = this layer's output size)
//   C : outputs,     AV rows x WH columns, computed in TTxTT tiles across the banks
// AV is the batch of 8 samples, assume one column strip (WH <= NTILES*TT)
void gemm(const uint32_t* A, const uint32_t* W, const uint32_t* bias_packed,
          int KK, int WH, const uint32_t* wscale_words, const uint32_t* ascale_words) {
#ifdef PERF_COUNTERS
    uint32_t _gt0 = rdcyc();
#endif
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
            
            TIME(pc_bias[pc_layer], 
            // seed each bank with its TTxTT bias tile
            for (int T = 0; T < tiles_this_strip; T++) {
                int idx = J/TT + T;                    // tile index
                for (int c = 0; c < 4; c++) {          // 4 column-pairs
                    // load it once (r=0) and seed every row with the same value
                    uint32_t word = bias_packed[idx*32 + c*8];
                    uint16_t lo = (uint16_t)(word & 0xFFFF);   // even col 2c
                    uint16_t hi = (uint16_t)(word >> 16);      // odd  col 2c+1

                    #pragma GCC unroll 8 // BATCH
                    for (int r = 0; r < BATCH; r++) {  // 8 rows
                        mac_bias(T, r, 2*c,   lo);
                        mac_bias(T, r, 2*c+1, hi);
                    }
                }
            }
            );

            // [rbs] PROBE: dump the PURE bias seed (zero accumulation steps) once,
            // right after seeding and before any do_k_tile. Isolates seed-write from
            // the rewritten INT MAC array. Compare this across builds.
            static int seed_dbg = 0;
            if (seed_dbg == 0) {
                seed_dbg = 1;
                print_str("\n>>> PURE BIAS SEED (before any accumulation)\n");
                dump_bram_checkpoint(1);   // tile 0 only
            }

            for (int K = 0; K < KK; K += (NVREG*BS))
            {
                // number of K-blocks reduced in this chunk
                int blk0 = K / BS;
                int blks = num_blocks - blk0;
                if (blks > NVREG) blks = NVREG;

                // Activation loads into v0..v(blks-1)
                VSETVLI(32);
                #pragma GCC unroll 32
                for (int vreg = 0; vreg < blks; vreg++)
                    load_vreg(vreg, &A[K + vreg*BS]);

                const uint32_t* Ascales = &ascale_words[(K/BS)*2];
                const uint32_t* Wscales = &wscale_words[(K/BS)*2];
                const uint32_t* weights = &W[(K/BS)*BS];

                // Tile Loop
                for (int T = 0; T < tiles_this_strip; T++)
                {
                    ACC_BANK(T);   // target accumulator bank T for this tile

                    #pragma GCC unroll 32
                    for (int vreg = 0; vreg < blks; vreg++) {
                        do_k_tile(vreg, &Ascales[vreg*2], &Wscales[vreg*2], weights);
                    }

                    Wscales += num_blocks*2;   // step one tile-column of Wscales  [(KK/BS)*TT bytes]
                    weights += num_blocks*BS;  // step one tile-column of weights  [(KK/2)*TT bytes]
                }
            }
            // result now lives in the accumulator banks
        }
    }
#ifdef PERF_COUNTERS
    pc_gemm[pc_layer] += (uint32_t)(rdcyc() - _gt0);
#endif       
}

// Run the forward pass on a batch of samples
void inference_batch(const uint32_t* inputs, int* predictions) {
    // Hidden layers read out to FP4 activations; the final layer argmaxes straight from the banks
    static uint32_t z1_packed[L1_DIM], z2_packed[L2_DIM];

    print_str("[DEBUG] Running Layer 1 GEMM...\n");

    PC_LAYER(0);
    gemm(inputs, w1_fp4, bias1_packed, IN_DIM, L1_DIM, wscale1, ascale1);
    TIME(pc_qact[0], readout_fp4(z1_packed, L1_DIM));       // hidden layer -> FP4 activations

//[stev]
 // ==================== CHECKPOINT HERE ====================
    // Tile count for Layer 1: L1_DIM (128) / TT (8) = 16 Tiles.
//     print_str("[DEBUG] Layer 1 GEMM complete! Dumping BRAM Status...\n");
//     dump_bram_checkpoint(16);

//     print_str("[DEBUG] Terminating execution for hardware validation.\n");
//    // __builtin_trap(); // Force halt / trigger trap
// // Kill simulation cleanly via MMIO
//     kill_simulation();

//end

    PC_LAYER(1);
    gemm(z1_packed, w2_fp4, bias2_packed, L1_DIM, L2_DIM, wscale2, ascale2);
    TIME(pc_qact[1], readout_fp4(z2_packed, L2_DIM));       // hidden layer -> FP4 activations

    PC_LAYER(2);
    gemm(z2_packed, w3_fp4, bias3_packed, L2_DIM, OUT_DIM, wscale3, ascale3);
    TIME(pc_argmax, argmax(predictions, OUT_DIM));  // final layer -> argmax
}

int main(void) {
    // Store one word per pixel for all batch lanes
    static uint32_t image_packed[NVREG*BS];
    //static uint32_t image_packed[IN_DIM];   
    int predictions[BATCH];

    build_pix_lut();
    build_bf16_fp4_lut();

    // Print prediction, truth, and reference format
    #ifdef PTM_PRINTS
    print_str("P|T|M\n");
    #endif

    // Count correct predictions
    int correct = 0;    
    // Count matches against reference
    int match = 0;      

    for (int s = 0; s < N_SAMPLES; s += BATCH) {
        int n = N_SAMPLES - s;
        if (n > BATCH) n = BATCH;
        int truth[BATCH], ref[BATCH];

        // Pack the batch of images
        TIME(pc_imgq, {
            // Leave unused lanes and the K pad tail (784..799) as zero
            for (int p = 0; p < IN_DIM; p++) {
                image_packed[p] = 0;
            }
            // Read the staging buffer 4 pixels to reduce MIMO accesses
            const volatile uint32_t *stage32 = (const volatile uint32_t *)IMG_STAGE;
            for (int j = 0; j < n; j++) {
                *IMG_LOAD = s + j;
                int sh = 4 * j;
                for (int w = 0; w < IN_REAL/4; w++) {   // 196 words = 784 real pixels
                    uint32_t four = stage32[w];
                    int p = w * 4;
                    image_packed[p+0] |= (uint32_t)pix_to_fp4[(four      ) & 0xFF] << sh;
                    image_packed[p+1] |= (uint32_t)pix_to_fp4[(four >>  8) & 0xFF] << sh;
                    image_packed[p+2] |= (uint32_t)pix_to_fp4[(four >> 16) & 0xFF] << sh;
                    image_packed[p+3] |= (uint32_t)pix_to_fp4[(four >> 24) & 0xFF] << sh;
                }
                truth[j] = *IMG_LABEL;
                ref[j]   = *IMG_PRED;
            }
        });

        inference_batch(image_packed, predictions);

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

    return 0;
}
