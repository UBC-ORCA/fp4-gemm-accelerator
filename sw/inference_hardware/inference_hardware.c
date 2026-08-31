// MLP inference on CVE2 with GEN3 spec

#include <assert.h>
#include "weights_blk32_pkgUINT32_scaleE8M0.h"
#include "image.h"

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
static uint64_t pc_bias[3];
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

    pc_line("gemm_1 |", pc_gemm[0]);    // gemm compute per layer (incl. bias seed)
    pc_line("gemm_2 |", pc_gemm[1]);
    pc_line("gemm_3 |", pc_gemm[2]);

    pc_line("qact_1 |", pc_qact[0]);    // hidden read-out from banks as packed FP4
    pc_line("qact_2 |", pc_qact[1]);

    pc_line("argmax |", pc_argmax);     // final read-out from banks + argmax

    pc_line("bias_1 |", pc_bias[0]);    // bias tile seed (subset of gemm)
    pc_line("bias_2 |", pc_bias[1]);
    pc_line("bias_3 |", pc_bias[2]);

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
    uint64_t bias_t = pc_bias[0]  + pc_bias[1]  + pc_bias[2];
    uint64_t qact_t = pc_qact[0]  + pc_qact[1];
    // end to end inference, imgq left out since real inputs arrive as FP4
    uint64_t infer_t = gemm_t + qact_t + pc_argmax;

    print_str("\n[PERF] total cycles over run\n");
    // gemm() only; read-out and argmax sit outside the function
    pc_line("gemm_T |", gemm_t);
    // the MAC datapath on its own
    pc_line("gemm_K |", gemm_k);
    // activation loads and bias seed, both inside gemm
    pc_line("load_T |", load_t);
    pc_line("bias_T |", bias_t);
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
// Encodings mirror matmul8_vec.S. CUSTOM1=0x2b, CUSTOM2=0x5b, funct3=0 unless noted
//   VMAC64(N,ptr)      VMAC64   CUSTOM1   vN x weight block at ptr -> raw tile
//   MAC_AS(a,b)        MAC_AS   f7=0x0A   apply act scale words a,b to the tile
//   MAC_WS(a,b)        MAC_WS   f7=0x0B   apply wgt scale words a,b -> fold into bank
//   MAC_BIAS(p,v)      MACBIAS  f7=0x0C   seed bf16 v into the cell addressed by p
//   ACC_BANK(t)        ACCBANK  f7=0x0D   select accumulator bank t
//   BRAM_RD(rd,p)      BRAMRD   f7=0x0E   read the bram pair at p into register rd
//   BRAM_RD_FP4(rd,p)  BRAMFP4  f7=0x08   read tile/col at p, 8 samples -> packed fp4
//   VSETVLI(avl)       vsetvli  OPV=0x57  set vl=avl, e32,m1,ta,ma
//   VLE32(N,ptr)       vle32.v  0x07      load vl words at ptr -> vN
#define VMAC64(N,ptr)       __asm__ volatile(".insn i 0x2b,0x0,x" #N ",%0,0" :: "r"(ptr))
#define MAC_AS(a,b)         __asm__ volatile(".insn r 0x5b,0x0,0x0a, x0,%0,%1" :: "r"(a),"r"(b))
#define MAC_WS(a,b)         __asm__ volatile(".insn r 0x5b,0x0,0x0b, x0,%0,%1" :: "r"(a),"r"(b))
#define MAC_BIAS(p,v)       __asm__ volatile(".insn r 0x5b,0x0,0x0c, x0,%0,%1" :: "r"(p),"r"(v))
#define ACC_BANK(t)         __asm__ volatile(".insn r 0x5b,0x0,0x0d, x0,%0,x0" :: "r"(t))
#define BRAM_RD(rd,p)       __asm__ volatile(".insn r 0x5b,0x0,0x0e, %0,%1,x0" : "=r"(rd) : "r"(p))
#define BRAM_RD_FP4(rd,p)   __asm__ volatile(".insn r 0x5b,0x0,0x08, %0,%1,x0" : "=r"(rd) : "r"(p))
#define VSETVLI(avl)        __asm__ volatile(".insn i 0x57,0x7,x0,%0,0xD0" :: "r"(avl))
#define VLE32(N,ptr)        __asm__ volatile(".insn i 0x07,0x6,x" #N ",%0,0x20" :: "r"(ptr))

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

// One K-block reduction
static inline __attribute__((always_inline))
void do_k_tile(int vreg, const uint32_t *As, const uint32_t *Ws, const uint32_t *weights) {
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

    int idx = (int)(abs_v * 2.0f + 0.5f);
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
            uint16_t best1 = best[row];
            uint16_t best2 = best[row+TT/2];
            int      pred1 = predictions[row];
            int      pred2 = predictions[row+TT/2];
            for (int col = 0; col < cols; col++) {
                int neuron = neuron0 + col;                              // column picks the neuron
                uint32_t pair = bram_rd(tile, 2 * row, col);
                uint16_t lo = bf16_ordered((uint16_t)(pair & 0xFFFF));   // sample = row
                uint16_t hi = bf16_ordered((uint16_t)(pair >> 16)   );   // sample = row + TT/2
                if (lo > best1) {
                    best1 = lo;
                    pred1 = neuron;
                }
                if (hi > best2) {
                    best2 = hi;
                    pred2 = neuron;
                }
            }
            best[row]      = best1;
            best[row+TT/2] = best2;
            predictions[row]      = pred1;
            predictions[row+TT/2] = pred2;
        }
    }
}


// Read banks back out as FP4 activations
static void readout_fp4(uint32_t *z, int WH) {
    int tiles = (WH + TT - 1) / TT;
    for (int tile = 0; tile < tiles; tile++) {
        int neuron0 = tile * TT;                   // first neuron of this bank
        #pragma GCC unroll 8 // TT
        for (int col = 0; col < TT; col++) {
            BRAM_RD_FP4(z[neuron0+col], BRAM_ADDR(tile, 0, col));
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
void inference_batch(const uint32_t* inputs, int* predictions) {
    // Hidden layers read out to FP4 activations; the final layer argmaxes straight from the banks
    static uint32_t z1_packed[L1_DIM], z2_packed[L2_DIM];

    PC_LAYER(0);
    TIME(pc_gemm[0], gemm(inputs, w1_fp4, bias1_packed, IN_DIM, L1_DIM, wscale1, ascale1));
    TIME(pc_qact[0], readout_fp4(z1_packed, L1_DIM));   // feeds layer 2

    PC_LAYER(1);
    TIME(pc_gemm[1], gemm(z1_packed, w2_fp4, bias2_packed, L1_DIM, L2_DIM, wscale2, ascale2));
    TIME(pc_qact[1], readout_fp4(z2_packed, L2_DIM));   // feeds layer 3

    PC_LAYER(2);
    TIME(pc_gemm[2], gemm(z2_packed, w3_fp4, bias3_packed, L2_DIM, OUT_DIM, wscale3, ascale3));
    TIME(pc_argmax, argmax(predictions, OUT_DIM));  // final layer -> argmax
}

int main(void) {
    assert(rdout_shift[1] == 3 && rdout_shift[2] == 3);
    // Store one word per pixel for all batch lanes
    static uint32_t image_packed[NVREG*BS];
    // Raw pixels for the whole batch
    static uint32_t stage_buf[BATCH][IN_REAL/4];
    int predictions[BATCH];

    build_pix_lut();

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
                for (int w = 0; w < IN_REAL/4; w++) {   // 196 words = 784 real pixels
                    stage_buf[j][w] = stage32[w];
                }
                truth[j] = *IMG_LABEL;
                ref[j]   = *IMG_PRED;
            }

            // Sample inner, so each word is built in a register and stored once
            for (int w = 0; w < IN_REAL/4; w++) {
                uint32_t a0 = 0;
                uint32_t a1 = 0;
                uint32_t a2 = 0;
                uint32_t a3 = 0;
                for (int j = 0; j < n; j++) {
                    uint32_t four = stage_buf[j][w];
                    int sh = 4 * j;
                    a0 |= (uint32_t)pix_to_fp4[(four      ) & 0xFF] << sh;
                    a1 |= (uint32_t)pix_to_fp4[(four >>  8) & 0xFF] << sh;
                    a2 |= (uint32_t)pix_to_fp4[(four >> 16) & 0xFF] << sh;
                    a3 |= (uint32_t)pix_to_fp4[(four >> 24) & 0xFF] << sh;
                }
                int p = w * 4;
                image_packed[p+0] = a0;
                image_packed[p+1] = a1;
                image_packed[p+2] = a2;
                image_packed[p+3] = a3;
            }

            // K pad tail (784..799) carries no pixels and must read as zero
            for (int p = IN_REAL; p < IN_DIM; p++) {
                image_packed[p] = 0;
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

    return 0;
}
