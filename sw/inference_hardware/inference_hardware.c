// MLP inference on CVE2 with GEN3 spec

#include "../headers/weights_blk32_pkgUINT32_scaleE8M0.h"
#include "image.h"

// Network dimensions for the MLP layers
#define IN_REAL   784             // real MNIST pixels
#define IN_DIM    800             // pad K only up to a BS multiple: 25*32=800 (last block 16 real + 16 zero)
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

#define QSHIFT (ACC_SHIFT - 2)    // Shift value to divide by four and round

// Enable performance counters
#define PERF_COUNTERS

// Enable intermediate UART prints (P|T|M results)
#define PTM_PRINTS

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
static uint64_t pc_qact[2];    
static uint64_t pc_bram[3];
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

    pc_line("gemm_1 |", pc_gemm[0]);    // first gemm (inc. bias load + bram->ram)
    pc_line("gemm_2 |", pc_gemm[1]);    // second gemm (inc. bias load + bram->ram)
    pc_line("gemm_3 |", pc_gemm[2]);    // third gemm (inc. bias load + bram->ram)

    pc_line("qact_1 |", pc_qact[0]);    // first activation apply + quant
    pc_line("qact_2 |", pc_qact[1]);    // second activation apply + quant

    pc_line("bram_1 |", pc_bram[0]);    // first bram->ram
    pc_line("bram_2 |", pc_bram[1]);    // second bram->ram
    pc_line("bram_3 |", pc_bram[2]);    // thrid bram->ram

    pc_line("bias_1 |", pc_bias[0]);    // first bias tile seed
    pc_line("bias_2 |", pc_bias[1]);    // second bias tile seed
    pc_line("bias_3 |", pc_bias[2]);    // third bias tile seed

    print_str("\n[PERF] total cycles over run\n");
    // total cycles spent in gemm INCLUDING bias and bram operations
    pc_line("gemm_T |", 
        pc_gemm[0] + pc_gemm[1] + pc_gemm[2]);
    // total cycles spent in gemm EXCLUDING bias and bram operations
    pc_line("gemm_X |", 
        pc_gemm[0] + pc_gemm[1] + pc_gemm[2] - 
        (pc_bram[0] + pc_bram[1] + pc_bram[2]) - 
        (pc_bias[0] + pc_bias[1] + pc_bias[2]));
    // total cycles spent applying activation + quantizing to FP4
    pc_line("qact_T |", 
        pc_qact[0] + pc_qact[1]);
    // total cycles spent reading from bram and storing
    pc_line("bram_T |", 
        pc_bram[0] + pc_bram[1] + pc_bram[2]);
    // total cycles spent performing tile bias seed
    pc_line("bias_T |", 
        pc_bias[0] + pc_bias[1] + pc_bias[2]);
}
#else
#define PC_LAYER(n)     ((void)0)
#define TIME(acc, stmt) do { stmt; } while (0)
#define pc_report()     ((void)0)
#endif

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
// Bfloat16 conversion helpers
// =======================================
// Cast float to bfloat16
static inline float to_bf16(float v) {
    union { float f; uint32_t u; } x;
    x.f = v;
    x.u = (x.u + 0x8000) & 0xFFFF0000;
    return x.f;
}

// Convert bf16 bit pattern back to float32
static inline float bf16_to_float(uint16_t h) {
    union { uint32_t u; float f; } x;
    x.u = (uint32_t)h << 16;
    return x.f;
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
// Scale extraction
// =======================================
// Extract the power of two shift from packed scales
static void extract_scale_exp(const uint32_t* words, int shift[8]) {
    for (int i = 0; i < 8; i++) {
        uint8_t byte = (uint8_t)(words[i / 4] >> (8 * (i % 4))); 
#ifdef SCALE_FMT_E8M0
        shift[i] = (int)byte - 127;               
#else
        shift[i] = (int)((byte >> 3) & 0xF) - 7;  
#endif
    }
}

// =======================================
// MAC64 hardware instructions (matmul8_vec.S)
// =======================================
//  C wrapper                 ISA op      effect
//  mac_zz()                  ZZMAC64     clear the 8x8 tile (raw + scale accum)
//  load_vN(ptr)              vle32.v     load 8 packed activation words into vN
//  vmac64_vN(base)           VMAC64      vN x weight block N -> raw tile (N*32 baked into imm)
//  mac_out(row, pair, 2)     MV2MAC64    read the pair {tile[row][2p+1], tile[row][2p]}
//  load_act_scales(base)     lw x2       load 2 words = 8 packed E8M0 act scales into t1,t2
//  mac_as()                  MAC_AS      apply the loaded activation scales to the tile
//  load_w_scales(base)       lw x2       load 2 words = 8 packed E8M0 weight scales into t3,t4
//  mac_ws()                  MAC_WS      apply the loaded weight scales to the tile
//  mac_bias()                MACBIAS     accumulate bf16 bias into tiles into acc banks
//  acc_bank()                ACCBANK     select target acc bank
//  bram_rd()                 BRAMRD      read BRAM
extern void     mac_zz(void);
extern uint32_t mac_out(uint32_t row, uint32_t pair, uint32_t mode);

extern void load_act_scales(const uint32_t *base);  
extern void mac_as(void);
extern void load_w_scales(const uint32_t *base);    
extern void mac_ws(void);

extern void load_v0 (const uint32_t *ptr);  
extern void vmac64_v0 (const uint32_t *ptr);
extern void load_v1 (const uint32_t *ptr);  
extern void vmac64_v1 (const uint32_t *ptr);
extern void load_v2 (const uint32_t *ptr);  
extern void vmac64_v2 (const uint32_t *ptr);
extern void load_v3 (const uint32_t *ptr);  
extern void vmac64_v3 (const uint32_t *ptr);
extern void load_v4 (const uint32_t *ptr);  
extern void vmac64_v4 (const uint32_t *ptr);
extern void load_v5 (const uint32_t *ptr);  
extern void vmac64_v5 (const uint32_t *ptr);
extern void load_v6 (const uint32_t *ptr);  
extern void vmac64_v6 (const uint32_t *ptr);
extern void load_v7 (const uint32_t *ptr);  
extern void vmac64_v7 (const uint32_t *ptr);
extern void load_v8 (const uint32_t *ptr);  
extern void vmac64_v8 (const uint32_t *ptr);
extern void load_v9 (const uint32_t *ptr);  
extern void vmac64_v9 (const uint32_t *ptr);
extern void load_v10(const uint32_t *ptr);  
extern void vmac64_v10(const uint32_t *ptr);
extern void load_v11(const uint32_t *ptr);  
extern void vmac64_v11(const uint32_t *ptr);
extern void load_v12(const uint32_t *ptr);  
extern void vmac64_v12(const uint32_t *ptr);
extern void load_v13(const uint32_t *ptr);  
extern void vmac64_v13(const uint32_t *ptr);
extern void load_v14(const uint32_t *ptr);  
extern void vmac64_v14(const uint32_t *ptr);
extern void load_v15(const uint32_t *ptr);  
extern void vmac64_v15(const uint32_t *ptr);
extern void load_v16(const uint32_t *ptr);  
extern void vmac64_v16(const uint32_t *ptr);
extern void load_v17(const uint32_t *ptr);  
extern void vmac64_v17(const uint32_t *ptr);
extern void load_v18(const uint32_t *ptr);  
extern void vmac64_v18(const uint32_t *ptr);
extern void load_v19(const uint32_t *ptr);  
extern void vmac64_v19(const uint32_t *ptr);
extern void load_v20(const uint32_t *ptr);  
extern void vmac64_v20(const uint32_t *ptr);
extern void load_v21(const uint32_t *ptr);  
extern void vmac64_v21(const uint32_t *ptr);
extern void load_v22(const uint32_t *ptr);  
extern void vmac64_v22(const uint32_t *ptr);
extern void load_v23(const uint32_t *ptr);  
extern void vmac64_v23(const uint32_t *ptr);
extern void load_v24(const uint32_t *ptr); 
extern void vmac64_v24(const uint32_t *ptr);
extern void load_v25(const uint32_t *ptr);  
extern void vmac64_v25(const uint32_t *ptr);
extern void load_v26(const uint32_t *ptr);  
extern void vmac64_v26(const uint32_t *ptr);
extern void load_v27(const uint32_t *ptr);  
extern void vmac64_v27(const uint32_t *ptr);
extern void load_v28(const uint32_t *ptr);  
extern void vmac64_v28(const uint32_t *ptr);
extern void load_v29(const uint32_t *ptr); 
extern void vmac64_v29(const uint32_t *ptr);
extern void load_v30(const uint32_t *ptr);  
extern void vmac64_v30(const uint32_t *ptr);
extern void load_v31(const uint32_t *ptr);  
extern void vmac64_v31(const uint32_t *ptr);

extern void mac_bias(uint8_t tile, uint8_t row, uint8_t column, uint16_t bf16);
extern void acc_bank(uint32_t tile);
extern uint32_t bram_rd(uint32_t tile, uint32_t row, uint32_t col);

// K-block reduction
static inline __attribute__((always_inline))
void do_k_tile(int vreg, const uint32_t *As, const uint32_t *Ws, const uint32_t *weights) {
    // we still need mac_zz
    mac_zz();
    load_act_scales(As);
    load_w_scales(Ws);
    switch (vreg) {
        case  0: vmac64_v0 (weights); break;
        case  1: vmac64_v1 (weights); break;
        case  2: vmac64_v2 (weights); break;
        case  3: vmac64_v3 (weights); break;
        case  4: vmac64_v4 (weights); break;
        case  5: vmac64_v5 (weights); break;
        case  6: vmac64_v6 (weights); break;
        case  7: vmac64_v7 (weights); break;
        case  8: vmac64_v8 (weights); break;
        case  9: vmac64_v9 (weights); break;
        case 10: vmac64_v10(weights); break;
        case 11: vmac64_v11(weights); break;
        case 12: vmac64_v12(weights); break;
        case 13: vmac64_v13(weights); break;
        case 14: vmac64_v14(weights); break;
        case 15: vmac64_v15(weights); break;
        case 16: vmac64_v16(weights); break;
        case 17: vmac64_v17(weights); break;
        case 18: vmac64_v18(weights); break;
        case 19: vmac64_v19(weights); break;
        case 20: vmac64_v20(weights); break;
        case 21: vmac64_v21(weights); break;
        case 22: vmac64_v22(weights); break;
        case 23: vmac64_v23(weights); break;
        case 24: vmac64_v24(weights); break;
        case 25: vmac64_v25(weights); break;
        case 26: vmac64_v26(weights); break;
        case 27: vmac64_v27(weights); break;
        case 28: vmac64_v28(weights); break;
        case 29: vmac64_v29(weights); break;
        case 30: vmac64_v30(weights); break;
        case 31: vmac64_v31(weights); break;
        default: break;
    }
    mac_as();
    mac_ws();
}

// =======================================
// Fixed point accumulation helpers
// =======================================
// Convert float to bf16 bit pattern with round half up
static inline uint16_t f_to_bf16_bits(float f) {
    union { float f; uint32_t u; } x; x.f = f;
    return (uint16_t)((x.u + 0x8000) >> 16);
}

// Convert bf16 bit pattern to int32 fixed point without float math
#define ACC_SHIFT 15
static inline int32_t bf16_to_fixed(uint16_t h) {
    int e = (h >> 7) & 0xFF;
    if (e == 0) return 0;                          
    int32_t m  = 0x80 | (h & 0x7F);               
    int     sh = e - 134 + ACC_SHIFT;             
    int32_t v  = (sh >= 0) ? (m << sh) : (m >> (-sh));
    return (h & 0x8000) ? -v : v;
}

static inline float fixed_to_float(int32_t x) {
    return (float)x / (float)(1 << ACC_SHIFT);    
}

// Save the accumulator banks (bias + all K products, already summed in BRAM)
static void bram_to_ram(int32_t *out, int Jbase, int tiles) {
    for (int T = 0; T < tiles; T++) {
        int ncol = Jbase + T*TT;                       // first neuron of this bank
        for (int g = 0; g < 4; g++) {                  // physical even rows 0,2,4,6
            for (int col = 0; col < TT; col++) {
                uint32_t pair = bram_rd(T, 2*g, col);
                int n = ncol + col;                    // output neuron
                out[n*BATCH + g]     = bf16_to_fixed((uint16_t)(pair & 0xFFFF)); // sample g
                out[n*BATCH + g + 4] = bf16_to_fixed((uint16_t)(pair >> 16));    // sample g+4
            }
        }
    }
}

// =======================================
// GEMM
// =======================================
void gemm(const uint32_t* A, const uint32_t* W, const uint32_t* bias_packed, int32_t* out,
          int KK, int WH, const uint32_t* wscale_words, const uint32_t* ascale_words) {
#ifdef PERF_COUNTERS
    uint32_t _gt0 = rdcyc();
#endif
    int num_blocks = (KK + BS - 1) / BS;   // K split into BS-blocks
    int num_tiles  = (WH + TT - 1) / TT;   // columns split into TTxTT tiles
    
    // For all rows
    for (int I = 0; I < AV; I += TT) 
    {   
        // For all columns 
        for (int J = 0; J < WH; J += (NTILES*TT))
        {
            // tiles in this NTILES*TT column strip: NTILES, or fewer on the last strip
            int tiles_this_strip = num_tiles - J/TT;
            if (tiles_this_strip > NTILES) tiles_this_strip = NTILES;
            
            TIME(pc_bias[pc_layer], 
            // seed each bank with its TTxTT bias tile
            for (int T = 0; T < tiles_this_strip; T++) {
                int idx = J/TT + T;                    // tile index
                for (int c = 0; c < 4; c++) {          // 4 column-pairs
                    for (int r = 0; r < BATCH; r++) {  // 8 rows
                        uint32_t word = bias_packed[idx*32 + c*8 + r];
                        mac_bias(T, r, 2*c,   (uint16_t)(word & 0xFFFF));   // seed into even cols (low)
                        mac_bias(T, r, 2*c+1, (uint16_t)(word >> 16));      // seed into odd cols (high)
                    }
                }            
            }
            );

            for (int K = 0; K < KK; K += (NVREG*BS)) 
            {
                // Activation vector loads into v0-v31
                load_v0  (&A[K+ 0*BS]);
                load_v1  (&A[K+ 1*BS]);
                load_v2  (&A[K+ 2*BS]);
                load_v3  (&A[K+ 3*BS]);
                load_v4  (&A[K+ 4*BS]);
                load_v5  (&A[K+ 5*BS]);
                load_v6  (&A[K+ 6*BS]);
                load_v7  (&A[K+ 7*BS]);
                load_v8  (&A[K+ 8*BS]);
                load_v9  (&A[K+ 9*BS]);
                load_v10 (&A[K+10*BS]);
                load_v11 (&A[K+11*BS]);
                load_v12 (&A[K+12*BS]);
                load_v13 (&A[K+13*BS]);
                load_v14 (&A[K+14*BS]);
                load_v15 (&A[K+15*BS]);
                load_v16 (&A[K+16*BS]);
                load_v17 (&A[K+17*BS]);
                load_v18 (&A[K+18*BS]);
                load_v19 (&A[K+19*BS]);
                load_v20 (&A[K+20*BS]);
                load_v21 (&A[K+21*BS]);
                load_v22 (&A[K+22*BS]);
                load_v23 (&A[K+23*BS]);
                load_v24 (&A[K+24*BS]);
                load_v25 (&A[K+25*BS]);
                load_v26 (&A[K+26*BS]);
                load_v27 (&A[K+27*BS]);
                load_v28 (&A[K+28*BS]);
                load_v29 (&A[K+29*BS]);
                load_v30 (&A[K+30*BS]);
                load_v31 (&A[K+31*BS]);

                const uint32_t* Ascales = &ascale_words[(K/BS)*2];
                const uint32_t* Wscales = &wscale_words[(K/BS)*2];
                const uint32_t* weights = &W[(K/BS)*BS];

                // number of K-blocks reduced in this K2 chunk: NVREG at most,
                // fewer on the final chunk (here KK <= NVREG*BS, so one chunk)
                int blk0 = K / BS;
                int blks = num_blocks - blk0;
                if (blks > NVREG) blks = NVREG;

                // Tile Loop
                for (int T = 0; T < tiles_this_strip; T++)
                {
                    // we start with an accBank T to target a new accumulator bank
                    acc_bank(T);

                    #pragma GCC unroll 32
                    for (int vreg = 0; vreg < blks; vreg++) {
                        do_k_tile(vreg, &Ascales[vreg*2], &Wscales[vreg*2], weights);
                    }

                    Wscales += num_blocks*2;   // step one tile-column of Wscales  [(KK/BS)*TT bytes]
                    weights += num_blocks*BS;  // step one tile-column of weights  [(KK/2)*TT bytes]
                }
            }
            // save BRAM to RAM
            TIME(pc_bram[pc_layer], bram_to_ram(out, J, tiles_this_strip));
        }
    }
#ifdef PERF_COUNTERS
    pc_gemm[pc_layer] += (uint32_t)(rdcyc() - _gt0);
#endif       
}

// hardtanh activation clamps values to [-4, 4]
static inline int hardtanh(int a) {
    if (a >  4) return  4;
    if (a < -4) return -4;
    return a;
}            

// Quantize fixed point values to FP4 using integer shifts
static void quantize_activation(const int32_t* A_in, uint32_t* Z_out, int dim) {
    for (int d = 0; d < dim; d++) {
        uint32_t word = 0;
        for (int j = 0; j < BATCH; j++) {
            int32_t a = A_in[d * BATCH + j];
            // Scale and round to nearest integer
            int z = (a >= 0) ? (int)(( a + (1 << (QSHIFT - 1))) >> QSHIFT)
                             : -(int)((-a + (1 << (QSHIFT - 1))) >> QSHIFT);
            z = hardtanh(z);

            int sign = (z < 0);
            int abs_v = sign ? -z : z;
            uint8_t mag = fp4_mag_lut[abs_v];
            uint32_t code = mag == 0 ? 0 : (uint32_t)(sign ? (0x8 | mag) : mag);
            word |= (code & 0xF) << (4 * j);
        }
        Z_out[d] = word;
    }
}

// Run the forward pass on a batch of samples
void inference_batch(const uint32_t* inputs, int* predictions) {
    // Gemm output goes directly into the int32 accumulator
    static int32_t out_l1[L1_DIM * BATCH], out_l2[L2_DIM * BATCH], logits[16 * BATCH];
    static uint32_t z1_packed[L1_DIM], z2_packed[L2_DIM];

    PC_LAYER(0);
    gemm(inputs, w1_fp4, bias1_packed, out_l1, IN_DIM, L1_DIM, wscale1, ascale1);
    TIME(pc_qact[0], quantize_activation(out_l1, z1_packed, L1_DIM));

    PC_LAYER(1);
    gemm(z1_packed, w2_fp4, bias2_packed, out_l2, L1_DIM, L2_DIM, wscale2, ascale2);
    TIME(pc_qact[1], quantize_activation(out_l2, z2_packed, L2_DIM));

    PC_LAYER(2);
    gemm(z2_packed, w3_fp4, bias3_packed, logits, L2_DIM, OUT_DIM, wscale3, ascale3);

    // Find the argmax using integer comparison on the fixed point logits
    for (int j = 0; j < BATCH; j++) {
        int best = 0;
        int32_t best_v = logits[0 * BATCH + j];
        for (int i = 1; i < OUT_DIM; i++) {
            int32_t v = logits[i * BATCH + j];
            if (v > best_v) { best_v = v; best = i; }
        }
        predictions[j] = best;
    }
}

int main(void) {
    // Store one word per pixel for all batch lanes
    static uint32_t image_packed[NVREG*BS];
    //static uint32_t image_packed[IN_DIM];   
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

    for (int s = 0; s < N_SAMPLES; s += BATCH) {
        int n = N_SAMPLES - s;
        if (n > BATCH) n = BATCH;
        int truth[BATCH], ref[BATCH];

        // Pack the batch of images
        TIME(pc_imgq, {
            // Leave unused lanes and the K pad tail (784..799) as zero
            for (int p = 0; p < IN_DIM; p++) image_packed[p] = 0;
            for (int j = 0; j < n; j++) {
                image_load(s + j);
                for (int p = 0; p < IN_REAL; p++) {   // only the 784 real pixels
                    uint32_t code = pix_to_fp4[IMG_STAGE[p]];
                    image_packed[p] |= code << (4 * j);
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