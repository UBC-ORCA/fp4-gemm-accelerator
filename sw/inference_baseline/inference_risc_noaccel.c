// MLP inference speed baseline on CVE2, no MAC accelerator

#include "weights_blk32_pkgUINT32_scaleE8M0.h"
#include "image.h"

// Network dimensions. No K padding here, the scalar loop has no block size
#define IN_REAL   784             // real MNIST pixels
#define L1_DIM    128
#define L2_DIM     96
#define OUT_DIM    10

#define TT          8             // FP4 codes per weight word
#define FP4_BITS    4             // bits per code, so 8 fit in a 32 bit word
#define BS  K1_STEP_HDR           // K elements per block, sets the weight stride

// Useful MACs per image
#define MACS_PER_IMAGE  ((uint64_t)IN_REAL * L1_DIM + \
                         (uint64_t)L1_DIM  * L2_DIM + \
                         (uint64_t)L2_DIM  * OUT_DIM)
// 1 MAC is one multiply plus one add
#define FLOPS_PER_IMAGE (2 * MACS_PER_IMAGE)

// Accumulator shift before requantizing to an FP4 code
#define RDOUT_SHIFT  3

// Enable performance counters
#define PERF_COUNTERS

// Enable intermediate UART prints (P|T|M results)
// #define PTM_PRINTS

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
static uint64_t pc_qact[2];    // requantize to FP4 codes between layers
static uint64_t pc_argmax;
static uint64_t pc_total;      // whole sample loop
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

// Print scaled/100 with 2 decimals, since there is no FP here
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

    pc_line("gemm_1 |", pc_gemm[0]);    // scalar gemm per layer, bias included
    pc_line("gemm_2 |", pc_gemm[1]);
    pc_line("gemm_3 |", pc_gemm[2]);

    pc_line("qact_1 |", pc_qact[0]);    // requantize to FP4 codes
    pc_line("qact_2 |", pc_qact[1]);

    pc_line("argmax |", pc_argmax);

    uint64_t gemm_x = pc_gemm[0] + pc_gemm[1] + pc_gemm[2];

    print_str("\n[PERF] total cycles over run\n");
    pc_line("gemm_T |", gemm_x + pc_qact[0] + pc_qact[1] + pc_argmax);
    pc_line("gemm_X |", gemm_x);
    pc_line("qact_T |", pc_qact[0] + pc_qact[1]);

    // No MAC array here, so usage and gemm_U do not apply
    uint64_t flops = FLOPS_PER_IMAGE * (uint64_t)N_SAMPLES;

    print_str("\n[PERF] flops (scalar, 1 mac = 2 flops)\n");
    pc_line("flops  |", flops);
    pc_rate("f/cyc  |", flops, pc_total);   // achieved, whole program
    pc_rate("f/cyc_g|", flops, gemm_x);     // achieved, gemm only
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
static int16_t fp4_quantize(float value) {
    int sign = (value < 0.0f);
    float abs_v = sign ? -value : value;

    int idx = (int)(abs_v * 4.0f + 0.5f);
    if (idx > 15) idx = 15;
    uint8_t mag = fp4_mag_lut[idx];

    if (mag == 0) return 0;
    return (int16_t)(sign ? (0x8 | mag) : mag);
}

// Precomputed table to convert pixel bytes to FP4
static int8_t pix_to_fp4[256];

static void build_pix_lut(void) {
    for (int v = 0; v < 256; v++) {
        { int8_t c = (int8_t)(fp4_quantize((float)v / 255.0f) & 0xF); pix_to_fp4[v] = (int8_t)((c ^ 8) - 8); }
    }
}

// =======================================
// GEMM
// =======================================
// One layer, one sample. C = A*W + bias, all scalar.
//   A    : KK FP4 codes, one per byte
//   W    : uint32 words of 8 FP4 codes, one per output column of a tile.
//          Tile T starts at W[T * stride] where stride is KK padded to BS
//   bias : bf16 words, read as integers since the result is wrong anyway
//   out  : WH int32 accumulators
static void gemm(const int8_t *A, const uint32_t *W, const uint32_t *bias_packed,
                 int32_t *out, int KK, int WH) {
#ifdef PERF_COUNTERS
    uint32_t _gt0 = rdcyc();
#endif
    int tiles  = (WH + TT - 1) / TT;
    int stride = ((KK + BS - 1) / BS) * BS;   // weights are stored K-padded

    for (int T = 0; T < tiles; T++) {
        int32_t acc[TT];

        // Seed this tile's 8 columns with their bias
        for (int c = 0; c < TT; c++) {
            uint32_t word = bias_packed[T * 32 + (c / 2) * 8];
            acc[c] = (c & 1) ? (int32_t)(int16_t)(word >> 16)
                             : (int32_t)(int16_t)(word & 0xFFFF);
        }

        const uint32_t *tile_weights = W + (uint32_t)T * stride;
        for (int k = 0; k < KK; k++) {
            int32_t act  = A[k];                      // feeds all 8 columns
            int32_t w_word = (int32_t)tile_weights[k];  // column 7 sits on top

            #pragma GCC unroll 8
            for (int c = TT - 1; c >= 0; c--) {
                acc[c] += act * (w_word >> (32 - FP4_BITS));
                w_word <<= FP4_BITS;
            }
        }

        int cols = WH - T * TT;
        if (cols > TT) cols = TT;
        for (int c = 0; c < cols; c++) {
            out[T * TT + c] = acc[c];
        }
    }
#ifdef PERF_COUNTERS
    pc_gemm[pc_layer] += (uint32_t)(rdcyc() - _gt0);
#endif
}

// Accumulators back down to FP4 codes for the next layer
static void requantize(const int32_t *in, int8_t *out, int dim) {
    for (int i = 0; i < dim; i++) {
        int32_t v = in[i] >> RDOUT_SHIFT;
        if (v >  7) v =  7;
        if (v < -8) v = -8;
        out[i] = (int8_t)v;
    }
}

static int argmax(const int32_t *logits, int dim) {
    int best = 0;
    for (int i = 1; i < dim; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

// Run the forward pass on one sample
static int inference(const int8_t *image) {
    static int32_t z1[L1_DIM], z2[L2_DIM], logits[OUT_DIM];
    static int8_t a1[L1_DIM], a2[L2_DIM];
    int pred;

    PC_LAYER(0);
    gemm(image, w1_fp4, bias1_packed, z1, IN_REAL, L1_DIM);
    TIME(pc_qact[0], requantize(z1, a1, L1_DIM));

    PC_LAYER(1);
    gemm(a1, w2_fp4, bias2_packed, z2, L1_DIM, L2_DIM);
    TIME(pc_qact[1], requantize(z2, a2, L2_DIM));

    PC_LAYER(2);
    gemm(a2, w3_fp4, bias3_packed, logits, L2_DIM, OUT_DIM);
    TIME(pc_argmax, pred = argmax(logits, OUT_DIM));

    return pred;
}

int main(void) {
    static int8_t image[IN_REAL];

    build_pix_lut();

    // Print prediction, truth, and reference format
    #ifdef PTM_PRINTS
    print_str("P|T|M\n");
    #endif

    // Count correct predictions
    int correct = 0;
    // Count matches against reference
    int match = 0;

#ifdef PERF_COUNTERS
    uint32_t _rt0 = rdcyc();
#endif

    for (int s = 0; s < N_SAMPLES; s++) {
        int truth, ref;

        // Load one image and quantize its pixels to FP4 codes
        TIME(pc_imgq, {
            image_load(s);
            const volatile uint8_t *stage = (const volatile uint8_t *)IMG_STAGE;
            for (int p = 0; p < IN_REAL; p++) {
                image[p] = pix_to_fp4[stage[p]];
            }
        });
        truth = *IMG_LABEL;
        ref   = *IMG_PRED;

        int pred = inference(image);
        if (pred == truth) correct++;
        if (pred == ref)   match++;

        #ifdef PTM_PRINTS
        putdec(pred);
        print_str("|");
        putdec(truth);
        print_str("|");
        putdec(ref);
        print_str("\n");
        #endif
    }

#ifdef PERF_COUNTERS
    pc_total = (uint32_t)(rdcyc() - _rt0);
#endif

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
