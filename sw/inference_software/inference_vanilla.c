// MLP inference on CVE2, Gen 3, software only build
// same as inference_fp4mac/inference.c, but the MAC64 tile is modeled in C
// instead of the matmul8_vec.S hardware instructions

#include "../headers/weights_blk32_pkgINT16_scaleE8M0.h"

// Samples
#define N_SAMPLES 80

// Test data loading
#define IMG_LOAD  ((volatile unsigned int  *) 0xFFFF0010)
#define IMG_LABEL ((volatile unsigned int  *) 0xFFFF0014)
#define IMG_PRED  ((volatile unsigned int  *) 0xFFFF0018)
#define IMG_STAGE ((volatile unsigned char *) 0x80070000)

// MLP Dimensions: 784 -> 128 -> 96 -> 10
#define IN_DIM    784
#define H1_DIM    128
#define H2_DIM     96
#define OUT_DIM    10

// number of MNIST samples solved in parallel (J dimension of A)
#define BATCH      8

#define K1_STEP    K1_STEP_HDR   // K elements per inner block
#define K2_SPAN    256           // K elements per K3 step

// write 1 to halt the simulator
#define DEV_HALT ((volatile int *) 0x20008)

// UART putchar
extern void putchar_uart(char c);

static void print_str(const char *s) {
  while (*s) putchar_uart(*s++);
}

void putdec(uint32_t n) {
    char buf[11];
    int i = 0;
    if (n == 0) { putchar_uart('0'); return; }
    while (n > 0) { buf[i++] = '0' + (n % 10); n /= 10; }
    while (i--) putchar_uart(buf[i]);
}

void* memcpy(void* dst, const void* src, int n) {
    char* d = (char*)dst;
    const char* s = (const char*)src;
    for (int i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

void* memset(void* dst, int c, int n) {
    char* d = (char*)dst;
    for (int i = 0; i < n; i++) d[i] = (char)c;
    return dst;
}

// fp4 magnitude LUT
static const uint8_t fp4_mag_lut[16] = {
//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    0, 1, 2, 3, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7
};

// float cast to bfloat16
static inline float to_bf16(float v) {
    union { float f; uint32_t u; } x;
    x.f = v;
    x.u = (x.u + 0x8000) & 0xFFFF0000;
    return x.f;
}

// convert input float to its nearest FP4 code
int16_t fp4_quantize(float value) {
    int sign = (value < 0.0f);
    float abs_v = sign ? -value : value;

    int idx = (int)(abs_v * 4.0f + 0.5f);
    if (idx > 15) idx = 15;
    uint8_t mag = fp4_mag_lut[idx];

    if (mag == 0) return 0;
    return (int16_t)(sign ? (0x8 | mag) : mag);
}

// pack 8 fp4 nibble codes into a 32-bit word
static inline uint32_t pack_fp4x8(const int16_t* codes) {
    uint32_t w = 0;
    for (int i = 0; i < 8; i++) {
        w |= (uint32_t)(codes[i] & 0xF) << (4 * i);
    }
    return w;
}

// extract the per-lane power-of-two shift from 8 packed scales (2 words)
// can be either E8M0 (defined in header) or E4M3
static void extract_scale_exp(const uint32_t* words, int shift[8]) {
    for (int i = 0; i < 8; i++) {
        uint8_t byte = (uint8_t)(words[i / 4] >> (8 * (i % 4)));  // lane i lives in byte i
#ifdef SCALE_FMT_E8M0
        shift[i] = (int)byte - 127;               // E8M0: byte is the biased exponent
#else
        shift[i] = (int)((byte >> 3) & 0xF) - 7;  // E4M3: exp field - bias(7)
#endif
    }
}

// multiply x by 2^e by adding e to the float exponent field
static inline float scale_pow2(float x, int e) {
    if (x == 0.0f) return 0.0f;
    union { float f; uint32_t u; } v;
    v.f = x;
    uint32_t exp = ((v.u >> 23) & 0xFF) + (uint32_t)e;   // shift the exponent
    v.u = (v.u & 0x807FFFFF) | (exp << 23);              // keep sign + mantissa
    return v.f;
}

// fp4 code to int5 LUT (the hardware decodes the packed nibbles the same way)
static const int8_t code_to_int[16] = {
//  0  1  2  3  4  5  6   7   8  9 10 11 12 13 14  15
    0, 1, 2, 3, 4, 6, 8, 12,  0,-1,-2,-3,-4,-6,-8,-12
};

// software model of the MAC64 tile, the only difference vs the hardware build
//  mac_zz()              clear the 8x8 tile
//  mac_hw(a, b)          T[i][j] += Aq[i] * Bq[j] from two packed fp4x8 words
//  mac_st2_readback(out) read the full 8x8 tile -> out, clears the tile
static int16_t T_tile[8][BATCH];

static inline void mac_zz(void) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < BATCH; j++) {
            T_tile[i][j] = 0;
        }
    }
}

static inline void mac_hw(uint32_t w_packed, uint32_t a_packed) {
    for (int i = 0; i < 8; i++) {
        int wi = code_to_int[(w_packed >> (4 * i)) & 0xF];   // rs1 nibble -> output row i
        for (int j = 0; j < BATCH; j++) {
            int aj = code_to_int[(a_packed >> (4 * j)) & 0xF];  // rs2 nibble -> batch col j
            T_tile[i][j] += (int16_t)(wi * aj);
        }
    }
}

static void mac_st2_readback(void* out) {
    int16_t (*T)[BATCH] = (int16_t (*)[BATCH])out;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < BATCH; j++) {
            T[i][j] = T_tile[i][j];
            T_tile[i][j] = 0;
        }
    }
}

// emulated vector register
typedef struct { uint32_t w[K1_STEP]; } vreg_t;

// vle_fp4: load + pack bs k-steps x 8 activation lanes into a vector reg
static inline void vle_fp4(vreg_t* v, const int16_t* A_v, int bs) {
    for (int k = 0; k < bs; k++) {
        v->w[k] = pack_fp4x8(&A_v[k * BATCH]);
    }
}

// macWs: load 8 per-row weight shifts
static inline void macWs(const uint32_t* words, int row_shift[8]) {
    extract_scale_exp(words, row_shift);
}

// macAs: load 8 per-col activation shifts
static inline void macAs(const uint32_t* words, int col_shift[8]) {
    extract_scale_exp(words, col_shift);
}

// vmac64: block MAC over bs k-steps, T[i][j] += sum_k W[k][i] * v[k][j]
static inline void vmac64(const int16_t* W_v, const vreg_t* v, int bs) {
    for (int k = 0; k < bs; k++) {
        // rs1 = W column (output rows i), rs2 = A row (batch cols j)
        mac_hw(pack_fp4x8(&W_v[k * 8]), v->w[k]);
    }
}

// macAcc: read the tile, scale each entry (block shift), accumulate into U as bf16
static void macAcc(float U[8][BATCH], const int shift[8][BATCH]) {
    int16_t T[8][BATCH];
    mac_st2_readback(T);
    // static int dbg_tile = 0;
    // if (dbg_tile < 50) {
    //     print_str("[FW TILE]\n");
    //     for (int i = 0; i < 8; i++) {
    //         for (int j = 0; j < BATCH; j++) {
    //             int v = T[i][j];
    //             if (v < 0) { putchar_uart('-'); v = -v; }
    //             putdec((uint32_t)v);
    //             putchar_uart(' ');
    //         }
    //         putchar_uart('\n');
    //     }
    //     dbg_tile++;
    // }
    // read the tile out (and clear it)
    for (int i = 0; i < 8; i++) {                   // padding rows discarded at the F write
        for (int j = 0; j < BATCH; j++) {
            U[i][j] = to_bf16(U[i][j] + scale_pow2((float)T[i][j], shift[i][j]));
        }
    }
}

// fill the 8x8 tile
static void tile_mac(const int16_t* W_t,
                     const int16_t* A_s,
                     int k_count) {
    mac_zz();
    vreg_t va;
    vle_fp4(&va, A_s, k_count);     // load activations into a vector reg
    vmac64(W_t, &va, k_count);      // block MAC over k_count k-steps
}

// F[i][j] = bias[i] + sum_k W[i][k] * A[k][j]
// W is pre-transposed, already in FP4
// A is [in_dim][BATCH] FP4, F is [out_dim][BATCH] bf16
// T accumulates a block in int16, U accumulates across K-blocks in bf16
void gemm(const int16_t* A, const int16_t* W, const int16_t* bias, float* F,
          int in_dim, int out_dim,
          const uint32_t* wscale_words, const uint32_t* ascale_words) {
    int num_K_blocks = (in_dim + K1_STEP - 1) / K1_STEP;

    // decode the per-block scales once
    // scale = 2^wshift[i] * 2^ashift[j] = 2^(wshift[i]+ashift[j])  (powers of two -> a shift)
    int wshift[8], ashift[8];
    macWs(wscale_words, wshift);   // per-row weight shifts
    macAs(ascale_words, ashift);   // per-col activation shifts

    // combine shift for activations and weights
    int shift[8][BATCH];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < BATCH; j++) {
            shift[i][j] = wshift[i] + ashift[j];
        }
    }

    // step through the output rows in chunks of 8
    for (int I = 0; I < out_dim; I += 8) {
        int I_tile = I / 8;

        // U holds running totals for these 8 rows across all BATCH columns
        float U[8][BATCH];

        // init U with the bias for these 8 rows, scaled by the per-row weight shift
        for (int i = 0; i < 8; i++) {
            float b = (I + i < out_dim) ? to_bf16(scale_pow2((float)bias[I + i], wshift[i])) : 0.0f;
            for (int j = 0; j < BATCH; j++) {
                U[i][j] = b;
            }
        }

        // k accumulation
        for (int K3 = 0; K3 < in_dim; K3 += K2_SPAN) {
            for (int K2 = 0; K2 < K2_SPAN && (K3 + K2) < in_dim; K2 += K1_STEP) {

                // clamp the last K-block if it runs past in_dim
                int k1_end = K1_STEP;
                if (K3 + K2 + k1_end > in_dim) k1_end = in_dim - K3 - K2;

                // point at this K-block's weight strip
                int K_block = (K3 + K2) / K1_STEP;
                const int16_t* W_strip = &W[(I_tile * num_K_blocks + K_block) * K1_STEP * 8];

                tile_mac(W_strip, &A[(K3 + K2) * BATCH], k1_end);  // fill tile
                macAcc(U, shift);                                  // drain -> U
            }
        }

        // write to the final output matrix F
        for (int i = 0; i < 8 && (I + i) < out_dim; i++) {
            for (int j = 0; j < BATCH; j++) {
                F[(I + i) * BATCH + j] = U[i][j];
            }
        }
    }
}

// hardtanh
static inline int hardtanh(int a) {
    if (a >  4) return  4;
    if (a < -4) return -4;
    return a;
}

// quantize to FP4 codes, and clamp with hardtanh
static void quantize_activation(const float* A, int16_t* codes, int dim) {
    int n = dim * BATCH;
    for (int i = 0; i < n; i++) {
        float a = A[i];
        // scale by 4 and round to nearest int (ties away from zero)
        int z = (int)(a * 4.0f + (a >= 0.0f ? 0.5f : -0.5f));
        z = hardtanh(z);

        // get magnitude
        int sign = (z < 0);
        int abs_v = sign ? -z : z;
        uint8_t mag = fp4_mag_lut[abs_v];
        codes[i] = mag == 0 ? 0 : (int16_t)(sign ? (0x8 | mag) : mag);
    }
}

// forward pass on a batch of BATCH samples
// 2 hidden layers with hardtanh activation function
void inference_batch(const int16_t* inputs, int* preds) {
    static float h1[H1_DIM * BATCH], h2[H2_DIM * BATCH], logits[OUT_DIM * BATCH];
    static int16_t h1_codes[H1_DIM * BATCH], h2_codes[H2_DIM * BATCH];

    gemm(inputs, w1_fp4, bias1, h1, IN_DIM, H1_DIM, wscale1, ascale1);
    quantize_activation(h1, h1_codes, H1_DIM);

    gemm(h1_codes, w2_fp4, bias2, h2, H1_DIM, H2_DIM, wscale2, ascale2);
    quantize_activation(h2, h2_codes, H2_DIM);

    gemm(h2_codes, w3_fp4, bias3, logits, H2_DIM, OUT_DIM, wscale3, ascale3);

    // argmax
    for (int j = 0; j < BATCH; j++) {
        int best = 0;
        float best_v = logits[0 * BATCH + j];
        for (int i = 1; i < OUT_DIM; i++) {
            float v = logits[i * BATCH + j];
            if (v > best_v) { best_v = v; best = i; }
        }
        preds[j] = best;
    }
}

int main(void) {
    static int16_t image_batch[IN_DIM * BATCH];
    int preds[BATCH];

    // pred | truth | mptorch reference, per sample
    print_str("P|T|M\n");

    int correct = 0;
    int match = 0;      // sw prediction vs mptorch reference

    for (int s = 0; s < N_SAMPLES; s += BATCH) {
        int n = N_SAMPLES - s;
        if (n > BATCH) n = BATCH;
        int truth[BATCH], ref[BATCH];

        for (int j = 0; j < BATCH; j++) {
            if (j < n) {
                *IMG_LOAD = s + j;                 // TB stages this image into DMEM
                for (int p = 0; p < IN_DIM; p++)
                    image_batch[p * BATCH + j] =
                        fp4_quantize((float)IMG_STAGE[p] / 255.0f);
                truth[j] = *IMG_LABEL;
                ref[j]   = *IMG_PRED;
            } else {
                for (int p = 0; p < IN_DIM; p++) image_batch[p * BATCH + j] = 0;
            }
        }

        inference_batch(image_batch, preds);

        for (int j = 0; j < n; j++) {
            if (preds[j] == truth[j]) correct++;
            if (preds[j] == ref[j])   match++;
            putdec(preds[j]);
            print_str("|");
            putdec(truth[j]);
            print_str("|");
            putdec(ref[j]);
            print_str("\n");
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

    return 0;
}
