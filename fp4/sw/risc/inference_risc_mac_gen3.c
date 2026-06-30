// MLP inference on CVE2, Gen 3

#include "weights_int16_t.h"
#include "test_data_10.h"

// MLP Dimensions: 784 -> 128 -> 96 -> 10
#define IN_DIM    784
#define H1_DIM    128
#define H2_DIM     96
#define OUT_DIM    10

// number of MNIST samples solved in parallel (J dimension of A)
#define BATCH      8

#define K1_STEP   16   // K elements per inner block
#define K2_SPAN  256   // K elements per K3 step (16 inner blocks)

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

// fp4 MAC instructions (defined in matmul8_vec.S)
//  mac_zz()              -> zzMAC64          clear the hardware tile
//  mac_hw(a, b)          -> hwMAC64 rs1,rs2  T[i][j] += Aq[i] * Bq[j]
//  mac_st2_readback(out) -> 32x mv2MAC64     read full 8x8 tile -> out, clears tile
//  mac_add_bias(bias)    -> 8x addMAC64      T[i][all] += bias[i]<<2, add bias to tile
extern void mac_zz(void);
extern void mac_hw(uint32_t a, uint32_t b);
extern void mac_st2_readback(void *out);
extern void mac_add_bias(const int16_t *bias_row);

// pack 8 fp4 nibble codes into a 32-bit word
static inline uint32_t pack_fp4x8(const int16_t* codes) {
    uint32_t w = 0;
    for (int i = 0; i < 8; i++)
        w |= (uint32_t)(codes[i] & 0xF) << (4 * i);
    return w;
}

// compute a tile of 8 output rows x BATCH columns
static void tile_mac(int16_t T_out[8][BATCH],
                     const int16_t* W_t,
                     const int16_t* A_s,
                     int k_count,
                     const int16_t* bias_row) {
    mac_zz();
    if (bias_row) mac_add_bias(bias_row);
    for (int K = 0; K < k_count; K++) {
        // rs1 = W column (output rows i), rs2 = A row (batch cols j)
        mac_hw(pack_fp4x8(&W_t[K * 8]),
               pack_fp4x8(&A_s[K * BATCH]));
    }
    mac_st2_readback(T_out);
}

// F[i][j] = bias[i] + sum_k W[i][k] * A[k][j]
// W is pre-transposed, already in FP4
// A is [in_dim][BATCH] FP4, F is [out_dim][BATCH] bf16
// T accumulates a block in int16, U accumulates across K-blocks in bf16
void gemm(const int16_t* A, const int16_t* W, const int16_t* bias,
          float* F, int in_dim, int out_dim, int layer_bias) {
    // real_product = w_int * a_int * per_block_scale
    float per_block_scale = 1.0f / (float)(1 << (layer_bias + 4));
    int num_K_blocks = (in_dim + K1_STEP - 1) / K1_STEP;

    // step through the output rows in chunks of 8
    for (int I = 0; I < out_dim; I += 8) {
        int I_tile = I / 8;

        // U holds running totals for these 8 rows across all BATCH columns
        float U[8][BATCH];

        // U init to zero
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < BATCH; j++) U[i][j] = 0.0f;

        // k accumulation
        for (int K3 = 0; K3 < in_dim; K3 += K2_SPAN) {
            for (int K2 = 0; K2 < K2_SPAN && (K3 + K2) < in_dim; K2 += K1_STEP) {

                int k1_end = K1_STEP;
                if (K3 + K2 + k1_end > in_dim) k1_end = in_dim - K3 - K2;

                int K_block = (K3 + K2) / K1_STEP;
                const int16_t* W_strip =
                    &W[(I_tile * num_K_blocks + K_block) * K1_STEP * 8];

                // bias only on the first K-block of each output tile
                const int16_t* tile_bias =
                    (K3 == 0 && K2 == 0) ? &bias[I] : (const int16_t*)0;

                int16_t T[8][BATCH];
                tile_mac(T, W_strip, &A[(K3 + K2) * BATCH], k1_end, tile_bias);

                // accumulate into U
                for (int i = 0; i < 8 && (I + i) < out_dim; i++) {
                    for (int j = 0; j < BATCH; j++) {
                        U[i][j] = to_bf16(U[i][j] + (float)T[i][j] * per_block_scale);
                    }
                }
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

    gemm(inputs, w1_fp4, bias1, h1, IN_DIM, H1_DIM, LAYER1_BIAS);
    quantize_activation(h1, h1_codes, H1_DIM);

    gemm(h1_codes, w2_fp4, bias2, h2, H1_DIM, H2_DIM, LAYER2_BIAS);
    quantize_activation(h2, h2_codes, H2_DIM);

    gemm(h2_codes, w3_fp4, bias3, logits, H2_DIM, OUT_DIM, LAYER3_BIAS);

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

    // pred | truth per sample
    print_str("P|T\n");

    int correct = 0;

    for (int s = 0; s < N_SAMPLES; s += BATCH) {
        int n = N_SAMPLES - s;
        if (n > BATCH) n = BATCH;

        for (int p = 0; p < IN_DIM; p++) {
            for (int j = 0; j < BATCH; j++) {
                image_batch[p * BATCH + j] =
                    (j < n) ? fp4_quantize((float)test_images[s + j][p] / 255.0f) : 0;
            }
        }

        inference_batch(image_batch, preds);

        for (int j = 0; j < n; j++) {
            if (preds[j] == test_labels[s + j]) correct++;
            putdec(preds[j]);
            print_str("|");
            putdec(test_labels[s + j]);
            print_str("\n");
        }
    }

    print_str("\nACCURACY: ");
    putdec(correct);
    putchar_uart('/');
    putdec(N_SAMPLES);
    print_str("\n");

    return 0;
}