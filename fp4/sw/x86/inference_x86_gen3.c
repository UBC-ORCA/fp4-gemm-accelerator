// MLP inference on x86, Gen 3

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "../headers/weights_int16_t.h"
#include "../headers/test_data.h"

// MLP Dimensions: 784 -> 128 -> 96 -> 10
#define IN_DIM    784
#define H1_DIM    128
#define H2_DIM     96
#define OUT_DIM    10

// number of MNIST samples solved in parallel (J dimension of A)
#define BATCH      8

#define K1_STEP   16   // K elements per inner block
#define K2_SPAN  256   // K elements per K3 step (16 inner blocks)

// fp4 code to int5 LUT
static const int8_t code_to_int[16] = {
//  0  1  2  3  4  5  6   7   8  9 10 11 12 13 14  15
    0, 1, 2, 3, 4, 6, 8, 12,  0,-1,-2,-3,-4,-6,-8,-12
};

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

// verify each entry is a valid FP4 and not raw int16
void validate_fp4(const int16_t* w, int n, const char* name) {
    int observed_value[16] = {0};
    for (int i = 0; i < n; i++) {
        if (w[i] < 0 || w[i] > 15) {
            printf("%s: index %d out of FP4 range: %d\n", name, i, w[i]);
            exit(1);
        }
        observed_value[w[i]] = 1;
    }

    int count = 0;
    for (int i = 0; i < 16; i++) if (observed_value[i]) count++;

    printf("Layer %s: %d distinct codes {", name, count);
    int first = 1;
    for (int i = 0; i < 16; i++) {
        if (observed_value[i]) {
            if (!first) printf(",");
            printf("%d", i);
            first = 0;
        }
    }
    printf("}\n");
}

// convert input float to its nearest FP4 code
// used only for the initial pixel quantization; hidden layers stay in int
int16_t fp4_quantize(float value) {
    int sign = (value < 0.0f);
    float abs_v = sign ? -value : value;

    int idx = (int)(abs_v * 4.0f + 0.5f);
    if (idx > 15) idx = 15;
    uint8_t mag = fp4_mag_lut[idx];

    if (mag == 0) return 0;
    return (int16_t)(sign ? (0x8 | mag) : mag);
}

// simulated MAC instructions
// v0 = W vector
// v1 = A vector
// T_tile = int16 accumulator
static int     v0[8];
static int     v1[BATCH];
static int16_t T_tile[8][BATCH];

// zzMAC: clear the T tile
static inline void zzMAC(void) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < BATCH; j++) T_tile[i][j] = 0;
}

// setWMAC v0, W'[k]: load 8 weights into v0
static inline void setWMAC(const int16_t* W_codes) {
    for (int i = 0; i < 8; i++) v0[i] = code_to_int[W_codes[i]];
}

// setAMAC v1, A[k]: load BATCH activations into v1
static inline void setAMAC(const int16_t* A_codes) {
    for (int j = 0; j < BATCH; j++) v1[j] = code_to_int[A_codes[j]];
}

// vhwMAC: outer product accumulate, T_tile[i][j] += v0[i] * v1[j]
static inline void vhwMAC(void) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < BATCH; j++)
            T_tile[i][j] += (int16_t)(v0[i] * v1[j]);
}

// st2MAC: copy the T tile to memory (real hw also clears it on readout)
static inline void st2MAC(int16_t T_out[8][BATCH]) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < BATCH; j++) T_out[i][j] = T_tile[i][j];
}

// outer-product MAC over one K-block, into the int16 T tile
static void tile_mac(int16_t T_out[8][BATCH],
                     const int16_t* W_t,
                     const int16_t* A_s,
                     int k_count) {
    zzMAC();
    for (int K = 0; K < k_count; K++) {
        setWMAC(&W_t[K * 8]);
        setAMAC(&A_s[K * BATCH]);
        vhwMAC();
    }
    st2MAC(T_out);
}

// gemm: F[i][j] = bias[i] + sum_k W[i][k] * A[k][j]
// W is pre-transposed, already in FP4
// A is [in_dim][BATCH] FP4, F is [out_dim][BATCH] bf16
// T accumulates a block in int16, U accumulates across K-blocks in bf16
void gemm(const int16_t* A, const int16_t* W, const int16_t* bias,
          float* F, int in_dim, int out_dim, int layer_bias) {
    // real_weight  = w_int * weight_scale
    // real_product = w_int * a_int * per_block_scale
    float weight_scale    = 1.0f / (float)(1 << (layer_bias + 2));
    float per_block_scale = 1.0f / (float)(1 << (layer_bias + 4));
    int num_K_blocks = (in_dim + K1_STEP - 1) / K1_STEP;

    // step through the output rows in chunks of 8
    for (int I = 0; I < out_dim; I += 8) {
        int I_tile = I / 8;

        // U holds running totals for these 8 rows across all BATCH columns
        float U[8][BATCH];

        // init U with the bias values for these 8 rows in bfloat16
        for (int i = 0; i < 8 && (I + i) < out_dim; i++) {
            float b_f = to_bf16((float)bias[I + i] * weight_scale);
            for (int j = 0; j < BATCH; j++) U[i][j] = b_f;
        }

        // k accumulation
        for (int K3 = 0; K3 < in_dim; K3 += K2_SPAN) {
            for (int K2 = 0; K2 < K2_SPAN && (K3 + K2) < in_dim; K2 += K1_STEP) {

                int k1_end = K1_STEP;
                if (K3 + K2 + k1_end > in_dim) k1_end = in_dim - K3 - K2;

                int K_block = (K3 + K2) / K1_STEP;
                const int16_t* W_strip =
                    &W[(I_tile * num_K_blocks + K_block) * K1_STEP * 8];

                int16_t T[8][BATCH];
                tile_mac(T, W_strip, &A[(K3 + K2) * BATCH], k1_end);

                // scale the integer sum to float, round to bf16, accumulate into U
                for (int i = 0; i < 8 && (I + i) < out_dim; i++) {
                    for (int j = 0; j < BATCH; j++) {
                        U[i][j] = to_bf16(U[i][j] + (float)T[i][j] * per_block_scale);
                    }
                }
            }
        }

        // write the completed 8 rows out to the final output matrix F
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

    // ensure weights are of correct format
    validate_fp4(w1_fp4, sizeof(w1_fp4)/sizeof(w1_fp4[0]), "w1_fp4");
    validate_fp4(w2_fp4, sizeof(w2_fp4)/sizeof(w2_fp4[0]), "w2_fp4");
    validate_fp4(w3_fp4, sizeof(w3_fp4)/sizeof(w3_fp4[0]), "w3_fp4");

    printf("P|T\n");

    int correct = 0;
    int total = (N_SAMPLES / BATCH) * BATCH;

    for (int s = 0; s < total; s += BATCH) {
        // pixels uint8 [0,255] -> float [0,1] -> FP4 code, packed [K][BATCH]
        for (int p = 0; p < IN_DIM; p++) {
            for (int j = 0; j < BATCH; j++) {
                image_batch[p * BATCH + j] =
                    fp4_quantize((float)test_images[s + j][p] / 255.0f);
            }
        }

        inference_batch(image_batch, preds);

        for (int j = 0; j < BATCH; j++) {
            if (preds[j] == test_labels[s + j]) correct++;
        }
    }

    printf("\nACCURACY: %d/%d\n", correct, total);
    return 0;
}