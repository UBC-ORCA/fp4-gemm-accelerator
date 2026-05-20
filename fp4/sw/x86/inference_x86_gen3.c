// MLP inference on x86, Gen 3
// 8x8 MAC tile, int16 within K-block of 16, bfloat16 output
// per-block scale factors duplicated from per-tensor

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "../headers/weights_int16.h"
#include "../headers/test_data.h"

// MLP Dimensions: 784 -> 128 -> 96 -> 10
#define IN_DIM    784
#define H1_DIM    128
#define H2_DIM     96
#define OUT_DIM    10

// activation scaling bias (currently only 0)
#define ACT_BIAS  0

// nibble to int5 LUT
static const int8_t nibble_to_int[16] = {
//  0  1  2  3  4  5  6   7   8  9 10 11 12 13 14  15
    0, 1, 2, 3, 4, 6, 8, 12,  0,-1,-2,-3,-4,-6,-8,-12
};

// FP4 LUT for quantization
static const float quant_fp4_lut[8] = {
    0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f
};

// 1 = truncate U writes to bfloat16, 0 = keep full float
#define USE_BF16 0

// float cast to bfloat16
static inline float to_bf16(float v) {
#if USE_BF16
    union { float f; uint32_t u; } x;
    x.f = v;
    x.u = (x.u + 0x8000) & 0xFFFF0000;
    return x.f;
#else
    return v;
#endif
}

// verify each entry is a valid FP4 and not raw int16
void validate_fp4(const int16_t* w, int n, const char* name) {
    // mark which nibble values appear in the weights
    int observed_value[16] = {0};
    for (int i = 0; i < n; i++) {
        if (w[i] < 0 || w[i] > 15) {
            printf("%s: index %d out of FP4 range: %d\n", name, i, w[i]);
            exit(1);
        }
        observed_value[w[i]] = 1;
    }

    // count distinct nibbles observed
    int count = 0;
    for (int i = 0; i < 16; i++) {
        if (observed_value[i]) count++;
    }

    printf("Layer %s: %d distinct nibbles {", name, count);
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

// convert input float to its nearest FP4 nibble
int16_t fp4_quantize(float value) {
    float abs_value = value < 0 ? -value : value;

    // round to nearest LUT entry
    int nearest_idx = 0;
    float min_qerror = abs_value;
    for (int j = 1; j < 8; j++) {
        float qerror = abs_value - quant_fp4_lut[j];
        if (qerror < 0) qerror = -qerror;
        if (qerror < min_qerror) {
            min_qerror  = qerror;
            nearest_idx = j;
        } else if (qerror == min_qerror && (j & 1) == 0) {
            nearest_idx = j;
        }
    }

    // encode nibble
    if (nearest_idx == 0) return 0;
    return (int16_t)(value < 0 ? (0x8 | nearest_idx) : nearest_idx);
}

// gemm: F[j] = bias[j] + sum_k A[k] * W[j,k]
// inputs are FP4 nibbles for both A and W
// inner T sums 16 K-element products in int16, could be int13 later
// outer U accumulates across all K-blocks in bfloat16
// block scale is the same value for every block, tensor-scaling fallback
void gemm(const int16_t* A, const int16_t* W, float* F,
               int in_dim, int out_dim, int layer_bias) {
    // real_weight = w_int * weight_scale
    // real_product = w_int * a_int * per_block_scale
    float weight_scale = 1.0f / (float)(1 << (layer_bias + 2));
    float per_block_scale = 1.0f / (float)(1 << (layer_bias + 4));
    int stride = in_dim + 1;

    // step through the output rows in chunks of 8
    // J=1 here so the tile is currently 8x1
    for (int I = 0; I < out_dim; I += 8) {
        // U holds running totals for these 8 rows
        float U[8] = {0};

        // init U with the bias values for these 8 rows in bfloat16
        for (int i = 0; i < 8 && (I + i) < out_dim; i++) {
            int row = (I + i) * stride;
            int b_int = nibble_to_int[W[row + in_dim]];
            U[i] = to_bf16((float)b_int * weight_scale);
        }

        // step through the K input features in blocks of 16
        for (int k0 = 0; k0 < in_dim; k0 += 16) {

            // T is the int16 accumulator inside the MAC unit, could be int13 later
            int T[8] = {0};

            // multiply and accumulate 16 inputs at a time
            for (int i = 0; i < 8 && (I + i) < out_dim; i++) {
                int row = (I + i) * stride;
                for (int k = 0; k < 16 && (k0 + k) < in_dim; k++) {
                    int w_int = nibble_to_int[W[row + k0 + k]];
                    int a_int = nibble_to_int[A[k0 + k]];
                    T[i] += w_int * a_int;
                }
            }

            // scale the integer sum to float, round to bf16, accumulate into U
            for (int i = 0; i < 8 && (I + i) < out_dim; i++) {
                U[i] = to_bf16(U[i] + (float)T[i] * per_block_scale);
            }
        }

        // write the completed 8 rows out to the final output matrix F
        for (int i = 0; i < 8 && (I + i) < out_dim; i++) {
            F[I + i] = U[i];
        }
    }
}

// hardtanh on float, clamps to [-1.0, 1.0]
void hardtanh_f(float* A, int dim) {
    for (int i = 0; i < dim; i++) {
        if (A[i] >  1.0f) A[i] =  1.0f;
        else if (A[i] < -1.0f) A[i] = -1.0f;
    }
}

// forward pass
// 2 hidden layers with hardtanh activation function
int inference(const int16_t* inputs) {
    static float h1[H1_DIM], h2[H2_DIM], logits[OUT_DIM];
    static int16_t h1_nibbles[H1_DIM], h2_nibbles[H2_DIM];

    gemm(inputs, w1_fp4, h1, IN_DIM, H1_DIM, LAYER1_BIAS);
    hardtanh_f(h1, H1_DIM);
    for (int i = 0; i < H1_DIM; i++) h1_nibbles[i] = fp4_quantize(h1[i]);

    gemm(h1_nibbles, w2_fp4, h2, H1_DIM, H2_DIM, LAYER2_BIAS);
    hardtanh_f(h2, H2_DIM);
    for (int i = 0; i < H2_DIM; i++) h2_nibbles[i] = fp4_quantize(h2[i]);

    gemm(h2_nibbles, w3_fp4, logits, H2_DIM, OUT_DIM, LAYER3_BIAS);

    int highest = 0;
    for (int i = 1; i < OUT_DIM; i++) {
        if (logits[i] > logits[highest]) highest = i;
    }
    return highest;
}

int main(void) {
    static int16_t image_pixels[IN_DIM];

    validate_fp4(w1_fp4, W1_FP4_ROWS * W1_FP4_COLS, "w1_fp4");
    validate_fp4(w2_fp4, W2_FP4_ROWS * W2_FP4_COLS, "w2_fp4");
    validate_fp4(w3_fp4, W3_FP4_ROWS * W3_FP4_COLS, "w3_fp4");

    // pred | truth per sample
    printf("P|T\n");

    int correct = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        // pixels uint8 [0,255] -> float [0,1] -> FP4 nibble
        for (int p = 0; p < IN_DIM; p++) {
            image_pixels[p] = fp4_quantize((float)test_images[i][p] / 255.0f);
        }

        int pred = inference(image_pixels);
        if (pred == test_labels[i]) correct++;
        // printf("%d|%d\n", pred, test_labels[i]);
    }

    printf("\nACCURACY: %d/%d\n", correct, N_SAMPLES);

    return 0;
}
