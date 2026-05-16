// MLP inference on x86
// FP4 weight values stored as int16, accumulation and activations in Q8.8

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include "../headers/weights_int16.h"
#include "../headers/test_data.h"

// MLP Dimensions: 784 -> 128 -> 96 -> 10
#define IN_DIM    784
#define H1_DIM    128
#define H2_DIM     96
#define OUT_DIM    10

// Q8.8 fixed-point parameters
#define FL          8
#define ONE_Q     256

// nibble to int5 LUT
static const int8_t nibble_to_int[16] = {
//  0  1  2  3  4  5  6   7   8  9 10 11 12 13 14  15
    0, 1, 2, 3, 4, 6, 8, 12,  0,-1,-2,-3,-4,-6,-8,-12
};

// verify each entry is a valid FP4 and not raw int16
void validate_fp4(const int16_t* w, int n, const char* name) {
    // mark whcih nibble values appear in the weights
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

// gemm: F[j] = bias[j] + sum_k A[k] * W[j,k]
// activations and weights Q8.8 int16, accumulator int32
// q_step is smallest step for FP4 in Q8.8 (64 >> layer_bias)
void gemm(const int16_t* A, const int16_t* W, int16_t* F, int in_dim, int out_dim, int layer_bias) {
    int stride = in_dim + 1;
    int q_step = 64 >> layer_bias;

    for (int j = 0; j < out_dim; j++) {
        int row = j * stride;
        // bias promoted to Q16.16
        int32_t acc = ((int32_t)nibble_to_int[W[row + in_dim]] * q_step) << FL;
        for (int k = 0; k < in_dim; k++) {
            int16_t w_q8 = (int16_t)((int)nibble_to_int[W[row + k]] * q_step);
            acc += (int32_t)A[k] * (int32_t)w_q8;
        }
        // round, shift back to Q8.8, clamp to int16
        int32_t shifted = (acc + (1 << (FL - 1))) >> FL;
        F[j] = shifted > 32767 ? 32767 : shifted < -32768 ? -32768 : (int16_t)shifted;
    }
}

// hardtanh clamps Q8.8 activations to [-ONE_Q, ONE_Q] = [-1.0, 1.0]
void hardtanh(int16_t* a, int dim) {
    for (int i = 0; i < dim; i++) {
        if (a[i] >  ONE_Q) a[i] = ONE_Q;
        else if (a[i] <  -ONE_Q) a[i] = -ONE_Q;
    }
}

// forward pass through 2 hardtanh-gated hidden layers, argmax on logits
int inference(const int16_t* image) {
    int16_t h1[H1_DIM], h2[H2_DIM], logits[OUT_DIM];

    gemm(image, w1_fp4, h1, IN_DIM, H1_DIM, LAYER1_BIAS);
    hardtanh(h1, H1_DIM);

    gemm(h1, w2_fp4, h2, H1_DIM, H2_DIM, LAYER2_BIAS);
    hardtanh(h2, H2_DIM);

    gemm(h2, w3_fp4, logits, H2_DIM, OUT_DIM, LAYER3_BIAS);

    int highest = 0;
    for (int i = 1; i < OUT_DIM; i++) {
        if (logits[i] > logits[highest]) highest = i;
    }
    return highest;
}

int main(void) {
    static int16_t image[IN_DIM];

    validate_fp4(w1_fp4, W1_FP4_ROWS * W1_FP4_COLS, "w1_fp4");
    validate_fp4(w2_fp4, W2_FP4_ROWS * W2_FP4_COLS, "w2_fp4");
    validate_fp4(w3_fp4, W3_FP4_ROWS * W3_FP4_COLS, "w3_fp4");

    // pred | truth per sample
    printf("P|T\n");

    int correct = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        // pixels uint8 [0,255] to Q8.8
        for (int p = 0; p < IN_DIM; p++)
            image[p] = (int16_t)(((int32_t)test_images[i][p] << FL) / 255);
        int pred = inference(image);
        if (pred == test_labels[i]) correct++;
        printf("%d|%d\n", pred, test_labels[i]);
    }

    printf("\nACCURACY: %d/%d\n", correct, N_SAMPLES);

    return 0;
}
