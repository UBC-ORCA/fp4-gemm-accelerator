// MLP inference baseline on CVE2
// FP4 reinterpreted as signed int4 and multiplied directly (results are wrong)

#include "simple_system_common.h"
#include "weights_encoded.h"
#include "test_data.h"

// MLP dimensions 784 -> 128 -> 96 -> 10
#define IN_DIM    784
#define H1_DIM    128
#define H2_DIM     96
#define OUT_DIM    10

#define DEV_HALT ((volatile int *) 0x20008)

// print unsigned int as decimal
void putdec(uint32_t n) {
    char buf[11];
    int i = 0;
    if (n == 0) { putchar('0'); return; }
    while (n > 0) { buf[i++] = '0' + (n % 10); n /= 10; }
    while (i--) putchar(buf[i]);
}

void* memcpy(void* dst, const void* src, int n) {
    char* d = (char*)dst;
    const char* s = (const char*)src;
    for (int i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

// unpack idx-th nibble (high nibble first) sign-extended to int4 [-8, 7]
static inline int8_t get_weight(const uint8_t* w, int idx) {
    // high (even) or low (odd) nibble
    uint8_t nibble = (idx & 1) ? (w[idx >> 1] & 0x0F) : (w[idx >> 1] >> 4);
    // sign-extend to int4
    return (nibble & 0x8) ? (int8_t)(nibble | 0xF0) : (int8_t)nibble;
}

// gemm: F[j] = bias[j] + sum_k A[k] * W[j,k]
// weights interpreted as raw int4, no FP4 decode or scaling
void gemm(const int16_t* A, const uint8_t* W, int16_t* F, int in_dim, int out_dim) {
    int stride = in_dim + 1;
    for (int j = 0; j < out_dim; j++) {
        int row = j * stride;
        int32_t acc = get_weight(W, row + in_dim);
        for (int k = 0; k < in_dim; k++) {
            acc += (int32_t)A[k] * get_weight(W, row + k);
        }
        // clamp to int16
        F[j] = acc > 32767 ? 32767 : acc < -32768 ? -32768 : (int16_t)acc;
    }
}

// hardtanh-style clamp on raw integer activations to [-256, 256]
void hardtanh(int16_t* a, int dim) {
    for (int i = 0; i < dim; i++) {
        if (a[i] >  256) a[i] = 256;
        else if (a[i] <  -256) a[i] = -256;
    }
}

// forward pass through 2 hardtanh-gated hidden layers, argmax on logits
int inference(const int16_t* image) {
    int16_t h1[H1_DIM], h2[H2_DIM], logits[OUT_DIM];

    gemm(image, w1_fp4, h1, IN_DIM, H1_DIM);
    hardtanh(h1, H1_DIM);

    gemm(h1, w2_fp4, h2, H1_DIM, H2_DIM);
    hardtanh(h2, H2_DIM);

    gemm(h2, w3_fp4, logits, H2_DIM, OUT_DIM);

    int highest = 0;
    for (int i = 1; i < OUT_DIM; i++) {
        if (logits[i] > logits[highest]) highest = i;
    }
    return highest;
}

int main(void) {
    static int16_t image[IN_DIM];

    // pred | truth per sample
    puts("P|T\n");

    int correct = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        // pixels uint8 [0,255] cast to int16 storage
        for (int p = 0; p < IN_DIM; p++)
            image[p] = (int16_t)test_images[i][p];
        int pred = inference(image);
        if (pred == test_labels[i]) correct++;
        putdec(pred);
        puts("|");
        putdec(test_labels[i]);
        puts("\n");
    }

    puts("\nACCURACY: ");
    putdec(correct);
    putchar('/');
    putdec(N_SAMPLES);
    puts("\n");

    // stop sim
    *DEV_HALT = 1;
    return 0;
}
