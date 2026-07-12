// MLP inference baseline on CVE2 
// no acceleration, incorrect results
// FP4 codes reinterpreted as signed int4 and multiplied directly in scalar C

#include "../headers/weights_blk8_pkgINT16_scaleE8M0.h"

// number of streamed MNIST samples (must match the .bin passed to the TB)
#define N_SAMPLES 80

// test-data streaming, same MMIO ports as the accelerated builds; writing s
// to IMG_LOAD makes the TB stage image s into DMEM starting at IMG_STAGE
#define IMG_LOAD  ((volatile unsigned int  *) 0xFFFF0010)
#define IMG_LABEL ((volatile unsigned int  *) 0xFFFF0014)
#define IMG_PRED  ((volatile unsigned int  *) 0xFFFF0018)
#define IMG_STAGE ((volatile unsigned char *) 0x80070000)

// MLP Dimensions: 784 -> 128 -> 96 -> 10
#define IN_DIM    784
#define H1_DIM    128
#define H2_DIM     96
#define OUT_DIM    10

#define K1_STEP   K1_STEP_HDR   // K elements per inner block

// UART putchar provided by uart.c
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

// the "wrong" part: take a raw FP4 code and read its 4 bits as a signed int4
// [-8, 7] instead of decoding the FP4 value. no LUT, no scaling.
static inline int fp4_as_int4(int16_t code) {
    code &= 0xF;
    return (code & 0x8) ? (int)(code - 16) : (int)code;
}

// gemm: F[i] = bias[i] + sum_k A[k] * W[i][k]   (scalar matrix x vector)
// W is the K1-strip transposed FP4 layout from the header:
//   W[i][k] = w_fp4[((I_tile * num_K_blocks + K_block) * K1_STEP + K1) * 8 + row]
// bias is a separate decoded-int array. no scaling is applied anywhere.
void gemm(const int16_t* A, const int16_t* W, const int16_t* bias,
          int16_t* F, int in_dim, int out_dim) {
    int num_K_blocks = (in_dim + K1_STEP - 1) / K1_STEP;

    for (int i = 0; i < out_dim; i++) {
        int I_tile = i / 8;             // which 8-row tile
        int row    = i % 8;             // row within the tile
        int32_t acc = bias[i];          // raw bias, unscaled

        for (int k = 0; k < in_dim; k++) {
            int K_block = k / K1_STEP;
            int K1      = k % K1_STEP;
            int idx = ((I_tile * num_K_blocks + K_block) * K1_STEP + K1) * 8 + row;
            acc += (int32_t)A[k] * fp4_as_int4(W[idx]);
        }

        // clamp to int16 (int32 intermediate avoids mid-accumulation overflow)
        F[i] = acc > 32767 ? 32767 : acc < -32768 ? -32768 : (int16_t)acc;
    }
}

// ReLU activation on the raw integer activations (matches the trained model)
static void relu(int16_t* a, int dim) {
    for (int i = 0; i < dim; i++) {
        if (a[i] < 0) a[i] = 0;
    }
}

// forward pass: 2 ReLU-gated hidden layers, argmax on logits
int inference(const int16_t* image) {
    int16_t h1[H1_DIM], h2[H2_DIM], logits[OUT_DIM];

    gemm(image, w1_fp4, bias1, h1, IN_DIM, H1_DIM);
    relu(h1, H1_DIM);

    gemm(h1, w2_fp4, bias2, h2, H1_DIM, H2_DIM);
    relu(h2, H2_DIM);

    gemm(h2, w3_fp4, bias3, logits, H2_DIM, OUT_DIM);

    int best = 0;
    for (int i = 1; i < OUT_DIM; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

int main(void) {
    static int16_t image[IN_DIM];

    // pred | truth | mptorch reference, per sample (M is meaningless here since
    // the baseline math is wrong, printed only for output-format consistency)
    print_str("P|T|M\n");

    int correct = 0;
    int match = 0;      // baseline prediction vs mptorch reference
    for (int s = 0; s < N_SAMPLES; s++) {
        *IMG_LOAD = s;                         // TB stages image s into DMEM at IMG_STAGE
        // raw pixels uint8 [0,255] cast to int16 (baseline uses raw pixels, no FP4)
        for (int p = 0; p < IN_DIM; p++) {
            image[p] = (int16_t)IMG_STAGE[p];
        }
        int truth = *IMG_LABEL;
        int ref   = *IMG_PRED;

        int pred = inference(image);
        if (pred == truth) correct++;
        if (pred == ref)   match++;

        putdec(pred);
        print_str("|");
        putdec(truth);
        print_str("|");
        putdec(ref);
        print_str("\n");
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