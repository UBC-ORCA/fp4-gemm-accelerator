// MLP inference on CVE2, Gen 1 + Vector Instruction (a.k.a. Gen 3 W.I.P.)

#include "../headers/weights_blk8_pkgUINT32_scaleE8M0.h"

// Samples
#define N_SAMPLES 80

// Test data loading
#define IMG_LOAD  ((volatile unsigned int  *) 0xFFFF0010)
#define IMG_LABEL ((volatile unsigned int  *) 0xFFFF0014)
#define IMG_PRED  ((volatile unsigned int  *) 0xFFFF0018)
#define IMG_STAGE ((volatile unsigned char *) 0x80070000)

// Perf markers to measure GEMM speed
#define PERF_START ((volatile unsigned int *) 0xFFFF0004)
#define PERF_END   ((volatile unsigned int *) 0xFFFF0008)

// MLP Dimensions: 784 -> 128 -> 96 -> 10
#define IN_DIM    784
#define H1_DIM    128
#define H2_DIM     96
#define OUT_DIM    10

// number of MNIST samples solved in parallel (J dimension of A)
#define BATCH      8

#define K1_STEP    K1_STEP_HDR   // K elements per inner block
#define K2_SPAN    256           // K elements per K3 step (32 inner blocks)

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

// real MAC64 hardware instructions (matmul8_vec.S)
//  vset8()                   -> vsetvli          vl=8, e32 (one call per gemm)
//  mac_zz()                  -> zzMAC64          clear the 8x8 tile
//  load_vN(ptr)              -> vle32.v vN,(ptr) load 8 packed activation words into vN
//  vmac_vN(ptr)              -> VMAC64 vN,0,(ptr) vN x packed weight block -> tile
//  mac_out(row, pair, mode)  -> mv (mode 2=pair) read {tile[row][2p+1],tile[row][2p]}
extern void     vset8(void);
extern void     mac_zz(void);
extern uint32_t mac_out(uint32_t row, uint32_t pair, uint32_t mode);

extern void load_v0(uint32_t *ptr);  extern void vmac_v0(uint32_t *ptr);
// extern void load_v1(uint32_t *ptr);  extern void vmac_v1(uint32_t *ptr);
// extern void load_v2(uint32_t *ptr);  extern void vmac_v2(uint32_t *ptr);
// extern void load_v3(uint32_t *ptr);  extern void vmac_v3(uint32_t *ptr);
// extern void load_v4(uint32_t *ptr);  extern void vmac_v4(uint32_t *ptr);
// extern void load_v5(uint32_t *ptr);  extern void vmac_v5(uint32_t *ptr);
// extern void load_v6(uint32_t *ptr);  extern void vmac_v6(uint32_t *ptr);
// extern void load_v7(uint32_t *ptr);  extern void vmac_v7(uint32_t *ptr);
// extern void load_v8(uint32_t *ptr);  extern void vmac_v8(uint32_t *ptr);
// extern void load_v9(uint32_t *ptr);  extern void vmac_v9(uint32_t *ptr);
// extern void load_v10(uint32_t *ptr); extern void vmac_v10(uint32_t *ptr);
// extern void load_v11(uint32_t *ptr); extern void vmac_v11(uint32_t *ptr);
// extern void load_v12(uint32_t *ptr); extern void vmac_v12(uint32_t *ptr);
// extern void load_v13(uint32_t *ptr); extern void vmac_v13(uint32_t *ptr);
// extern void load_v14(uint32_t *ptr); extern void vmac_v14(uint32_t *ptr);
// extern void load_v15(uint32_t *ptr); extern void vmac_v15(uint32_t *ptr);
// extern void load_v16(uint32_t *ptr); extern void vmac_v16(uint32_t *ptr);
// extern void load_v17(uint32_t *ptr); extern void vmac_v17(uint32_t *ptr);
// extern void load_v18(uint32_t *ptr); extern void vmac_v18(uint32_t *ptr);
// extern void load_v19(uint32_t *ptr); extern void vmac_v19(uint32_t *ptr);
// extern void load_v20(uint32_t *ptr); extern void vmac_v20(uint32_t *ptr);
// extern void load_v21(uint32_t *ptr); extern void vmac_v21(uint32_t *ptr);
// extern void load_v22(uint32_t *ptr); extern void vmac_v22(uint32_t *ptr);
// extern void load_v23(uint32_t *ptr); extern void vmac_v23(uint32_t *ptr);
// extern void load_v24(uint32_t *ptr); extern void vmac_v24(uint32_t *ptr);
// extern void load_v25(uint32_t *ptr); extern void vmac_v25(uint32_t *ptr);
// extern void load_v26(uint32_t *ptr); extern void vmac_v26(uint32_t *ptr);
// extern void load_v27(uint32_t *ptr); extern void vmac_v27(uint32_t *ptr);
// extern void load_v28(uint32_t *ptr); extern void vmac_v28(uint32_t *ptr);
// extern void load_v29(uint32_t *ptr); extern void vmac_v29(uint32_t *ptr);
// extern void load_v30(uint32_t *ptr); extern void vmac_v30(uint32_t *ptr);
// extern void load_v31(uint32_t *ptr); extern void vmac_v31(uint32_t *ptr);

// load block index into its vector register
static void load_block(int b, uint32_t *ptr) {
    switch (b) {
        case 0:  load_v0(ptr);  break;
        // only v0 works
        // case 1:  load_v1(ptr);  break;
        // case 2:  load_v2(ptr);  break;  case 3:  load_v3(ptr);  break;
        // case 4:  load_v4(ptr);  break;  case 5:  load_v5(ptr);  break;
        // case 6:  load_v6(ptr);  break;  case 7:  load_v7(ptr);  break;
        // case 8:  load_v8(ptr);  break;  case 9:  load_v9(ptr);  break;
        // case 10: load_v10(ptr); break;  case 11: load_v11(ptr); break;
        // case 12: load_v12(ptr); break;  case 13: load_v13(ptr); break;
        // case 14: load_v14(ptr); break;  case 15: load_v15(ptr); break;
        // case 16: load_v16(ptr); break;  case 17: load_v17(ptr); break;
        // case 18: load_v18(ptr); break;  case 19: load_v19(ptr); break;
        // case 20: load_v20(ptr); break;  case 21: load_v21(ptr); break;
        // case 22: load_v22(ptr); break;  case 23: load_v23(ptr); break;
        // case 24: load_v24(ptr); break;  case 25: load_v25(ptr); break;
        // case 26: load_v26(ptr); break;  case 27: load_v27(ptr); break;
        // case 28: load_v28(ptr); break;  case 29: load_v29(ptr); break;
        // case 30: load_v30(ptr); break;  case 31: load_v31(ptr); break;
    }
}
static void vmac_block(int b, uint32_t *ptr) {
    switch (b) {
        case 0:  vmac_v0(ptr);  break;
        // only v0 works in hardware (VMAC64 vs1 index not wired); v1..v31 disabled.
        // case 1:  vmac_v1(ptr);  break;
        // case 2:  vmac_v2(ptr);  break;  case 3:  vmac_v3(ptr);  break;
        // case 4:  vmac_v4(ptr);  break;  case 5:  vmac_v5(ptr);  break;
        // case 6:  vmac_v6(ptr);  break;  case 7:  vmac_v7(ptr);  break;
        // case 8:  vmac_v8(ptr);  break;  case 9:  vmac_v9(ptr);  break;
        // case 10: vmac_v10(ptr); break;  case 11: vmac_v11(ptr); break;
        // case 12: vmac_v12(ptr); break;  case 13: vmac_v13(ptr); break;
        // case 14: vmac_v14(ptr); break;  case 15: vmac_v15(ptr); break;
        // case 16: vmac_v16(ptr); break;  case 17: vmac_v17(ptr); break;
        // case 18: vmac_v18(ptr); break;  case 19: vmac_v19(ptr); break;
        // case 20: vmac_v20(ptr); break;  case 21: vmac_v21(ptr); break;
        // case 22: vmac_v22(ptr); break;  case 23: vmac_v23(ptr); break;
        // case 24: vmac_v24(ptr); break;  case 25: vmac_v25(ptr); break;
        // case 26: vmac_v26(ptr); break;  case 27: vmac_v27(ptr); break;
        // case 28: vmac_v28(ptr); break;  case 29: vmac_v29(ptr); break;
        // case 30: vmac_v30(ptr); break;  case 31: vmac_v31(ptr); break;
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

// read the 8x8 hardware tile via mac_out
// the tile is transposed vs ours (hw row = batch, col = output), so T[i][j] = tile[j][i].
static void read_tile(int16_t T[8][BATCH]) {
    for (int j = 0; j < 8; j++) {                 // hardware row = batch sample
        for (int pair = 0; pair < 4; pair++) {    // each pair = two output neurons
            uint32_t packed = mac_out((uint32_t)j, (uint32_t)pair, 2);
            T[2 * pair]     [j] = (int16_t)(packed & 0xFFFF);   // output i = 2*pair
            T[2 * pair + 1] [j] = (int16_t)(packed >> 16);      // output i = 2*pair+1
        }
    }
}

// macAcc: read the hardware tile, scale each entry (block shift), accumulate into U bf16
static void macAcc(float U[8][BATCH], const int shift[8][BATCH]) {
    int16_t T[8][BATCH];
    read_tile(T);                                   // 32x mac_out from the HW tile
    for (int i = 0; i < 8; i++) {                   // padding rows discarded at the F write
        for (int j = 0; j < BATCH; j++) {
            U[i][j] = to_bf16(U[i][j] + scale_pow2((float)T[i][j], shift[i][j]));
        }
    }
}

// F[i][j] = bias[i] + sum_k W[i][k] * A[k][j]
// W is pre-transposed and pre-packed: one uint32 per (block, k-step) = 8 output rows
// A is pre-packed: one uint32 per k-step (8 batch lanes); 
// F [out_dim][BATCH] has bf16 bias, then each scaled tile is accumulated onto it
void gemm(const uint32_t* A, const uint32_t* W, const int16_t* bias, float* F,
          int in_dim, int out_dim,
          const uint32_t* wscale_words, const uint32_t* ascale_words) {
    int num_K_blocks = (in_dim + K1_STEP - 1) / K1_STEP;
    int num_I_tiles  = (out_dim + 7) / 8;

    // decode the per-block scales once
    // scale = 2^wshift[i] * 2^ashift[j] = 2^(wshift[i]+ashift[j])
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

    vset8();   // vl=8, e32 for every load_vN / vmac_vN below

    // init F with bias for all output tiles, scaled by the per-row weight shift
    for (int It = 0; It < num_I_tiles; It++) {
        for (int i = 0; i < 8; i++) {
            int I = It * 8 + i;
            float b = (I < out_dim) ? to_bf16(scale_pow2((float)bias[I], wshift[i])) : 0.0f;
            for (int j = 0; j < BATCH; j++) {
                F[I * BATCH + j] = b;
            }
        }
    }

    // K accumulation
    for (int K3 = 0; K3 < in_dim; K3 += K2_SPAN) {
        // number of K-blocks in this chunk
        int n_blk = 0;
        for (int K2 = 0; K2 < K2_SPAN && (K3 + K2) < in_dim; K2 += K1_STEP) n_blk++;

        for (int It = 0; It < num_I_tiles; It++) {
            mac_zz();                                     // clear the tile for this chunk
            for (int b = 0; b < n_blk; b++) {
                int K_block = K3 / K1_STEP + b;
                const uint32_t* W_strip = &W[(It * num_K_blocks + K_block) * K1_STEP];
                // v0-only: reload activations into v0 for each block
                load_block(0, (uint32_t *)&A[K3 + b * K1_STEP]);   // v0 <- 8 packed act words
                vmac_block(0, (uint32_t *)W_strip);                // v0 x packed weights -> tile
            }
            // read + scale the tile, accumulate onto F's 8 rows for this tile
            macAcc((float(*)[BATCH])&F[It * 8 * BATCH], shift);
        }
    }
}

// hardtanh
static inline int hardtanh(int a) {
    if (a >  4) return  4;
    if (a < -4) return -4;
    return a;
}

// quantize to FP4 and pack the 8 fp4 codes per uint32 word
static void quantize_activation(const float* A, uint32_t* packed, int dim) {
    for (int d = 0; d < dim; d++) {
        uint32_t word = 0;
        for (int j = 0; j < BATCH; j++) {
            float a = A[d * BATCH + j];
            // scale by 4 and round to nearest int (ties away from zero)
            int z = (int)(a * 4.0f + (a >= 0.0f ? 0.5f : -0.5f));
            z = hardtanh(z);

            int sign = (z < 0);
            int abs_v = sign ? -z : z;
            uint8_t mag = fp4_mag_lut[abs_v];
            uint32_t code = mag == 0 ? 0 : (uint32_t)(sign ? (0x8 | mag) : mag);
            word |= (code & 0xF) << (4 * j);
        }
        packed[d] = word;
    }
}

// forward pass on a batch of BATCH samples
// 2 hidden layers with hardtanh activation function
void inference_batch(const uint32_t* inputs, int* preds) {
    // logits padded to a whole number of 8-row tiles (OUT_DIM=10 -> 2 tiles = 16 rows):
    // gemm writes all num_I_tiles*8 rows into the U spill, padding rows past OUT_DIM stay 0.
    static float h1[H1_DIM * BATCH], h2[H2_DIM * BATCH], logits[16 * BATCH];
    static uint32_t h1_packed[H1_DIM], h2_packed[H2_DIM];

    *PERF_START = 1;    // begin timed region -> TB reports KERNEL_CYCLES
    gemm(inputs, w1_fp4, bias1, h1, IN_DIM, H1_DIM, wscale1, ascale1);
    *PERF_END = 1;      // end timed region
    quantize_activation(h1, h1_packed, H1_DIM);

    gemm(h1_packed, w2_fp4, bias2, h2, H1_DIM, H2_DIM, wscale2, ascale2);
    quantize_activation(h2, h2_packed, H2_DIM);

    gemm(h2_packed, w3_fp4, bias3, logits, H2_DIM, OUT_DIM, wscale3, ascale3);

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
    static uint32_t image_packed[IN_DIM];   // one word per pixel, 8 batch lanes packed
    int preds[BATCH];

    // pred | truth | mptorch reference, per sample
    print_str("P|T|M\n");

    int correct = 0;
    int match = 0;      // sw prediction vs mptorch reference

    for (int s = 0; s < N_SAMPLES; s += BATCH) {
        int n = N_SAMPLES - s;
        if (n > BATCH) n = BATCH;
        int truth[BATCH], ref[BATCH];

        // pack the batch of images: nibble j of image_packed[p] = pixel p of sample j
        for (int p = 0; p < IN_DIM; p++) image_packed[p] = 0;   // unused lanes stay 0
        for (int j = 0; j < n; j++) {
            *IMG_LOAD = s + j;                 // TB stages this image into DMEM
            for (int p = 0; p < IN_DIM; p++) {
                uint32_t code = (uint32_t)(fp4_quantize((float)IMG_STAGE[p] / 255.0f) & 0xF);
                image_packed[p] |= code << (4 * j);
            }
            truth[j] = *IMG_LABEL;
            ref[j]   = *IMG_PRED;
        }

        inference_batch(image_packed, preds);

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