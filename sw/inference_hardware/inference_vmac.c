// MLP inference on CVE2, Gen 1 + Vector Instruction (a.k.a. Gen 3 W.I.P.)

#include "../headers/weights_blk8_pkgUINT32_scaleE8M0.h"

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

#define BLK_SIZE         K1_STEP_HDR   // K elements per inner block

// performance counters: define PERF_COUNTERS to enable
#define PERF_COUNTERS 1

// UART putchar
extern void putchar_uart(char c);

static void print_str(const char *s) {
  while (*s) putchar_uart(*s++);
}

static void putdec(uint32_t n) {
    char buf[11];
    int i = 0;
    if (n == 0) { putchar_uart('0'); return; }
    while (n > 0) { buf[i++] = '0' + (n % 10); n /= 10; }
    while (i--) putchar_uart(buf[i]);
}

#ifdef PERF_COUNTERS
// read the machine cycle counter; enable zicsr just for this read since the
// base march is rv32im (no CSR ops)
static inline uint32_t rdcyc(void) {
    uint32_t c;
    __asm__ volatile (".option push\n\t"
                      ".option arch, +zicsr\n\t"
                      "csrr %0, mcycle\n\t"
                      ".option pop" : "=r"(c));
    return c;
}

// accumulated cycles per region
static uint64_t pc_imgq;       // image quantize + pack
static uint64_t pc_gemm[3];    // gemm, per layer
static uint64_t pc_qact[2];    // quantize_activation, per layer
static uint64_t pc_macc[3];    // macAcc inside gemm, per layer
static int      pc_layer;      // which layer the running gemm feeds

// PC_LAYER selects the macAcc slot; TIME adds a statement's cycles to acc
#define PC_LAYER(n)     (pc_layer = (n))
#define TIME(acc, stmt) do { uint32_t _t = rdcyc(); stmt; (acc) += (uint32_t)(rdcyc() - _t); } while (0)

static void putdec64(uint64_t n) {   // totals can exceed 32 bits over a full run
    char buf[21]; int i = 0;
    if (n == 0) { putchar_uart('0'); return; }
    while (n > 0) { buf[i++] = '0' + (int)(n % 10); n /= 10; }
    while (i--) putchar_uart(buf[i]);
}

static void pc_line(const char *name, uint64_t v) {
    print_str("  "); print_str(name); print_str(" "); putdec64(v); print_str("\n");
}

static void pc_report(void) {
    print_str("\n[PERF] cycles over run\n");
    pc_line("imgq ", pc_imgq);
    pc_line("gemm1", pc_gemm[0]); pc_line("gemm2", pc_gemm[1]); pc_line("gemm3", pc_gemm[2]);
    pc_line("qact1", pc_qact[0]); pc_line("qact2", pc_qact[1]);
    pc_line("macc1", pc_macc[0]); pc_line("macc2", pc_macc[1]); pc_line("macc3", pc_macc[2]);
    pc_line("gemm ", pc_gemm[0] + pc_gemm[1] + pc_gemm[2]);
    pc_line("qact ", pc_qact[0] + pc_qact[1]);
    pc_line("macc ", pc_macc[0] + pc_macc[1] + pc_macc[2]);
}
#else
#define PC_LAYER(n)     ((void)0)
#define TIME(acc, stmt) do { stmt; } while (0)
#define pc_report()     ((void)0)
#endif

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

// pixel byte (0..255) -> FP4 code, precomputed since inputs are uint8
static uint8_t pix_to_fp4[256];

static void build_pix_lut(void) {
    for (int v = 0; v < 256; v++) {
        pix_to_fp4[v] = (uint8_t)(fp4_quantize((float)v / 255.0f) & 0xF);
    }
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
//  mac_zz()                  -> zzMAC64          clear the 8x8 tile
//  load_vN(ptr)              -> vle32.v vN,(ptr) load 8 packed activation words into vN
//  vmac64_vN(base)           -> VMAC64 vN,N*32,(base) vN x weight block N -> tile (N*32 encoded in instruction)
//  mac_out(row, pair, mode)  -> mv (mode 2=pair) read {tile[row][2p+1],tile[row][2p]}
extern void     mac_zz(void);
extern uint32_t mac_out(uint32_t row, uint32_t pair, uint32_t mode);

extern void load_v0 (uint32_t *ptr);  extern void vmac64_v0 (uint32_t *ptr);
extern void load_v1 (uint32_t *ptr);  extern void vmac64_v1 (uint32_t *ptr);
extern void load_v2 (uint32_t *ptr);  extern void vmac64_v2 (uint32_t *ptr);
extern void load_v3 (uint32_t *ptr);  extern void vmac64_v3 (uint32_t *ptr);
extern void load_v4 (uint32_t *ptr);  extern void vmac64_v4 (uint32_t *ptr);
extern void load_v5 (uint32_t *ptr);  extern void vmac64_v5 (uint32_t *ptr);
extern void load_v6 (uint32_t *ptr);  extern void vmac64_v6 (uint32_t *ptr);
extern void load_v7 (uint32_t *ptr);  extern void vmac64_v7 (uint32_t *ptr);
extern void load_v8 (uint32_t *ptr);  extern void vmac64_v8 (uint32_t *ptr);
extern void load_v9 (uint32_t *ptr);  extern void vmac64_v9 (uint32_t *ptr);
extern void load_v10(uint32_t *ptr);  extern void vmac64_v10(uint32_t *ptr);
extern void load_v11(uint32_t *ptr);  extern void vmac64_v11(uint32_t *ptr);
extern void load_v12(uint32_t *ptr);  extern void vmac64_v12(uint32_t *ptr);
extern void load_v13(uint32_t *ptr);  extern void vmac64_v13(uint32_t *ptr);
extern void load_v14(uint32_t *ptr);  extern void vmac64_v14(uint32_t *ptr);
extern void load_v15(uint32_t *ptr);  extern void vmac64_v15(uint32_t *ptr);
extern void load_v16(uint32_t *ptr);  extern void vmac64_v16(uint32_t *ptr);
extern void load_v17(uint32_t *ptr);  extern void vmac64_v17(uint32_t *ptr);
extern void load_v18(uint32_t *ptr);  extern void vmac64_v18(uint32_t *ptr);
extern void load_v19(uint32_t *ptr);  extern void vmac64_v19(uint32_t *ptr);
extern void load_v20(uint32_t *ptr);  extern void vmac64_v20(uint32_t *ptr);
extern void load_v21(uint32_t *ptr);  extern void vmac64_v21(uint32_t *ptr);
extern void load_v22(uint32_t *ptr);  extern void vmac64_v22(uint32_t *ptr);
extern void load_v23(uint32_t *ptr);  extern void vmac64_v23(uint32_t *ptr);
extern void load_v24(uint32_t *ptr);  extern void vmac64_v24(uint32_t *ptr);
extern void load_v25(uint32_t *ptr);  extern void vmac64_v25(uint32_t *ptr);
extern void load_v26(uint32_t *ptr);  extern void vmac64_v26(uint32_t *ptr);
extern void load_v27(uint32_t *ptr);  extern void vmac64_v27(uint32_t *ptr);
extern void load_v28(uint32_t *ptr);  extern void vmac64_v28(uint32_t *ptr);
extern void load_v29(uint32_t *ptr);  extern void vmac64_v29(uint32_t *ptr);
extern void load_v30(uint32_t *ptr);  extern void vmac64_v30(uint32_t *ptr);
extern void load_v31(uint32_t *ptr);  extern void vmac64_v31(uint32_t *ptr);

// index a vector register by block number
static void (*const load_fn[32])(uint32_t *) = {
    load_v0,  load_v1,  load_v2,  load_v3,  load_v4,  load_v5,  load_v6,  load_v7,
    load_v8,  load_v9,  load_v10, load_v11, load_v12, load_v13, load_v14, load_v15,
    load_v16, load_v17, load_v18, load_v19, load_v20, load_v21, load_v22, load_v23,
    load_v24, load_v25, load_v26, load_v27, load_v28, load_v29, load_v30, load_v31
};
static void (*const vmac_fn[32])(uint32_t *) = {
    vmac64_v0,  vmac64_v1,  vmac64_v2,  vmac64_v3,  vmac64_v4,  vmac64_v5,  vmac64_v6,  vmac64_v7,
    vmac64_v8,  vmac64_v9,  vmac64_v10, vmac64_v11, vmac64_v12, vmac64_v13, vmac64_v14, vmac64_v15,
    vmac64_v16, vmac64_v17, vmac64_v18, vmac64_v19, vmac64_v20, vmac64_v21, vmac64_v22, vmac64_v23,
    vmac64_v24, vmac64_v25, vmac64_v26, vmac64_v27, vmac64_v28, vmac64_v29, vmac64_v30, vmac64_v31
};

// macWs: load 8 per-row weight shifts
static inline void macWs(const uint32_t* words, int row_shift[8]) {
    extract_scale_exp(words, row_shift);
}

// macAs: load 8 per-col activation shifts
static inline void macAs(const uint32_t* words, int col_shift[8]) {
    extract_scale_exp(words, col_shift);
}

// read the 8x8 hardware tile via mac_out
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
    for (int i = 0; i < 8; i++) {                   // padding (0) rows written to F are ignored
        for (int j = 0; j < BATCH; j++) {
            U[i][j] = to_bf16(U[i][j] + scale_pow2((float)T[i][j], shift[i][j]));
        }
    }
}

// F[i][j] = bias[i] + sum_k W[i][k] * A[k][j]
// W is pre-transposed and pre-packed: one uint32 per (block, k-step) = 8 output rows
// A is pre-packed: one uint32 per k-step (8 batch lanes)
// F [out_dim][BATCH] has bf16 bias, then each scaled tile is accumulated onto it
void gemm(const uint32_t* A, const uint32_t* W, const int16_t* bias, float* F,
          int in_dim, int out_dim, const uint32_t* wscale_words, const uint32_t* ascale_words) {
    const int NVREG = 32;     // vector registers v0..v31
    const int TT = 8;         // tile dimension (8x8 MACs)

    int num_K_blocks = (in_dim + BLK_SIZE - 1) / BLK_SIZE;
    int num_I_tiles  = (out_dim + TT - 1) / TT;   // number of 8-row output tiles

    // decode the scaling factors once
    // scale = 2^wshift[i] * 2^ashift[j] = 2^(wshift[i]+ashift[j])
    int wshift[8], ashift[8];
    macWs(wscale_words, wshift);   // per-row weight shifts
    macAs(ascale_words, ashift);   // per-col activation shifts

    int shift[8][BATCH];

    // init F with bias for all output tiles, scaled by the per-row weight shift
    for (int o = 0; o < num_I_tiles; o++) {
        for (int i = 0; i < TT; i++) {
            int I = o * TT + i;
            float b = (I < out_dim) ? to_bf16(scale_pow2((float)bias[I], wshift[i])) : 0.0f;
            for (int j = 0; j < BATCH; j++) {
                F[I * BATCH + j] = b;
            }
        }
    }

    // K accumulation
    // load each chunk's activation blocks into v0..v(n_blk-1) once
    for (int K = 0; K < in_dim; K += NVREG * BLK_SIZE) {
        // number of K-blocks in this chunk (<= 32, one per vector register)
        int n_blk = 0;
        for (int b = 0; b < NVREG && (K + b * BLK_SIZE) < in_dim; b++) n_blk++;

        // load this chunk's activations into v0..v(n_blk-1), once
        for (int b = 0; b < n_blk; b++) {
            load_fn[b]((uint32_t *)&A[K + b * BLK_SIZE]);
        }

        // reuse the loaded activations across all output tiles
        for (int J = 0; J < out_dim; J += TT) {
            int o = J / TT;
            // clear the 8x8 tile before output tile o
            mac_zz();

            const uint32_t* W_base = &W[(o * num_K_blocks + K / BLK_SIZE) * BLK_SIZE];
            for (int b = 0; b < n_blk; b++) {
                vmac_fn[b]((uint32_t *)W_base);
                macAs(ascale_words, ashift);   // apply activation scaling factors
                macWs(wscale_words, wshift);   // apply weight scaling factors
            }
            // combine shift for activations and weights
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < BATCH; j++)
                    shift[i][j] = wshift[i] + ashift[j];
            // read + scale the tile, accumulate onto F's 8 rows for this tile
            TIME(pc_macc[pc_layer], macAcc((float(*)[BATCH])&F[J * BATCH], shift));
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
    // gemm writes all num_I_tiles*8 rows into U, padding rows stay 0
    static float h1[H1_DIM * BATCH], h2[H2_DIM * BATCH], logits[16 * BATCH];
    static uint32_t h1_packed[H1_DIM], h2_packed[H2_DIM];

    PC_LAYER(0);
    TIME(pc_gemm[0], gemm(inputs, w1_fp4, bias1, h1, IN_DIM, H1_DIM, wscale1, ascale1));
    TIME(pc_qact[0], quantize_activation(h1, h1_packed, H1_DIM));

    PC_LAYER(1);
    TIME(pc_gemm[1], gemm(h1_packed, w2_fp4, bias2, h2, H1_DIM, H2_DIM, wscale2, ascale2));
    TIME(pc_qact[1], quantize_activation(h2, h2_packed, H2_DIM));

    PC_LAYER(2);
    TIME(pc_gemm[2], gemm(h2_packed, w3_fp4, bias3, logits, H2_DIM, OUT_DIM, wscale3, ascale3));

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

    build_pix_lut();

    // pred | truth | mptorch reference, per sample
    print_str("P|T|M\n");

    int correct = 0;    // prediction vs ground truth
    int match = 0;      // prediction vs mptorch reference

    for (int s = 0; s < N_SAMPLES; s += BATCH) {
        int n = N_SAMPLES - s;
        if (n > BATCH) n = BATCH;
        int truth[BATCH], ref[BATCH];

        // pack the batch of images
        TIME(pc_imgq, {
            for (int p = 0; p < IN_DIM; p++) image_packed[p] = 0;   // unused lanes stay 0
            for (int j = 0; j < n; j++) {
                *IMG_LOAD = s + j;
                for (int p = 0; p < IN_DIM; p++) {
                    uint32_t code = pix_to_fp4[IMG_STAGE[p]];
                    image_packed[p] |= code << (4 * j);
                }
                truth[j] = *IMG_LABEL;
                ref[j]   = *IMG_PRED;
            }
        });

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

    pc_report();

    return 0;
}