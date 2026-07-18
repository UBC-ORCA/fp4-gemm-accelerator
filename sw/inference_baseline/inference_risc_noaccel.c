// MLP inference baseline on CVE2 
// no acceleration, incorrect results
// FP4 codes reinterpreted as signed int4 and multiplied directly in scalar C
#include "../headers/weights_blk32_pkgINT16_scaleE8M0.h"

#define IMG_LOAD  ((volatile unsigned int  *) 0xFFFF0010)
#define IMG_LABEL ((volatile unsigned int  *) 0xFFFF0014)
#define IMG_PRED  ((volatile unsigned int  *) 0xFFFF0018)
#define IMG_STAGE ((volatile unsigned char *) 0x80070000)

#define N_SAMPLES 80

#define IN_DIM    784
#define L1_DIM    128
#define L2_DIM     96
#define OUT_DIM    10

#define K1_STEP   K1_STEP_HDR   

#define PERF_COUNTERS 1

#define PTM_PRINTS    1

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

#ifdef PERF_COUNTERS
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
static uint64_t pc_qact[2];    
static int      pc_layer;      

#define PC_LAYER(n)     (pc_layer = (n))
#define TIME(acc, stmt) do { uint32_t _t = rdcyc(); stmt; (acc) += (uint32_t)(rdcyc() - _t); } while (0)

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

static void pc_report(void) {
    print_str("\n[PERF] cycles over run\n");
    pc_line("imgq   |", pc_imgq);
    
    pc_line("gemm_1 |", pc_gemm[0]); 
    pc_line("gemm_2 |", pc_gemm[1]); 
    pc_line("gemm_3 |", pc_gemm[2]);
    
    pc_line("qact_1 |", pc_qact[0]); 
    pc_line("qact_2 |", pc_qact[1]);
    
    print_str("\n[PERF] total cycles over run\n");
    pc_line("gemm_T |", pc_gemm[0] + pc_gemm[1] + pc_gemm[2]);
    pc_line("qact_T |", pc_qact[0] + pc_qact[1]);
}
#else
#define PC_LAYER(n)     ((void)0)
#define TIME(acc, stmt) do { stmt; } while (0)
#define pc_report()     ((void)0)
#endif

static inline int fp4_as_int4(int16_t code) {
    code &= 0xF;
    return (code & 0x8) ? (int)(code - 16) : (int)code;
}

void gemm(const int16_t* A, const int16_t* W, const int16_t* bias,
          int16_t* out, int in_dim, int out_dim) {
#ifdef PERF_COUNTERS
    uint32_t _gt0 = rdcyc();
#endif
    int num_K_blocks = (in_dim + K1_STEP - 1) / K1_STEP;

    for (int i = 0; i < out_dim; i++) {
        int I_tile = i / 8;             
        int row    = i % 8;             
        int32_t acc = bias[i];          

        for (int k = 0; k < in_dim; k++) {
            int K_block = k / K1_STEP;
            int K1      = k % K1_STEP;
            int idx = ((I_tile * num_K_blocks + K_block) * K1_STEP + K1) * 8 + row;
            acc += (int32_t)A[k] * fp4_as_int4(W[idx]);
        }

        out[i] = acc > 32767 ? 32767 : acc < -32768 ? -32768 : (int16_t)acc;
    }
#ifdef PERF_COUNTERS
    pc_gemm[pc_layer] += (uint32_t)(rdcyc() - _gt0);
#endif
}

static void hardtanh(int16_t* a, int dim) {
    for (int i = 0; i < dim; i++) {
        if (a[i] >  4) a[i] =  4;
        if (a[i] < -4) a[i] = -4;
    }
}

int inference(const int16_t* image) {
    int16_t out_l1[L1_DIM], out_l2[L2_DIM], logits[OUT_DIM];

    PC_LAYER(0); 
    gemm(image, w1_fp4, bias1, out_l1, IN_DIM, L1_DIM);
    TIME(pc_qact[0], hardtanh(out_l1, L1_DIM));

    PC_LAYER(1); 
    gemm(out_l1, w2_fp4, bias2, out_l2, L1_DIM, L2_DIM);
    TIME(pc_qact[1], hardtanh(out_l2, L2_DIM));

    PC_LAYER(2); 
    gemm(out_l2, w3_fp4, bias3, logits, L2_DIM, OUT_DIM);

    int best = 0;
    for (int i = 1; i < OUT_DIM; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

int main(void) {
    static int16_t image[IN_DIM];

    #ifdef PTM_PRINTS
    print_str("P|T|M\n");
    #endif

    int correct = 0;
    int match = 0;      
    for (int s = 0; s < N_SAMPLES; s++) {
        TIME(pc_imgq, {
            *IMG_LOAD = s;
            for (int p = 0; p < IN_DIM; p++) {
                image[p] = (int16_t)IMG_STAGE[p];
            }
        });
        int truth = *IMG_LABEL;
        int ref   = *IMG_PRED;

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