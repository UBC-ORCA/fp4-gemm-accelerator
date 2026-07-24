#include <stdint.h>

extern void putchar_uart(char c);
extern void matmul8_vec(const volatile uint32_t *a,
                        const volatile uint32_t *b,
                        volatile uint32_t *c,
                        volatile uint32_t *tmp_prod);

#define MAT_N 8
#define TT 8
#define BS 32
#define WORDS_PER_VREG 32
//#define NUM_VREGS 32
#define NUM_VREGS 2

static volatile uint32_t *const DONE_MMIO       = (volatile uint32_t *)0xFFFF0000u;
static volatile uint32_t *const COMP_START_MMIO = (volatile uint32_t *)0xFFFF0004u;
static volatile uint32_t *const COMP_END_MMIO   = (volatile uint32_t *)0xFFFF0008u;

// 32 vregs x 32 words = 1024 words (4096 Bytes)
static volatile uint32_t mat_a[WORDS_PER_VREG * NUM_VREGS]
    __attribute__((section(".mat_a"), used));

// Declared as a standard global array, placing it directly into default RAM
uint32_t mat_b[MAT_N * MAT_N];

// Declared as a standard global array, placing it directly into default RAM
uint32_t weights[WORDS_PER_VREG * NUM_VREGS];

static volatile uint32_t mat_c[MAT_N * MAT_N]
    __attribute__((section(".mat_c"), used));
static volatile uint32_t tmp_prod[MAT_N]
    __attribute__((section(".tmp_prod"), used));

static void print_u32_hex(uint32_t x) {
  const char *hex = "0123456789abcdef";
  for (int i = 7; i >= 0; --i) putchar_uart(hex[(x >> (i * 4)) & 0xFu]);
}

static void print_str(const char *s) {
  while (*s) putchar_uart(*s++);
}

// Low-level controls
extern void mac_zz(void);
extern void mac_hw(uint32_t a, uint32_t b);
extern uint32_t mac_out_even(void);
extern uint32_t mac_out_odd(void);
extern uint32_t mac_out_pair(void);
extern void mac_max(int16_t threshold);
extern void mac_add_row(uint32_t row, int16_t value);
extern void mac_ld2(void *base);
extern void mac_st2(void *base);

// Load vector registers (32-word VLEN entries)
extern void load_v0(volatile uint32_t *ptr);
extern void load_v1(volatile uint32_t *ptr);
extern void load_v2(volatile uint32_t *ptr);
extern void load_v3(volatile uint32_t *ptr);
extern void load_v4(volatile uint32_t *ptr);
extern void load_v5(volatile uint32_t *ptr);
extern void load_v6(volatile uint32_t *ptr);
extern void load_v7(volatile uint32_t *ptr);
extern void load_v8(volatile uint32_t *ptr);
extern void load_v9(volatile uint32_t *ptr);
extern void load_v10(volatile uint32_t *ptr);
extern void load_v11(volatile uint32_t *ptr);
extern void load_v12(volatile uint32_t *ptr);
extern void load_v13(volatile uint32_t *ptr);
extern void load_v14(volatile uint32_t *ptr);
extern void load_v15(volatile uint32_t *ptr);
extern void load_v16(volatile uint32_t *ptr);
extern void load_v17(volatile uint32_t *ptr);
extern void load_v18(volatile uint32_t *ptr);
extern void load_v19(volatile uint32_t *ptr);
extern void load_v20(volatile uint32_t *ptr);
extern void load_v21(volatile uint32_t *ptr);
extern void load_v22(volatile uint32_t *ptr);
extern void load_v23(volatile uint32_t *ptr);
extern void load_v24(volatile uint32_t *ptr);
extern void load_v25(volatile uint32_t *ptr);
extern void load_v26(volatile uint32_t *ptr);
extern void load_v27(volatile uint32_t *ptr);
extern void load_v28(volatile uint32_t *ptr);
extern void load_v29(volatile uint32_t *ptr);
extern void load_v30(volatile uint32_t *ptr);
extern void load_v31(volatile uint32_t *ptr);

// Memory-backed VMAC operations
extern void mac_mem_test_v0(uint32_t *ptr);
extern void mac_mem_test_v1(uint32_t *ptr);
extern void mac_mem_test_v2(uint32_t *ptr);
extern void mac_mem_test_v3(uint32_t *ptr);
extern void mac_mem_test_v4(uint32_t *ptr);
extern void mac_mem_test_v5(uint32_t *ptr);
extern void mac_mem_test_v6(uint32_t *ptr);
extern void mac_mem_test_v7(uint32_t *ptr);
extern void mac_mem_test_v8(uint32_t *ptr);
extern void mac_mem_test_v9(uint32_t *ptr);
extern void mac_mem_test_v10(uint32_t *ptr);
extern void mac_mem_test_v11(uint32_t *ptr);
extern void mac_mem_test_v12(uint32_t *ptr);
extern void mac_mem_test_v13(uint32_t *ptr);
extern void mac_mem_test_v14(uint32_t *ptr);
extern void mac_mem_test_v15(uint32_t *ptr);
extern void mac_mem_test_v16(uint32_t *ptr);
extern void mac_mem_test_v17(uint32_t *ptr);
extern void mac_mem_test_v18(uint32_t *ptr);
extern void mac_mem_test_v19(uint32_t *ptr);
extern void mac_mem_test_v20(uint32_t *ptr);
extern void mac_mem_test_v21(uint32_t *ptr);
extern void mac_mem_test_v22(uint32_t *ptr);
extern void mac_mem_test_v23(uint32_t *ptr);
extern void mac_mem_test_v24(uint32_t *ptr);
extern void mac_mem_test_v25(uint32_t *ptr);
extern void mac_mem_test_v26(uint32_t *ptr);
extern void mac_mem_test_v27(uint32_t *ptr);
extern void mac_mem_test_v28(uint32_t *ptr);
extern void mac_mem_test_v29(uint32_t *ptr);
extern void mac_mem_test_v30(uint32_t *ptr);
extern void mac_mem_test_v31(uint32_t *ptr);

// Scale configuration
//uint32_t weight_scales[2] = {0x40404040, 0x40404040};
//uint32_t act_scales[2]    = {0x40404040, 0x40404040};
uint32_t weight_scales[2] = {0x80808080, 0x80808080};
uint32_t act_scales[2]    = {0x80808080, 0x80808080};


extern void load_act_scales(const uint32_t *base);
extern void load_w_scales(const uint32_t *base);
extern void mac_as(void);
extern void mac_ws(void);

// Mode: 0 = even, 1 = odd, 2 = pair
extern uint32_t mac_out(uint32_t row, uint32_t pair, uint32_t mode);

//BRAM
extern void mac_bias(uint8_t tile, uint8_t row, uint8_t column, uint16_t bf16);

int main(void)
{
  uint32_t chk = 0;

  // 1. Initialize Activation Matrix (mat_a) with easy-to-track visual patterns
  for (int v = 0; v < NUM_VREGS; v++) {
    for (int i = 0; i < WORDS_PER_VREG; i++) {
//      mat_a[v * WORDS_PER_VREG + i] = 0x11111111 * (v + 1);
      mat_a[v * WORDS_PER_VREG + i] = 0x11111111;

    }
  }

  // 2. Initialize Weight Matrix to sequential tracking offsets
/*
  for (int i = 0; i < WORDS_PER_VREG * NUM_VREGS; i++) {
    weights[i] = i + 1;
  }
*/

// Initialize Weight Matrix with readable packed FP4 values
// Each uint32_t contains 8 FP4 values (4 bits each)
// Pattern: 0x11111111 ... 0x88888888, repeat

for (int i = 0; i < WORDS_PER_VREG * NUM_VREGS; i++) {
    uint32_t fp4 = (i % 8) + 1;   // 1..8
    weights[i] = fp4 |
                 (fp4 << 4) |
                 (fp4 << 8) |
                 (fp4 << 12) |
                 (fp4 << 16) |
                 (fp4 << 20) |
                 (fp4 << 24) |
                 (fp4 << 28);
}

  // Begin hardware performance profiling
  *COMP_START_MMIO = 1u;

  // Set up standard scaling scales for validation pipeline


//  mac_bias(1,0,0,0x00aa);   // distinctive value to trace how the scale-fold reads propagate it, too small to show
//mac_bias(0,0,0,0x3f80); // too small to show
   mac_bias(0,0,0,0x4120);   // tile0 row7 col7 (ODD row - must not assert or spill to tile1)
   mac_bias(0,7,7,0x4120);   // tile0 row7 col7 (ODD row - must not assert or spill to tile1)
  mac_bias(1,0,0,0x4120);   // distinctive value to trace how the scale-fold reads propagate it

   load_act_scales(act_scales);
  load_w_scales(weight_scales);

  // ==========================================
  // BRING-UP TEST 1: Register v0 Verification
  // ==========================================
  load_v0(&mat_a[0]);               // Loads v0 (Words 0 to 31)
  mac_zz();                         // Clear accumulator tile
 mac_mem_test_v0(weights);         // Multiplies v0 by weights[0..31]
  mac_as();                         // Apply activation scales
  mac_ws();                         // Apply weight scales

  // ==========================================
  // BRING-UP TEST 2: Register v1 Verification
  // ==========================================
  load_act_scales(act_scales);
  load_w_scales(weight_scales);
  load_v1(&mat_a[32]);              // Loads v1 (Words 32 to 63)
  mac_zz();
  mac_mem_test_v1(weights);         // Multiplies v1 by weights[32..63]
  mac_as();
  mac_ws();
  chk = mac_out(0, 0, 2);
mac_bias(31,7,7,0x3fc0); 

//BRAM write check
  // mac_bias(0,0,0,0x3f80);   // tile0 row0 col0 (even row)
  // mac_bias(0,0,1,0x4000);   // tile0 row0 col1 (neighbor col - must not clobber col0)
  // mac_bias(0,7,7,0x4120);   // tile0 row7 col7 (ODD row - must not assert or spill to tile1)
  // mac_bias(1,0,0,0x4040);   // tile1 row0 col0 (must stay clean)
  // mac_bias(31,7,7,0x3fc0);  // tile31 row7 col7 (ODD row, last tile - must not spill past)

  // mac_bias(0,0,0,0x3f80);   // tile0 row0 col0 (even row), test double write to one cell

  // mac_bias(0,7,7,0x4120);   // tile0 row7 col7 (ODD row - must not assert or spill to tile1)

  // ==========================================
  // BRING-UP TEST 3: Register v31 Boundary Verification
  // ==========================================
/*
  load_v31(&mat_a[992]);            // Loads v31 (Words 992 to 1023)
  mac_zz();
  mac_mem_test_v31(weights);        // Multiplies v31 by weights[992..1023]
  mac_as();
  mac_ws();
  chk = mac_out(0, 0, 2);
*/
  // End hardware profiling and report output register status
  *COMP_END_MMIO = 1u;
  *DONE_MMIO = chk;

  while (1) {}
}
