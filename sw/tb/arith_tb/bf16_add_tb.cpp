
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "verilated.h"
#include "Vparameterized_adder.h"

// EXHAUST TEST FLAGS
#define IGNORE_NAN  1
#define IGNORE_ZERO 0
#define IGNORE_SUBNORMAL  0
#define IGNORE_INF 1
#define SUBNORMAL_FLUSH_TO_ZERO 1
#define RUN_EXHAUSTIVE 1

#define PRINT_MACRO(x) \
    std::cout << #x << " = " << (x) << '\n'


using bf16_t = uint16_t;

/* exp == 0xFF, mant != 0 */
bool bf16_is_nan(bf16_t x){
    uint16_t exp  = (x >> 7) & 0xFF;
    uint16_t mant = x & 0x7F;
    return (exp == 0xFF) && (mant != 0);
}

/* magnitude is zero, regardless of sign */
bool bf16_is_zero(bf16_t x){
    return (x & 0x7FFF) == 0;
}

/* exp == 0, mant != 0 */
bool bf16_is_subnormal(bf16_t x){
    uint16_t exp  = (x >> 7) & 0xFF;
    uint16_t mant = x & 0x7F;
    return (exp == 0) && (mant != 0);
}

bool bf16_is_inf(bf16_t x){
    uint16_t exp  = (x >> 7) & 0xFF;
    uint16_t mant = x & 0x7F;
    return exp == 0xFF;
}

float bf16_to_float(bf16_t x){

    /* SPECIAL CASE, subnormal value */

    #if SUBNORMAL_FLUSH_TO_ZERO
        if (bf16_is_subnormal(x)){
            return 0; 
        }
    #endif

    uint32_t raw_bytes = static_cast<uint32_t>(x) << 16;
    float result;
    std::memcpy(&result, &raw_bytes, sizeof(result));

    return result;
}

bf16_t bf16_add(bf16_t a, bf16_t b){

    float fa, fb, add_res;
    fa = bf16_to_float(a);
    fb = bf16_to_float(b);

    add_res = fa + fb;
    uint32_t raw_add_res;
    std::memcpy(&raw_add_res, &add_res, sizeof(raw_add_res));

    /* BF16 rounding */
    uint32_t lsb = (raw_add_res >> 16) & 0x1;
    uint32_t roundeed = raw_add_res + 0x7FFF + lsb;
    uint16_t res = roundeed >> 16;
    #if SUBNORMAL_FLUSH_TO_ZERO
        if (bf16_is_subnormal(static_cast<bf16_t>(res))){
            return 0;
        }
    #endif

    return static_cast<bf16_t>(res);
}



/* Sign is ignored when either side is a NaN or a zero. */
bool bf16_results_match(bf16_t dut_val, bf16_t exp_val){
    bool dut_nan = bf16_is_nan(dut_val);
    bool exp_nan = bf16_is_nan(exp_val);
    if (dut_nan || exp_nan){
        return dut_nan && exp_nan;
    }

    bool dut_zero = bf16_is_zero(dut_val);
    bool exp_zero = bf16_is_zero(exp_val);
    if (dut_zero || exp_zero){
        return dut_zero && exp_zero;
    }

    return dut_val == exp_val;
}

/*
 *  BOUNDARY / SPECIAL VALUE CONSTRUCTION
 */

constexpr bf16_t BF16_POS_ZERO = 0x0000;
constexpr bf16_t BF16_NEG_ZERO = 0x8000;
constexpr bf16_t BF16_POS_INF  = 0x7F80;
constexpr bf16_t BF16_NEG_INF  = 0xFF80;
constexpr bf16_t BF16_POS_NAN  = 0x7FC0;
constexpr bf16_t BF16_NEG_NAN  = 0xFFC0;

bf16_t bf16_max_normal(bool sign)  { return (static_cast<bf16_t>(sign) << 15) | (0xFEu << 7) | 0x7Fu; }
bf16_t bf16_min_normal(bool sign)  { return (static_cast<bf16_t>(sign) << 15) | (0x01u << 7); }
bf16_t bf16_max_subnorm(bool sign) { return (static_cast<bf16_t>(sign) << 15) | 0x7Fu; }
bf16_t bf16_min_subnorm(bool sign) { return (static_cast<bf16_t>(sign) << 15) | 0x01u; }

class Scoreboard {
public:
    uint64_t tests_failed = 0;
    uint64_t num_tests = 0;

    void start() {
        start_time = std::chrono::steady_clock::now();
    }

    void end() {
        end_time = std::chrono::steady_clock::now();
    }

    void CHECK(bool cond){
        num_tests++;
        if (!cond) { tests_failed++; }
    }

    void print_summary() const {
        using namespace std::chrono;

        double elapsed_ms =
            duration<double, std::milli>(end_time - start_time).count();

        std::printf(
            "\n"
            "==============================\n"
            "        TEST SUMMARY\n"
            "==============================\n"
            "Tests Passed   : %llu/%llu\n"
            "Tests Failed   : %llu\n"
            "Elapsed Time   : %.3f ms\n"
            "==============================\n",
            static_cast<unsigned long long>(num_tests - tests_failed),
            static_cast<unsigned long long>(num_tests),
            static_cast<unsigned long long>(tests_failed),
            elapsed_ms);

        std::puts(tests_failed == 0 ? "PASS" : "FAIL");
    }

private:
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
};

bool eval_dut(Vparameterized_adder& dut, Scoreboard &sb, bf16_t a, bf16_t b){

    dut.a = a;
    dut.b = b;
    dut.eval();

    /* Check sum */
    bf16_t dut_sum = dut.sum;
    bf16_t exp_sum = bf16_add(a, b);

    bool match = bf16_results_match(dut_sum, exp_sum);
    sb.CHECK(match);

    if (!match){
        std::printf("[MISMATCH] a=%g (%04x) b=%g (%04x), exp=%g (%04x), res =%g (%04x)\n",
                                bf16_to_float(a), a,
                                bf16_to_float(b), b,
                                bf16_to_float(exp_sum), exp_sum,
                                bf16_to_float(dut_sum), dut_sum);
    }

    return match;
}

/**
 * Prints out progress bar.
 */
void progress(double p) {
    constexpr int width = 50;
    std::cout << "\r[";
    int pos = width * p;
    for (int i = 0; i < width; i++)
        std::cout << (i < pos ? '=' : i == pos ? '>' : ' ');
    std::cout << "] " << int(p * 100) << "%";
    std::cout.flush();
}

auto print_help = [](const char* prog) {
    std::cout <<
        "Usage: " << prog << " [OPTIONS]\n\n"
        "Runs directed + constrained-random bf16_add tests, then exhaustively\n"
        "tests bf16_add over all (a, b) pairs of 16-bit patterns.\n\n"
        "Options:\n"
        "  --a-offset <n>   Starting A value, 0-65535 (default: 0)\n"
        "  --a-stride <n>   Stride for A values (default: 1)\n"
        "  --b-offset <n>   Starting B value, 0-65535 (default: 0)\n"
        "  --b-stride <n>   Stride for B values (default: 1)\n"
        "  -h, --help       Show this help message\n";
};

/////////////////////////////////////////////////////////////////////////////////////
//
//                      TESTS
//
/////////////////////////////////////////////////////////////////////////////////////

/**
 *  CONSTRAINED (DIRECTED) TEST VECTOR GENERATION
 */

/* One DUT input vector: {a, b}. */
using test_case_t   = std::pair<bf16_t, bf16_t>;
using test_vector_t = std::vector<test_case_t>;
/* A named group of test vectors so groups can be looped over generically
   and reported on individually. */
using named_group_t = std::pair<std::string, test_vector_t>;

test_vector_t make_zero_tests() {
    return {
        /*  {a,                        b} */
        {   BF16_POS_ZERO,             BF16_POS_ZERO},
        {   BF16_NEG_ZERO,             BF16_POS_ZERO},
        {   BF16_POS_ZERO,             BF16_NEG_ZERO},
        {   BF16_NEG_ZERO,             BF16_NEG_ZERO},
        {   BF16_POS_ZERO,             bf16_max_normal(false)},
        {   bf16_max_normal(false),    BF16_NEG_ZERO},
        {   bf16_max_normal(true),     bf16_max_normal(false)},  // cancels to zero
        {   bf16_min_subnorm(false),   bf16_min_subnorm(true)},  // cancels to zero
    };
}

test_vector_t make_nan_tests() {
    return {
        /*  {a,                         b} */
        {   BF16_POS_NAN,               BF16_POS_NAN},
        {   BF16_NEG_NAN,               BF16_NEG_NAN},
        {   BF16_POS_NAN,               bf16_max_normal(false)},
        {   bf16_max_normal(false),     BF16_NEG_NAN},
        {   BF16_POS_NAN,               BF16_POS_ZERO},
        {   BF16_NEG_NAN,               bf16_min_subnorm(false)},
        {   BF16_POS_NAN,               BF16_NEG_NAN},
        {   BF16_POS_INF,               BF16_NEG_INF},            // inf - inf = NaN
        {   BF16_NEG_INF,               BF16_POS_INF},
    };
}

test_vector_t make_subnormal_tests() {
    return {
        /*  {a,                          b} */
        {   bf16_min_subnorm(false),     bf16_min_subnorm(false)},
        {   bf16_max_subnorm(false),     bf16_max_subnorm(false)},
        {   bf16_min_subnorm(true),      bf16_max_subnorm(false)},
        {   bf16_max_subnorm(true),      bf16_min_subnorm(true)},
        {   bf16_min_subnorm(false),     bf16_max_normal(false)},
        {   bf16_max_subnorm(false),     bf16_min_normal(false)},
        {   bf16_min_normal(false),      bf16_min_subnorm(true)}, // effective subtraction, straddles the normal/subnormal boundary
        {   static_cast<bf16_t>(0x0004), static_cast<bf16_t>(0x0005)},
    };
}

test_vector_t make_edge_case_tests() {
    return {
        /*  {a,                          b} */
        {   bf16_max_normal(false),      bf16_max_normal(false)},  // overflow to +inf
        {   bf16_max_normal(true),       bf16_max_normal(false)},  // cancels to zero
        {   bf16_min_normal(false),      bf16_min_normal(false)},
        {   bf16_min_normal(false),      bf16_max_subnorm(false)}, // right at the normal/subnormal boundary
        {   bf16_max_subnorm(false),     bf16_min_normal(false)},
        {   bf16_min_normal(true),       bf16_max_subnorm(false)}, // effective subtraction at the boundary
        {   bf16_max_normal(false),      bf16_min_subnorm(false)},
        {   bf16_max_normal(true),       bf16_min_subnorm(true)},
        {   BF16_POS_INF,                BF16_POS_INF},            // inf + inf = inf
        {   BF16_NEG_INF,                BF16_NEG_INF},
    };
}

void constrained_test(Vparameterized_adder &dut, Scoreboard &sb){
    std::cout << "CONSTRAINED TESTING" << std::endl;

    std::vector<named_group_t> groups = {
        {"ZERO",       make_zero_tests()},
        {"NAN",        make_nan_tests()},
        {"SUBNORMAL",  make_subnormal_tests()},
        {"EDGE_CASES", make_edge_case_tests()},
    };

    for (auto &[name, cases] : groups) {
        std::cout << "  [" << name << "] running " << cases.size()
                   << " test vector(s)" << std::endl;
        for (auto &[a, b] : cases) {
            eval_dut(dut, sb, a, b);
        }
    }
}

/**
 *  CONSTRAINED-RANDOM TEST VECTOR GENERATION
 *
 *  Input space is partitioned into normal / subnormal / boundary values
 *  (mirroring the nvfp8_mul_tb strategy) so that each region -- and the
 *  normal/subnormal crossover in particular -- gets dedicated coverage
 *  instead of being diluted by uniform random sampling.
 */

/* Small PRNG wrapper for constrained-random generation. */
class RandGen {
public:
    explicit RandGen(uint64_t seed = std::random_device{}()) : rng_(seed) {}

    bool coin() { return bern_(rng_); }
    int range(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng_); }

private:
    std::mt19937_64 rng_;
    std::bernoulli_distribution bern_{0.5};
};

/* exp in [2,253]: strictly interior, away from MIN_NORMAL (exp=1) and the
   MAX_NORMAL/inf/NaN region (exp=254/255). Mantissa unrestricted. */
bf16_t rand_normal_bf16(RandGen &rg) {
    uint16_t sign = rg.coin();
    uint16_t exp  = static_cast<uint16_t>(rg.range(2, 253));
    uint16_t mant = static_cast<uint16_t>(rg.range(0, 0x7F));
    return static_cast<bf16_t>((sign << 15) | (exp << 7) | mant);
}

/* exp == 0, mantissa in [1,127] (mantissa==0 would be zero, not subnormal). */
bf16_t rand_subnormal_bf16(RandGen &rg) {
    uint16_t sign = rg.coin();
    uint16_t mant = static_cast<uint16_t>(rg.range(1, 0x7F));
    return static_cast<bf16_t>((sign << 15) | mant);
}

/* exp == 1, mantissa unrestricted: the smallest-magnitude normal values --
   one ULP above the largest subnormal. Kept separate from rand_normal_bf16's
   interior range [2,253] so this specific boundary gets dedicated coverage.
   exp==1 and subnormals both represent the effective exponent (1-bias); they
   differ only in the implicit leading 1 bit, making this the likeliest spot
   for a hidden-bit/alignment bug. */
bf16_t rand_min_exp_normal_bf16(RandGen &rg) {
    uint16_t sign = rg.coin();
    uint16_t mant = static_cast<uint16_t>(rg.range(0, 0x7F));
    return static_cast<bf16_t>((sign << 15) | (1u << 7) | mant);
}

/* Zero counts as a boundary value here, so it is only ever produced by this
   generator, never by rand_normal_bf16 or rand_subnormal_bf16. */
bf16_t rand_boundary_bf16(RandGen &rg) {
    bool sign = rg.coin();
    switch (rg.range(0, 4)) {
        case 0:  return bf16_max_normal(sign);
        case 1:  return bf16_min_normal(sign);
        case 2:  return bf16_max_subnorm(sign);
        case 3:  return bf16_min_subnorm(sign);
        default: return sign ? BF16_NEG_ZERO : BF16_POS_ZERO;
    }
}

/* Exactly 1 of {A, B} is subnormal, the other is a random normal value.
   This directly targets the normal/subnormal crossover in the adder's
   exponent-alignment logic. */
void random_subnormal_1_test(Vparameterized_adder &dut, Scoreboard &sb, RandGen &rg, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        bool a_is_sub = rg.coin();
        bf16_t a = a_is_sub ? rand_subnormal_bf16(rg) : rand_normal_bf16(rg);
        bf16_t b = a_is_sub ? rand_normal_bf16(rg)    : rand_subnormal_bf16(rg);
        eval_dut(dut, sb, a, b);
    }
}

/* Both A and B are (independently randomized) subnormal. */
void random_subnormal_2_test(Vparameterized_adder &dut, Scoreboard &sb, RandGen &rg, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        eval_dut(dut, sb, rand_subnormal_bf16(rg), rand_subnormal_bf16(rg));
    }
}

/* Exactly 1 of {A, B} has exp==1 (min-exponent normal), the other is
   subnormal. This is the exact normal/subnormal crossover -- see
   rand_min_exp_normal_bf16. */
void random_min_exp_normal_vs_subnormal_test(Vparameterized_adder &dut, Scoreboard &sb, RandGen &rg, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        bool a_is_min_exp = rg.coin();
        bf16_t a = a_is_min_exp ? rand_min_exp_normal_bf16(rg) : rand_subnormal_bf16(rg);
        bf16_t b = a_is_min_exp ? rand_subnormal_bf16(rg)      : rand_min_exp_normal_bf16(rg);
        eval_dut(dut, sb, a, b);
    }
}

/* Exactly 1 of {A, B} is a boundary value, the other is a random normal value. */
void random_boundary_1_test(Vparameterized_adder &dut, Scoreboard &sb, RandGen &rg, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        bool a_is_boundary = rg.coin();
        bf16_t a = a_is_boundary ? rand_boundary_bf16(rg) : rand_normal_bf16(rg);
        bf16_t b = a_is_boundary ? rand_normal_bf16(rg)   : rand_boundary_bf16(rg);
        eval_dut(dut, sb, a, b);
    }
}

/* Both A and B are (independently randomized) boundary values. */
void random_boundary_2_test(Vparameterized_adder &dut, Scoreboard &sb, RandGen &rg, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        eval_dut(dut, sb, rand_boundary_bf16(rg), rand_boundary_bf16(rg));
    }
}

/* Both A and B are random normal values, neither at a boundary. */
void random_normal_test(Vparameterized_adder &dut, Scoreboard &sb, RandGen &rg, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        eval_dut(dut, sb, rand_normal_bf16(rg), rand_normal_bf16(rg));
    }
}

void constrained_random_test(Vparameterized_adder &dut, Scoreboard &sb, uint32_t iterations = 100000) {
    std::cout << "CONSTRAINED RANDOM TESTING" << std::endl;
    RandGen rg;

    std::cout << "  [SUBNORMAL_1] " << iterations << " iteration(s)" << std::endl;
    random_subnormal_1_test(dut, sb, rg, iterations);

    std::cout << "  [SUBNORMAL_2] " << iterations << " iteration(s)" << std::endl;
    random_subnormal_2_test(dut, sb, rg, iterations);

    std::cout << "  [MIN_EXP_NORMAL_VS_SUBNORMAL] " << iterations << " iteration(s)" << std::endl;
    random_min_exp_normal_vs_subnormal_test(dut, sb, rg, iterations);

    std::cout << "  [BOUNDARY_1] " << iterations << " iteration(s)" << std::endl;
    random_boundary_1_test(dut, sb, rg, iterations);

    std::cout << "  [BOUNDARY_2] " << iterations << " iteration(s)" << std::endl;
    random_boundary_2_test(dut, sb, rg, iterations);

    std::cout << "  [NORMAL] " << iterations << " iteration(s)" << std::endl;
    random_normal_test(dut, sb, rg, iterations);
}

//////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv){
    Verilated::commandArgs(argc, argv);

    uint32_t a_offset = 0, a_stride = 1;
    uint32_t b_offset = 0, b_stride = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        }

        if (i + 1 >= argc) {
            std::cerr << "Missing value for option '" << arg << "'\n";
            print_help(argv[0]);
            return 1;
        }

        if (arg == "--a-offset")
            a_offset = std::stoul(argv[++i]);
        else if (arg == "--a-stride")
            a_stride = std::stoul(argv[++i]);
        else if (arg == "--b-offset")
            b_offset = std::stoul(argv[++i]);
        else if (arg == "--b-stride")
            b_stride = std::stoul(argv[++i]);
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_help(argv[0]);
            return 1;
        }
    }

    Vparameterized_adder dut;
    Scoreboard sb;

    uint64_t num_a = (65536 - a_offset + a_stride - 1) / a_stride;
    uint64_t num_b = (65536 - b_offset + b_stride - 1) / b_stride;
    uint64_t total_tests = num_a * num_b;
    uint64_t completed = 0;

    sb.start();

    // constrained_test(dut, sb);
    // constrained_random_test(dut, sb);
    eval_dut(dut, sb, 0x25a3, 0xa5da);

    std::cout << "EXHAUSTIVE TESTING" << std::endl;
    std::cout << "EXHAUSTING TESTING FLAGS" << std::endl;
    #if(IGNORE_NAN)
        PRINT_MACRO(IGNORE_NAN);
    #endif

    #if(IGNORE_ZERO)
        PRINT_MACRO(IGNORE_ZERO);
    #endif

    #if(IGNORE_SUBNORMAL)
        PRINT_MACRO(IGNORE_SUBNORMAL);
    #endif
    
    #if (IGNORE_INF)
        PRINT_MACRO(IGNORE_INF);
    #endif

    #if (RUN_EXHAUSTIVE)
        PRINT_MACRO(RUN_EXHAUSTIVE);
        for (uint32_t a = a_offset; a < 65536; a += a_stride) {
            for (uint32_t b = b_offset; b < 65536; b += b_stride) {

                /* Skip nan and zero values; already covered directly above. */
                #if(IGNORE_NAN)
                if (bf16_is_nan(a) || bf16_is_nan(b)){
                    continue;
                }
                #endif

                #if(IGNORE_ZERO)
                if (bf16_is_zero(a) || bf16_is_zero(b)){
                    continue;
                }
                #endif


                #if(IGNORE_INF)
                if (bf16_is_inf(a) || bf16_is_inf(b)){
                    continue;
                }
                #endif


                #if(IGNORE_SUBNORMAL)
                if (bf16_is_subnormal(a) || bf16_is_subnormal(b)){
                    continue;
                }
                #endif

                eval_dut(dut, sb, static_cast<bf16_t>(a), static_cast<bf16_t>(b));

                ++completed;

                // Update every million tests to reduce overhead
                if ((completed % 1000000) == 0) {
                    progress(
                        static_cast<float>(completed) /
                        static_cast<float>(total_tests));
                }
            }
        }

        progress(1.0);
    #endif

    sb.end();
    sb.print_summary();

    return sb.tests_failed != 0;
}
