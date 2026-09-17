#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "verilated.h"
#include "Vmac_cell.h"

static vluint64_t main_time = 0;
double sc_time_stamp() { return static_cast<double>(main_time); }

/* Toggles clk for one full cycle; DUT registers sample on the rising edge. */
static void tick(Vmac_cell &dut) {
    dut.clk = 0;
    dut.eval();
    main_time++;
    dut.clk = 1;
    dut.eval();
    main_time++;
}

class fp4_t {
public:
    const uint8_t raw;

    /* Constructor from byte */
    fp4_t(uint8_t x) : raw(x & 0xF){}

    /* Constants */
    static const uint8_t EXP_BIAS = 1;
    static const uint8_t EXP_WIDTH = 2;
    static const uint8_t MANT_WIDTH = 1;

    /* SPECIAL VALUES */
    static fp4_t ZERO(uint8_t sign) { return fp4_t(sign ? 0x8 : 0x0); }
    static fp4_t MIN_VAL() { return fp4_t(0xf); }
    static fp4_t MAX_VAL() { return fp4_t(0x7); }

    uint8_t mant() const{ return raw & 0x1; }
    uint8_t exp() const{ return (raw >> 0x1) & 0x3;}
    uint8_t sign() const{ return (raw >> 0x3) & 0x1;}
    uint8_t get_raw() const{ return raw & 0xf; }
    bool is_subnormal() const{ return exp() == 0; }
    bool is_zero() const{ return get_raw() == 0; }

    /**
     *  Convert FP4 values to double precision values.
     */
    double get_real() const{
        if (is_zero()){
            return 0;
        }

        double frac = ((double) mant()) / std::exp2(MANT_WIDTH);
        double mag;

        if (is_subnormal()){
            mag = std::exp2(-EXP_BIAS+1) * frac;
        } else {
            mag = (1.0 + frac) * std::exp2(exp()-EXP_BIAS);
        }

        return sign() ? -mag : mag;
    }
};

class mac_cell_transaction_t {
public:
    fp4_t a;
    fp4_t b;
    bool mac_en;
    bool clear;

    mac_cell_transaction_t(fp4_t a, fp4_t b, bool mac_en, bool clear)
        : a(a), b(b), mac_en(mac_en), clear(clear)
    {}
};

/**
 *  Predictor Model
 */
class MacCellRef {
public:
    /* State: Accumulator */
    int16_t acc;

    MacCellRef(){
        acc = 0;
    }

    void processTransaction(const mac_cell_transaction_t &t){
        /* clear_i/mv_clear_i win over mac_en_i */
        if (t.clear){
            acc = 0;
            return;
        }

        /* we_i (mac_en_i) deasserted -> accumulator holds its value */
        if (!t.mac_en){
            return;
        }

        int32_t acc_sum =
            static_cast<int32_t>(acc) + fp4_to_int(t.a) * fp4_to_int(t.b);

        /* Saturating add */
        if (acc_sum < std::numeric_limits<int16_t>::min()){
            acc = std::numeric_limits<int16_t>::min();
        } else if (acc_sum > std::numeric_limits<int16_t>::max()){
            acc = std::numeric_limits<int16_t>::max();
        } else {
            acc = static_cast<int16_t>(acc_sum);
        }
    }

private:

    /*
        Convert fp4 to int, in units of the FP4 quanta (0.5).
        u8_mag_o in the RTL is v(mag(a)) * v(mag(b)) * 4, so
        fp4_to_int(a) * fp4_to_int(b) == (v(a)*2) * (v(b)*2) reproduces
        that same scaled (and signed) product.
    */
    int32_t fp4_to_int(const fp4_t &x){
        return static_cast<int32_t>(x.get_real() / 0.5);
    }
};

/*
    Test Case Interface

    Concrete test cases derive from this, implement run() using CHECK()
    to record individual assertions, and get timing/pass-fail tracking
    and reporting for free.
*/
class ITestCase {
public:
    struct Stats {
        uint64_t num_tests = 0;
        uint64_t num_fail = 0;
        double elapsed_ms = 0.0;
    };

    explicit ITestCase(std::string name) : name_(std::move(name)) {}
    virtual ~ITestCase() = default;

    /* Test body. Implemented by subclasses; report results via CHECK(). */
    virtual void run() = 0;

    /* Invoked by TestSuite: times run() and (re)computes pass/fail + stats. */
    void execute() {
        stats_ = Stats{};

        const auto t0 = std::chrono::steady_clock::now();
        run();
        const auto t1 = std::chrono::steady_clock::now();

        stats_.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    const std::string &name() const { return name_; }
    bool passed() const { return stats_.num_fail == 0; }
    const Stats &stats() const { return stats_; }

    /* Prints pass/fail status plus the standard stats for this test case. */
    void print_status() const {
        std::printf("[%s] %-28s checks=%llu/%llu  elapsed=%.3fms\n",
            passed() ? "PASS" : "FAIL",
            name_.c_str(),
            static_cast<unsigned long long>(stats_.num_tests- stats_.num_fail),
            static_cast<unsigned long long>(stats_.num_tests),
            stats_.elapsed_ms);
    }

protected:
    /* Records one check's outcome; any failing check fails the whole case. */
    void CHECK(bool cond, const std::string &msg = "") {
        stats_.num_tests++;
        if (cond) {
            return;
        }

        stats_.num_fail++;
        if (!msg.empty()) {
            std::printf("  [%s] CHECK FAILED: %s\n", name_.c_str(), msg.c_str());
        }
    }

private:
    std::string name_;
    Stats stats_;
};

/*
    Test Cases:
    1) FP4 computation, check if FP4 computation matches
    2) clear behaviour
    3) accumulator behaviour, done over 5 rounds
    4) saturating behaviour
*/

/* Shared DUT-driving helpers for mac_cell test cases. */
class MacCellTestCase : public ITestCase {
public:
    MacCellTestCase(std::string name, Vmac_cell &dut)
        : ITestCase(std::move(name)), dut_(dut) {}

protected:
    Vmac_cell &dut_;
    MacCellRef ref_;

    /* Synchronously clears both the DUT and reference model accumulators. */
    void reset() {
        dut_.mac_en_i = 0;
        dut_.mv_clear_i = 0;
        dut_.clear_i = 1;
        dut_.act_i = 0;
        dut_.wt_i = 0;
        tick(dut_);
        dut_.clear_i = 0;

        ref_ = MacCellRef();
    }

    /* Drives one MAC cycle on the DUT and the reference model in lockstep,
       returning the DUT's resulting accumulator value. */
    int16_t step(fp4_t a, fp4_t b, bool mac_en, bool clear, bool mv_clear = false) {
        dut_.act_i = a.get_raw();
        dut_.wt_i = b.get_raw();
        dut_.mac_en_i = mac_en;
        dut_.clear_i = clear;
        dut_.mv_clear_i = mv_clear;
        tick(dut_);

        ref_.processTransaction(mac_cell_transaction_t(a, b, mac_en, clear || mv_clear));

        return static_cast<int16_t>(dut_.accum_o);
    }
};

/* Exhaustively checks a single MAC op (from a cleared accumulator) for
   every combination of FP4 inputs against the reference model. */
class Fp4ComputationTest : public MacCellTestCase {
public:
    explicit Fp4ComputationTest(Vmac_cell &dut)
        : MacCellTestCase("fp4_computation", dut) {}

    void run() override {
        for (uint8_t araw = 0; araw < 16; ++araw) {
            for (uint8_t braw = 0; braw < 16; ++braw) {
                reset();

                int16_t got = step(fp4_t(araw), fp4_t(braw), /*mac_en=*/true, /*clear=*/false);

                CHECK(got == ref_.acc,
                    "a=0x" + std::to_string(araw) + " b=0x" + std::to_string(braw) +
                    " got=" + std::to_string(got) + " exp=" + std::to_string(ref_.acc));
            }
        }
    }
};

/* Checks that clear_i and mv_clear_i both force the accumulator to zero,
   even in the same cycle as an incoming MAC operation. */
class ClearBehaviorTest : public MacCellTestCase {
public:
    explicit ClearBehaviorTest(Vmac_cell &dut)
        : MacCellTestCase("clear_behavior", dut) {}

    void run() override {
        reset();

        int16_t got = step(fp4_t::MAX_VAL(), fp4_t::MAX_VAL(), /*mac_en=*/true, /*clear=*/false);
        CHECK(got != 0, "accumulator failed to build up a nonzero value before clear");

        got = step(fp4_t::MAX_VAL(), fp4_t::MAX_VAL(), /*mac_en=*/true, /*clear=*/true);
        CHECK(got == 0 && ref_.acc == 0, "clear_i did not reset accumulator to zero");

        step(fp4_t::MAX_VAL(), fp4_t::MAX_VAL(), /*mac_en=*/true, /*clear=*/false);
        got = step(fp4_t::MAX_VAL(), fp4_t::MAX_VAL(), /*mac_en=*/true, /*clear=*/false, /*mv_clear=*/true);
        CHECK(got == 0 && ref_.acc == 0, "mv_clear_i did not reset accumulator to zero");
    }
};

/* Accumulates over 5 rounds of distinct FP4 input pairs and checks the
   running accumulator value after every round, then checks that
   deasserting mac_en_i holds the accumulator steady. */
class AccumulatorBehaviorTest : public MacCellTestCase {
public:
    explicit AccumulatorBehaviorTest(Vmac_cell &dut)
        : MacCellTestCase("accumulator_behavior", dut) {}

    void run() override {
        reset();

        static constexpr int kRounds = 5;
        static constexpr uint8_t kAVals[kRounds] = {0x1, 0x3, 0x9, 0xC, 0x5};
        static constexpr uint8_t kBVals[kRounds] = {0x2, 0xB, 0x4, 0xD, 0x7};

        for (int round = 0; round < kRounds; ++round) {
            int16_t got = step(fp4_t(kAVals[round]), fp4_t(kBVals[round]),
                                /*mac_en=*/true, /*clear=*/false);

            CHECK(got == ref_.acc,
                "round=" + std::to_string(round) +
                " got=" + std::to_string(got) + " exp=" + std::to_string(ref_.acc));
        }

        int16_t held = ref_.acc;
        int16_t got = step(fp4_t::MAX_VAL(), fp4_t::MAX_VAL(), /*mac_en=*/false, /*clear=*/false);
        CHECK(got == held && ref_.acc == held,
            "accumulator changed while mac_en_i was deasserted");
    }
};

/* Drives repeated max-magnitude MAC ops of a fixed sign to check that the
   accumulator saturates (rather than wraps) at INT16_MAX/INT16_MIN. */
class SaturationTest : public MacCellTestCase {
public:
    explicit SaturationTest(Vmac_cell &dut)
        : MacCellTestCase("saturation_behavior", dut) {}

    void run() override {
        test_direction(/*negative=*/false);
        test_direction(/*negative=*/true);
    }

private:
    void test_direction(bool negative) {
        reset();

        fp4_t a = fp4_t::MAX_VAL();
        fp4_t b = negative ? fp4_t::MIN_VAL() : fp4_t::MAX_VAL();

        /* Each step contributes +-144; a few hundred rounds comfortably
           saturates a 16-bit accumulator either direction. */
        int16_t got = 0;
        for (int i = 0; i < 500; ++i) {
            got = step(a, b, /*mac_en=*/true, /*clear=*/false);
        }

        int16_t expect = negative ? std::numeric_limits<int16_t>::min()
                                   : std::numeric_limits<int16_t>::max();

        CHECK(got == expect,
            std::string("failed to saturate to ") + (negative ? "INT16_MIN" : "INT16_MAX") +
            " got=" + std::to_string(got));
        CHECK(ref_.acc == expect, "reference model saturation mismatch");
    }
};

/*
    TestSuite

    Owns a name -> TestCase registry and runs a caller-specified subset of
    test cases (or all of them, in registration order, if none are named).
*/
class TestSuite {
public:
    void add(std::unique_ptr<ITestCase> tc) {
        const std::string name = tc->name();
        order_.push_back(name);
        registry_[name] = std::move(tc);
    }

    /* Runs `names`, or every registered test case (in registration order)
       if `names` is empty. Returns true iff every test case ran and passed. */
    bool run(const std::vector<std::string> &names = {}) {
        const std::vector<std::string> &to_run = names.empty() ? order_ : names;

        bool all_passed = true;
        uint64_t total_checks = 0;
        uint64_t total_passed = 0;

        std::printf("Running %zu test case(s)...\n", to_run.size());
        for (const auto &name : to_run) {
            auto it = registry_.find(name);
            if (it == registry_.end()) {
                std::printf("[SKIP] unknown test case '%s'\n", name.c_str());
                all_passed = false;
                continue;
            }

            ITestCase &tc = *it->second;
            tc.execute();
            tc.print_status();

            all_passed = all_passed && tc.passed();
            total_checks += tc.stats().num_tests;
            total_passed += tc.stats().num_tests - tc.stats().num_fail;
        }

        std::printf(
            "\n==============================\n"
            "        SUITE SUMMARY\n"
            "==============================\n"
            "Checks Passed : %llu/%llu\n"
            "Result        : %s\n"
            "==============================\n",
            static_cast<unsigned long long>(total_passed),
            static_cast<unsigned long long>(total_checks),
            all_passed ? "PASS" : "FAIL");

        return all_passed;
    }

private:
    std::unordered_map<std::string, std::unique_ptr<ITestCase>> registry_;
    std::vector<std::string> order_;
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    Vmac_cell dut;

    TestSuite suite;
    suite.add(std::make_unique<Fp4ComputationTest>(dut));
    suite.add(std::make_unique<ClearBehaviorTest>(dut));
    suite.add(std::make_unique<AccumulatorBehaviorTest>(dut));
    suite.add(std::make_unique<SaturationTest>(dut));

    /* Extra CLI args (if any) name which test cases to run; none => run all. */
    std::vector<std::string> selected(argv + 1, argv + argc);
    bool ok = suite.run(selected);

    dut.final();
    return ok ? 0 : 1;
}
