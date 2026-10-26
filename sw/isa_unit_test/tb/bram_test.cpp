/**
 *  BRAM gen2 unit test driver.
 *
 *  For each of N_RUNS:
 *      1. Reload the firmware image (.text/.rodata -> IMEM, .data -> DMEM)
 *      2. Fill the bias input with random bf16, scribble over the readout buffers
 *      3. Reset the DUT and run until the firmware raises DONE
 *      4. Report a non-zero STATUS (the firmware's exit code)
 *
 *  The firmware does its own checking; the random readout contents just make
 *  sure stale data from a previous run can't pass for a fresh readout.
 */

#include "mmio.h"
#include "bram_test_mmio.h"
#include "verilated.h"
#include "Vcve2_top.h"
#include "tb_memory.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>

static constexpr int N_RUNS = 10;

struct cliVars {
    uint64_t max_cycles = 2000000;
    uint32_t seed = 1;
    bool dumpLog = true;
    std::string imemHexPath;
    std::string dmemHexPath;
    std::string dumpPath = "uart.log";
};

struct RunResult {
    bool     done;
    uint64_t cycles;
    int32_t  status;
};

static cliVars parseArgs(int argc, char **argv){
    cliVars args;

    for (int i = 1; i < argc; ++i){
        std::string a = argv[i];
        bool hasValue = i + 1 < argc;

        if      (a == "--imem" && hasValue)       args.imemHexPath = argv[++i];
        else if (a == "--dmem" && hasValue)       args.dmemHexPath = argv[++i];
        else if (a == "--max-cycles" && hasValue) args.max_cycles  = std::stoull(argv[++i]);
        else if (a == "--seed" && hasValue)       args.seed        = std::stoul(argv[++i]);
        else if (a == "--uart-log" && hasValue)   args.dumpPath    = argv[++i];
        else if (a == "--no-uart-log")            args.dumpLog     = false;
    }

    if (args.imemHexPath.empty() || args.dmemHexPath.empty()){
        std::cerr << "Usage: " << argv[0]
                  << " --imem <imem.hex> --dmem <dmem.hex>"
                     " [--max-cycles N] [--seed N] [--uart-log PATH | --no-uart-log]\n";
        std::exit(2);
    }

    return args;
}

static Memory initMemSystem(const cliVars& args){

    Memory memSystem;

    /* Program Memory map */
    memSystem.addMemRegion(std::make_unique<MemoryRegion>(
        PROGRAM_MEM_BASE, PROGRAM_MEM_WORDS
    ));

    /* Memory data region */
    memSystem.addMemRegion(std::make_unique<MemoryRegion>(
        DATA_MEM_BASE, DATA_MEM_WORDS)
    );

    /* Staging buffer setup for bram operations */
    memSystem.addMemRegion(std::make_unique<MemoryRegion>(
        STAGE_MEM_BASE, STAGE_MEM_WORDS)
    );

    /* MMIO setup */
    memSystem.addMemRegion(std::make_unique<DoneMMIO>());
    memSystem.addMemRegion(std::make_unique<ScalarMMIO>(STATUS_MMIO_ADDR));
    memSystem.addMemRegion(
        std::make_unique<UartMMIO>(args.dumpPath, args.dumpLog));

    return memSystem;
}

/*
    Fresh firmware image: .bss is NOLOAD and start.S doesn't clear it,
    so zero DMEM first to give C statics their zero init.
*/
static void loadProgram(Memory& mem, const cliVars& args){
    clearRegion(mem, DATA_MEM_BASE, DATA_MEM_WORDS);
    loadVerilogHex(mem, args.imemHexPath, PROGRAM_MEM_BASE, PROGRAM_MEM_WORDS);
    loadVerilogHex(mem, args.dmemHexPath, DATA_MEM_BASE, DATA_MEM_WORDS);
}

/*
    Random bf16 around the readout quantizer's range (|x| ~ 2^-9 .. 2^1),
    so every FP4 code, the flush to zero and the clamp all get hit.
*/
static uint16_t randomBf16(std::mt19937& rng){
    std::uniform_int_distribution<uint32_t> signDist(0, 1);
    std::uniform_int_distribution<uint32_t> expDist(118, 128);
    std::uniform_int_distribution<uint32_t> manDist(0, 0x7F);

    return (uint16_t)((signDist(rng) << 15) | (expDist(rng) << 7) | manDist(rng));
}

static void randomizeStaging(Memory& mem, std::mt19937& rng){

    /* Bias input, two bf16 per word */
    for (uint32_t i = 0; i < BIAS_IN_SIZE; ++i){
        uint32_t word = randomBf16(rng) | ((uint32_t)randomBf16(rng) << 16);
        mem.write(BIAS_IN_BASE + i * sizeof(uint32_t), word);
    }

    /* Readout buffers, garbage the firmware must overwrite */
    for (uint32_t i = 0; i < READOUT_BF16_SIZE; ++i){
        mem.write(READOUT_BF16_BASE + i * sizeof(uint32_t), rng());
    }

    for (uint32_t i = 0; i < READOUT_FP4_SIZE; ++i){
        mem.write(READOUT_FP4_BASE + i * sizeof(uint32_t), rng());
    }
}

/* One clock, in the order the drivers expect (see tb_memory.hpp) */
static void tick(Vcve2_top& dut, IMemDriver& ifMemDriver, DMemDriver& dMemDriver){
    dut.clk_i = 0;
    ifMemDriver.drive();
    dMemDriver.drive();
    dut.eval();

    ifMemDriver.sample();
    dMemDriver.sample();

    dut.clk_i = 1;
    dut.eval();
}

/* Hold reset with the bus ports idle, then release and enable fetch */
static void resetDut(Vcve2_top& dut, IMemDriver& ifMemDriver, DMemDriver& dMemDriver){
    dut.rst_ni         = 0;
    dut.fetch_enable_i = 0;
    dut.hart_id_i      = 0;
    dut.boot_addr_i    = PROGRAM_MEM_BASE;

    ifMemDriver.reset();
    dMemDriver.reset();

    for (int i = 0; i < 10; ++i){
        dut.clk_i = 0; dut.eval();
        dut.clk_i = 1; dut.eval();
    }

    dut.rst_ni = 1;
    for (int i = 0; i < 5; ++i){
        dut.clk_i = 0; dut.eval();
        dut.clk_i = 1; dut.eval();
    }

    dut.fetch_enable_i = 1;
}

static RunResult runOnce(Vcve2_top& dut, Memory& mem,
                         IMemDriver& ifMemDriver, DMemDriver& dMemDriver,
                         uint64_t maxCycles){

    DoneMMIO::resetDoneMMIO(mem);
    mem.write(STATUS_MMIO_ADDR, 0);

    resetDut(dut, ifMemDriver, dMemDriver);

    for (uint64_t cyc = 0; cyc < maxCycles; ++cyc){
        tick(dut, ifMemDriver, dMemDriver);

        /* Check for done signal from the MMIO */
        if (DoneMMIO::checkDone(mem)){
            return {true, cyc, (int32_t)mem.read(STATUS_MMIO_ADDR)};
        }
    }

    return {false, maxCycles, 0};
}

static int run_sim(const cliVars& args){

    /* Sole owner of the model; Verilator models are big, so keep it on the heap */
    auto dut = std::make_unique<Vcve2_top>();

    /* Simulated memories */
    Memory mem = initMemSystem(args);

    /* Drivers borrow the dut and mem, both outlive them */
    IMemDriver ifMemDriver(*dut, mem);
    DMemDriver dMemDriver(*dut, mem);

    std::mt19937 rng(args.seed);
    int failures = 0;

    for (int run = 0; run < N_RUNS; ++run){
        loadProgram(mem, args);
        randomizeStaging(mem, rng);

        RunResult result = runOnce(*dut, mem, ifMemDriver, dMemDriver, args.max_cycles);

        std::cout << "[TB] run " << run << ": ";
        if (!result.done){
            std::cout << "TIMEOUT after " << result.cycles << " cycles\n";
            ++failures;
        } else if (result.status != 0){
            std::cout << "FAIL, status " << result.status
                      << " at cycle " << result.cycles << "\n";
            ++failures;
        } else {
            std::cout << "PASS in " << result.cycles << " cycles\n";
        }
    }

    dut->final();

    std::cout << "[TB] " << (N_RUNS - failures) << "/" << N_RUNS
              << " runs passed (seed " << args.seed << ")\n";

    return failures == 0 ? 0 : 1;
}

int main(int argc, char **argv){
    Verilated::commandArgs(argc, argv);
    cliVars args = parseArgs(argc, argv);

    try {
        return run_sim(args);
    } catch (const std::exception& e){
        std::cerr << e.what() << "\n";
        return 1;
    }
}
