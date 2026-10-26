#pragma once

#include "verilated.h"
#include "Vcve2_top.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <ostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include <stdexcept>
/* BRAM Side MMIO */

#include "uart_logging.h"

#include "bram_test_mmio.h"


/*
    Memory predictor model,
*/

/*
    Base type for anything mapped into the address space.
    Owns the address range only; subclasses supply read/write behaviour.
*/
class AbstractMemoryRegion {

public:
    uint64_t baseAddress;
    size_t n;

    /*
        Region spans n * uint32_t (4 bytes) starting at baseAddress
     */
    AbstractMemoryRegion(uint64_t baseAddress, size_t nWords)
        : baseAddress(baseAddress), n(nWords) {}

    virtual ~AbstractMemoryRegion() = default;

    virtual uint32_t read(uint64_t address) = 0;
    virtual void write(uint64_t address, uint32_t data) = 0;

    bool contains(uint64_t address) const {
        return address >= baseAddress && address < baseAddress + n * sizeof(uint32_t);
    }

protected:
    /* Byte address -> element index within the region */
    size_t translateAddress(uint64_t address) const {
        assert(contains(address));
        return (address - baseAddress) / sizeof(uint32_t);
    }
};

/* Plain backing storage, normal r/w */
class MemoryRegion : public AbstractMemoryRegion {

public:
    std::vector<uint32_t> dataArray;

    MemoryRegion(uint64_t baseAddress, size_t nWords)
        : AbstractMemoryRegion(baseAddress, nWords), dataArray(nWords) {}

    uint32_t read(uint64_t address) override {
        return dataArray[translateAddress(address)];
    }

    void write(uint64_t address, uint32_t data) override {
        dataArray[translateAddress(address)] = data;
    }
};

/*
    Single address MMIO
    Override onRead / onWrite for side effects (status regs, triggers, ...).
    Default behaviour is a plain register.
*/
class ScalarMMIO : public AbstractMemoryRegion {

public:
    uint32_t value{};

    ScalarMMIO(uint64_t baseAddress)
        : AbstractMemoryRegion(baseAddress, 1) {}

    uint32_t read(uint64_t address) override {
        translateAddress(address);
        return onRead();
    }

    void write(uint64_t address, uint32_t data) override {
        translateAddress(address);
        onWrite(data);
    }

protected:
    virtual uint32_t onRead() {
        return value;
    }

    virtual void onWrite(uint32_t data) {
        value = data;
    }
};

class UartMMIO : public ScalarMMIO {

private:
    std::ofstream uartLog;
    bool shouldDumpFile;
public:

    UartMMIO(std::string dumpFilePath, bool dumpFile = true) 
        : ScalarMMIO(UART_MMIO_ADDR)
    {   
        shouldDumpFile = dumpFile;
        if (dumpFile){
            uartLog.open(dumpFilePath);
        }
    }

    void onWrite(uint32_t data) override {
        char uartChar = data & 0xFF;
        std::cout << uartChar << std::flush;
        
        if (shouldDumpFile){
            uartLog << uartChar << std::flush;
        }
    }

    uint32_t onRead() override {
        return 0;
    }
};



/*
    Memory owns its regions (unique_ptr). Callers get back a plain reference,
    which is non-owning and valid for as long as the Memory is.
*/
class Memory {
private:
    std::vector<std::unique_ptr<AbstractMemoryRegion>> regions;  

    /* 
        Decodes address and selects the region,
    */
    AbstractMemoryRegion* regionSelect(uint64_t address){

        for (auto& region : this->regions){
            if (region->contains(address)){
                return region.get();
            }
        }   
        
        return nullptr;
    }

public:

    /*
        Adds a memory region to the memory system.
        Moves ownership of the object to the memory class.
    */  
    void addMemRegion(std::unique_ptr<AbstractMemoryRegion> region){
        /* Nullptr check */
        if (!region){
            return;
        }
        regions.push_back(std::move(region));
    }

    /* True if some region decodes this address */
    bool mapped(uint64_t address){
        return regionSelect(address) != nullptr;
    }

    /* Service Write Transaction */
    void write(uint64_t address, uint32_t data){
        auto writeRegion = regionSelect(address);
        
        if (writeRegion == nullptr){
            throw std::out_of_range("[SYSTEM MEM] Invalid write address accessed");
        }

        writeRegion->write(address, data);
    }

    /* Service Read Transaction */
    uint32_t read(uint64_t address){
        auto readRegion = regionSelect(address);
        
        if (readRegion == nullptr){
            throw std::out_of_range("[SYSTEM MEM] Invalid read address accessed");
        }

        return readRegion->read(address);
    }
};


/* Plain register at DONE_MMIO_ADDR (mmio.h); firmware writes 1 when finished */
class DoneMMIO : public ScalarMMIO {

public:
    static void resetDoneMMIO(
        Memory& mem
    ){
        mem.write(DONE_MMIO_ADDR, 0);
    }

    static bool checkDone(
        Memory& mem
    ){
        return mem.read(DONE_MMIO_ADDR) == 1;
    }

    DoneMMIO()
        : ScalarMMIO(DONE_MMIO_ADDR) {}

    bool done() const {
        return value != 0;
    }
};



/*
    Loads an `objcopy -O verilog` image (@addr records + byte tokens) into mem.
    Every byte must land in [base, base + nWords * 4), so an image built for
    the wrong region fails loudly instead of silently landing elsewhere.
    Returns the number of bytes written.
*/
inline size_t loadVerilogHex(Memory& mem, const std::string& path,
                             uint64_t base, size_t nWords){
    std::ifstream hexFile(path);
    if (!hexFile){
        throw std::runtime_error("[LOADER] Cannot open " + path);
    }

    const uint64_t end = base + nWords * sizeof(uint32_t);
    uint64_t addr = 0;
    size_t   nBytes = 0;
    std::string tok;

    while (hexFile >> tok){
        if (tok[0] == '@'){
            addr = std::stoull(tok.substr(1), nullptr, 16);
            continue;
        }

        if (addr < base || addr >= end){
            throw std::out_of_range("[LOADER] " + path + ": byte outside its region");
        }

        /* Byte-wise read-modify-write into the word-addressed region */
        uint32_t byte  = std::stoul(tok, nullptr, 16) & 0xFFu;
        uint64_t word  = addr & ~uint64_t{3};
        unsigned shift = (addr & 3) * 8;
        uint32_t old   = mem.read(word);
        mem.write(word, (old & ~(0xFFu << shift)) | (byte << shift));

        ++addr;
        ++nBytes;
    }

    return nBytes;
}

/* Zero nWords starting at base */
inline void clearRegion(Memory& mem, uint64_t base, size_t nWords){
    for (size_t i = 0; i < nWords; ++i){
        mem.write(base + i * sizeof(uint32_t), 0);
    }
}



/**
 *  Memory Drivers
 *
 *  OBI responders for the core's instruction and data ports, lifted from
 *  min_tb_inference.cpp. One transaction in flight per port: gnt is held low
 *  while a response is due, rvalid comes the cycle after the accept.
 *
 *  Per-cycle call order:
 *      clk_i = 0; drive(); eval(); sample(); clk_i = 1; eval();
 *
 *  sample() runs on the settled clk_i = 0 state, which is what the coming
 *  posedge latches, so req && gnt there is the real handshake.
 *
 *  Drivers borrow the DUT and Memory (non-owning); both must outlive the
 *  driver. Pin pointers alias the DUT's port storage.
 */
class IMemDriver {

private:
    Memory& imem;

    /* DUT outputs */
    const uint8_t*  req;
    const uint32_t* addr;

    /* DUT inputs */
    uint8_t*  gnt;
    uint8_t*  rvalid;
    uint32_t* rdata;
    uint8_t*  err;

    bool     respDue  = false;
    uint32_t respAddr = 0;

public:

    IMemDriver(Vcve2_top& dut, Memory& imem)
        : imem(imem),
          req(&dut.instr_req_o), addr(&dut.instr_addr_o),
          gnt(&dut.instr_gnt_i), rvalid(&dut.instr_rvalid_i),
          rdata(&dut.instr_rdata_i), err(&dut.instr_err_i) {}

    /* Idle the port: no grant, no response, nothing in flight */
    void reset(){
        respDue = false;
        *gnt = 0; *rvalid = 0; *rdata = 0; *err = 0;
    }

    /* Before eval() with clk_i = 0: respond to last cycle's accept */
    void drive(){
        *rvalid = respDue;
        *err    = 0;
        *rdata  = (respDue && imem.mapped(respAddr)) ? imem.read(respAddr) : 0;
        *gnt    = !respDue;
    }

    /* After eval() with clk_i = 0: latch an accepted fetch */
    void sample(){
        respDue = *req && *gnt;
        if (respDue) respAddr = *addr;
    }
};



class DMemDriver {

private:
    Memory& dmem;

    /* DUT outputs */
    const uint8_t*  req;
    const uint8_t*  we;
    const uint8_t*  be;
    const uint32_t* addr;
    const uint32_t* wdata;

    /* DUT inputs */
    uint8_t*  gnt;
    uint8_t*  rvalid;
    uint32_t* rdata;
    uint8_t*  err;

    bool     respDue     = false;
    bool     respIsWrite = false;
    uint32_t respAddr    = 0;

public:

    DMemDriver(Vcve2_top& dut, Memory& dmem)
        : dmem(dmem),
          req(&dut.data_req_o), we(&dut.data_we_o), be(&dut.data_be_o),
          addr(&dut.data_addr_o), wdata(&dut.data_wdata_o),
          gnt(&dut.data_gnt_i), rvalid(&dut.data_rvalid_i),
          rdata(&dut.data_rdata_i), err(&dut.data_err_i) {}

    /* Idle the port: no grant, no response, nothing in flight */
    void reset(){
        respDue = false;
        *gnt = 0; *rvalid = 0; *rdata = 0; *err = 0;
    }

    /* Before eval() with clk_i = 0: respond to last cycle's accept.
       Reads are serviced here, at response time. */
    void drive(){
        uint32_t data = 0;
        if (respDue && !respIsWrite && dmem.mapped(respAddr))
            data = dmem.read(respAddr);

        *rvalid = respDue;
        *err    = 0;
        *rdata  = data;
        *gnt    = !respDue;
    }

    /* After eval() with clk_i = 0: latch an accepted request.
       Writes are committed here, at accept time. */
    void sample(){
        respDue = *req && *gnt;
        if (!respDue) return;

        respAddr    = *addr;
        respIsWrite = *we;

        if (respIsWrite && dmem.mapped(respAddr)) {
            uint32_t mask = 0;
            /* Byte select logic */
            for (int i = 0; i < 4; i++)
                if (*be & (1u << i)) mask |= 0xFFu << (8 * i);

            /* Full word skips the read so MMIO onRead side effects don't fire */
            uint32_t old = (mask == 0xFFFFFFFFu) ? 0 : dmem.read(respAddr);
            dmem.write(respAddr, (old & ~mask) | (*wdata & mask));
        }
    }
};

