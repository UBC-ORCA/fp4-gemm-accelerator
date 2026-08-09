#include "Vcve2_top.h"
#include "verilated.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef DONE_MMIO_ADDR
#define DONE_MMIO_ADDR 0xFFFF0000u
#endif

#ifndef UART_MMIO_ADDR
#define UART_MMIO_ADDR 0x10000000u
#endif

static vluint64_t main_time = 0;
double sc_time_stamp() { return static_cast<double>(main_time); }

// UART bytes are also teed to this file so the program output can be read
// cleanly, separated from the RTL $display noise on stdout. opened in main()
// only when logging is enabled (default on; disable with --no-uart).
static std::ofstream uart_log;

// derive the UART log filename from the software directory in the hex path, so
// each version writes its own file instead of clobbering a shared one, e.g.
//   ".../sw/inference_fp4mac/inference.hex" -> "uart_out_inference_fp4mac.txt"
static std::string uart_name_from_hex(const std::string& hex_path) {
  size_t slash = hex_path.find_last_of('/');
  if (slash == std::string::npos) return "uart_out.txt";   // no dir -> default
  std::string dir  = hex_path.substr(0, slash);            // strip "/inference.hex"
  size_t prev      = dir.find_last_of('/');
  std::string leaf = (prev == std::string::npos) ? dir : dir.substr(prev + 1);
  return leaf.empty() ? "uart_out.txt" : ("uart_out_" + leaf + ".txt");
}

static constexpr uint32_t IMEM_BASE  = 0x00000000u;
static constexpr uint32_t DMEM_BASE  = 0x80000000u;

static constexpr uint32_t IMEM_BYTES = 512 * 1024;
static constexpr uint32_t DMEM_BYTES = 512 * 1024;

// For loading test data into DMEM
static constexpr uint32_t IMG_LOAD_ADDR  = 0xFFFF0010u;
static constexpr uint32_t IMG_LABEL_ADDR = 0xFFFF0014u;
static constexpr uint32_t IMG_PRED_ADDR  = 0xFFFF0018u;
static constexpr uint32_t IMG_STAGE_ADDR = 0x80070000u;   // in DMEM
static constexpr uint32_t IMG_PIXELS     = 784u;

// host-side dataset, loaded once from a raw .bin (not compiled into the ELF)
static std::vector<uint8_t> host_images, host_labels, host_preds;
static uint32_t img_label = 0, img_pred = 0;   // latched for the selected sample


static std::vector<uint8_t> imem(IMEM_BYTES, 0);
static std::vector<uint8_t> dmem(DMEM_BYTES, 0);

static inline bool imem_translate(uint32_t addr, uint32_t& off) {
  if (addr >= IMEM_BASE && addr < (IMEM_BASE + IMEM_BYTES)) {
    off = addr - IMEM_BASE;
    return true;
  }
  return false;
}

static inline bool dmem_translate(uint32_t addr, uint32_t& idx) {
  if (addr >= DMEM_BASE && addr < (DMEM_BASE + DMEM_BYTES)) {
    idx = addr - DMEM_BASE;
    return true;
  }
  // if (addr < IMEM_BYTES) {
  //   idx = addr;
  //   return true;
  // }
  return false;
}

static inline uint32_t load_le_u32(const std::vector<uint8_t>& mem, uint32_t off) {
  if (off + 3u >= mem.size()) return 0;
  return (uint32_t)mem[off]
       | ((uint32_t)mem[off+1] << 8)
       | ((uint32_t)mem[off+2] << 16)
       | ((uint32_t)mem[off+3] << 24);
}

static inline void store_le_u32(std::vector<uint8_t>& mem, uint32_t off,
                                uint32_t wdata, uint8_t be) {
  if (off + 3u >= mem.size()) return;
  for (int i = 0; i < 4; i++)
    if (be & (1u << i)) mem[off + i] = (uint8_t)((wdata >> (8*i)) & 0xFF);
}

static size_t load_hex(const std::string& path) {
  std::ifstream f(path);
  if (!f) return 0;

  std::string tok;
  uint32_t addr = 0;
  size_t written = 0;
  enum class Dest { IMEM, DMEM, NONE } dst = Dest::IMEM;

  auto from_hex = [](const std::string& s) -> uint32_t {
    return (uint32_t)std::strtoul(s.c_str(), nullptr, 16);
  };

  while (f >> tok) {
    if (tok.empty()) continue;
    if (tok[0] == '@') {
      uint32_t a = from_hex(tok.substr(1));
      uint32_t off = 0;
      if (imem_translate(a, off)) { dst = Dest::IMEM; addr = off; }
      else if (dmem_translate(a, off)) { dst = Dest::DMEM; addr = off; }
      else dst = Dest::NONE;
      continue;
    }
    uint8_t b = (uint8_t)(from_hex(tok) & 0xFF);
    if (dst == Dest::IMEM && addr < imem.size()) { imem[addr++] = b; written++; }
    else if (dst == Dest::DMEM && addr < dmem.size()) { dmem[addr++] = b; written++; }
  }
  return written;
}

// file layout: [u32 n] then n*784 image bytes, then n labels, then n preds
static uint32_t load_dataset(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return 0;
  uint32_t n = 0;
  f.read(reinterpret_cast<char*>(&n), 4);
  host_images.resize((size_t)n * IMG_PIXELS);
  host_labels.resize(n);
  host_preds.resize(n);
  f.read(reinterpret_cast<char*>(host_images.data()), host_images.size());
  f.read(reinterpret_cast<char*>(host_labels.data()), n);
  f.read(reinterpret_cast<char*>(host_preds.data()),  n);
  return n;
}


int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);

  Verilated::commandArgs(argc, argv);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <inference.hex> [--data FILE] [--max-cycles N] [--print-every N]"
                 " [--stall-after N]"
                 " [--trace-if] [--trace-d] [--no-uart] [--uart-file NAME]\n";
    return 1;
  }

  std::string hex_path   = argv[1];
  std::string data_path;
  uint64_t max_cycles    = 50000000;
  uint64_t print_every   = 1000000;
  // Cycles of UART silence before declaring a stall and dumping state. 0 = off.
  uint64_t stall_after   = 2000000;
  bool trace_if          = false;
  bool trace_d           = false;
  bool uart_file         = true;   // tee UART to a file (disable: --no-uart)
  std::string uart_path;           // explicit --uart-file NAME (else derived from hex dir)

  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--data" && i+1 < argc)         data_path   = argv[++i];
    else if (a == "--max-cycles" && i+1 < argc)   max_cycles  = std::stoull(argv[++i]);
    else if (a == "--print-every" && i+1 < argc) print_every = std::stoull(argv[++i]);
    else if (a == "--stall-after" && i+1 < argc) stall_after = std::stoull(argv[++i]);
    else if (a == "--trace-if") trace_if = true;
    else if (a == "--trace-d")  trace_d  = true;
    else if (a == "--no-uart")  uart_file = false;
    else if (a == "--uart-file" && i+1 < argc) uart_path = argv[++i];
    else { std::cerr << "Unknown arg: " << a << "\n"; return 1; }
  }

  // open the UART tee file only if enabled (otherwise it is never created).
  // name defaults to the software dir in the hex path; --uart-file overrides.
  if (uart_file) {
    if (uart_path.empty()) uart_path = uart_name_from_hex(hex_path);
    uart_log.open(uart_path);
    std::cout << "[TB] UART -> " << uart_path << "\n";
  }

  size_t loaded = load_hex(hex_path);
  std::cout << "[TB] Loaded " << hex_path << " (" << loaded << " bytes)\n";

  if (!data_path.empty()) {
    uint32_t nimg = load_dataset(data_path);
    if (nimg == 0) {
      std::cerr << "[TB] ERROR: could not load dataset " << data_path << "\n";
      return 1;
    }
    std::cout << "[TB] Dataset " << data_path << " (" << nimg << " images)\n";
  }

  Vcve2_top* dut = new Vcve2_top();
  dut->clk_i         = 0;
  dut->rst_ni        = 0;
  dut->fetch_enable_i = 0;
  dut->hart_id_i     = 0;
  dut->boot_addr_i   = 0x00000000;
  dut->instr_gnt_i   = 0;
  dut->instr_rvalid_i = 0;
  dut->instr_rdata_i = 0;
  dut->instr_err_i   = 0;
  dut->data_gnt_i    = 0;
  dut->data_rvalid_i = 0;
  dut->data_rdata_i  = 0;
  dut->data_err_i    = 0;

  for (int i = 0; i < 10; i++) {
    dut->clk_i = 0; dut->eval(); main_time++;
    dut->clk_i = 1; dut->eval(); main_time++;
  }
  dut->rst_ni = 1;
  for (int i = 0; i < 5; i++) {
    dut->clk_i = 0; dut->eval(); main_time++;
    dut->clk_i = 1; dut->eval(); main_time++;
  }
  dut->fetch_enable_i = 1;
  std::cout << "[TB] Reset released, running...\n";

  bool if_resp_due  = false;
  uint32_t if_resp_addr = 0;

  // Stall watchdog: ring of the last fetched PCs, plus when UART last moved
  static const int PC_RING = 32;
  uint32_t pc_ring[PC_RING] = {0};
  int      pc_ring_n = 0;
  // Accepted data-bus transactions, to see where an address goes bad
  static const int D_RING = 24;
  uint32_t d_ring_addr[D_RING] = {0};
  bool     d_ring_wr[D_RING]   = {false};
  int      d_ring_n = 0;
  bool     first_bad_seen = false;
  uint64_t last_uart_chars = 0;
  uint64_t last_uart_cyc   = 0;
  bool d_resp_due   = false;
  uint32_t d_resp_addr  = 0;
  bool d_resp_is_write  = false;

  bool done_seen        = false;
  uint64_t done_cycle   = 0;
  uint64_t uart_chars   = 0;

  // wall-clock timing of the run loop (reference only, not for evaluation)
  auto wall_start = std::chrono::steady_clock::now();

  // confirm the clock returned a real timestamp (not the default/epoch value)
  if (wall_start != std::chrono::steady_clock::time_point{})
    std::cout << "[TB] Clock started\n";
  else
    std::cerr << "[TB] WARNING: wall clock did not start\n";

  for (uint64_t cyc = 0; cyc < max_cycles; cyc++) {
    dut->clk_i = 0;

    dut->instr_rvalid_i = if_resp_due ? 1 : 0;
    dut->instr_err_i    = 0;
    if (if_resp_due) {
      uint32_t off = 0;
      uint32_t insn = imem_translate(if_resp_addr, off) ? load_le_u32(imem, off) : 0;
      dut->instr_rdata_i = insn;
      if (trace_if)
        std::printf("[IF] resp pc=0x%08x insn=0x%08x\n", if_resp_addr, insn);
    } else {
      dut->instr_rdata_i = 0;
    }

    dut->data_rvalid_i = d_resp_due ? 1 : 0;
    dut->data_err_i    = 0;
    if (d_resp_due) {
      uint32_t off = 0;
      uint32_t rdata = 0;
      if (!d_resp_is_write) {
        if      (d_resp_addr == IMG_LABEL_ADDR) rdata = img_label;
        else if (d_resp_addr == IMG_PRED_ADDR)  rdata = img_pred;
        else if (imem_translate(d_resp_addr, off)) rdata = load_le_u32(imem, off);
        else if (dmem_translate(d_resp_addr, off)) {
          rdata = load_le_u32(dmem, off);
          static int dbg_rd = 0;
          if (dbg_rd < 12) {
            std::printf("[DBG RD] addr=0x%08x off=0x%x rdata=0x%08x\n",
                        d_resp_addr, off, rdata);
            dbg_rd++;
          }
        }
      }
      dut->data_rdata_i = rdata;
      if (trace_d)
        std::printf("[D ] resp addr=0x%08x (%s) rdata=0x%08x\n",
                    d_resp_addr, d_resp_is_write ? "WR" : "RD", rdata);
    } else {
      dut->data_rdata_i = 0;
    }

    dut->instr_gnt_i = if_resp_due ? 0 : 1;
    dut->data_gnt_i  = d_resp_due  ? 0 : 1;

    dut->eval();
    main_time++;

    bool if_fire = dut->instr_req_o && dut->instr_gnt_i;
    bool d_fire  = dut->data_req_o  && dut->data_gnt_i;

    if_resp_due = false;
    if (if_fire) {
      if_resp_due  = true;
      if_resp_addr = (uint32_t)dut->instr_addr_o;
      pc_ring[pc_ring_n % PC_RING] = if_resp_addr;
      pc_ring_n++;
      if (trace_if) std::printf("[IF] accept pc=0x%08x\n", if_resp_addr);
    }

    d_resp_due = false;
    if (d_fire) {
      d_resp_addr     = (uint32_t)dut->data_addr_o;
      d_resp_is_write = (bool)dut->data_we_o;
      // First data access that is neither mapped memory nor a known MMIO port.
      // Dump the preceding accesses so the good->bad transition is visible.
      if (!first_bad_seen) {
        uint32_t tmp = 0;
        bool mmio  = d_resp_addr == UART_MMIO_ADDR
                  || d_resp_addr == DONE_MMIO_ADDR
                  || d_resp_addr == IMG_LOAD_ADDR
                  || d_resp_addr == IMG_LABEL_ADDR
                  || d_resp_addr == IMG_PRED_ADDR;
        bool known = mmio
                  || ((imem_translate(d_resp_addr, tmp)
                       || dmem_translate(d_resp_addr, tmp))
                      && (d_resp_addr & 3) == 0);   // in range AND word aligned
        if (!known) {
          first_bad_seen = true;
          std::printf("\n[TB] FIRST BAD ACCESS at cyc=%llu: %s 0x%08x%s\n",
                      (unsigned long long)cyc,
                      d_resp_is_write ? "WR" : "RD", d_resp_addr,
                      (d_resp_addr & 3) ? "  UNALIGNED" : "");
          // instr_addr_o is the next fetch address, not the culprit PC. The
          // fetched-PC ring is what shows the instruction stream that led here.
          std::printf("[TB]   next fetch addr 0x%08x (not the culprit PC)\n",
                      (uint32_t)dut->instr_addr_o);
          int qn = pc_ring_n < PC_RING ? pc_ring_n : PC_RING;
          int qs = pc_ring_n < PC_RING ? 0 : (pc_ring_n % PC_RING);
          std::printf("[TB]   last %d fetched PCs:\n", qn);
          for (int k = 0; k < qn; k++) {
            if (k % 8 == 0) std::printf("[TB]    ");
            std::printf(" 0x%08x", pc_ring[(qs + k) % PC_RING]);
            if (k % 8 == 7 || k == qn - 1) std::printf("\n");
          }
          std::printf("[TB]   preceding data accesses:\n");
          int pn = d_ring_n < D_RING ? d_ring_n : D_RING;
          int ps = d_ring_n < D_RING ? 0 : (d_ring_n % D_RING);
          for (int k = 0; k < pn; k++) {
            uint32_t a = d_ring_addr[(ps + k) % D_RING];
            std::printf("[TB]     %s 0x%08x\n",
                        d_ring_wr[(ps + k) % D_RING] ? "WR" : "RD", a);
          }
          std::fflush(stdout);
        }
      }

      d_ring_addr[d_ring_n % D_RING] = d_resp_addr;
      d_ring_wr[d_ring_n % D_RING]   = d_resp_is_write;
      d_ring_n++;
      uint32_t wdata  = (uint32_t)dut->data_wdata_o;
      uint8_t  be     = (uint8_t)dut->data_be_o;
      d_resp_due      = true;

      if (d_resp_is_write) {
        if (d_resp_addr == UART_MMIO_ADDR) {
          char uc = (char)(wdata & 0xFF);
          std::cout << uc << std::flush;
          if (uart_file) uart_log << uc << std::flush;
          uart_chars++;
        } else if (d_resp_addr == DONE_MMIO_ADDR) {
          done_seen  = true;
          done_cycle = cyc;
        } else if (d_resp_addr == IMG_LOAD_ADDR) {
          // stage the selected sample's 784 pixels into DMEM, latch label/pred
          uint32_t idx = wdata;
          uint32_t off = 0;
          if (idx < host_labels.size() && dmem_translate(IMG_STAGE_ADDR, off)) {
            std::memcpy(&dmem[off],
                        &host_images[(size_t)idx * IMG_PIXELS], IMG_PIXELS);
            img_label = host_labels[idx];
            img_pred  = host_preds[idx];
            static int dbg_load = 0;
            if (dbg_load < 3) {
              std::printf("[DBG LOAD] idx=%u off=0x%x staged=[%u %u %u] "
                          "host=[%u %u %u] label=%u pred=%u\n",
                          idx, off, dmem[off], dmem[off+1], dmem[off+2],
                          host_images[(size_t)idx*IMG_PIXELS],
                          host_images[(size_t)idx*IMG_PIXELS+1],
                          host_images[(size_t)idx*IMG_PIXELS+2],
                          img_label, img_pred);
              dbg_load++;
            }
          }
        } else {
          uint32_t off = 0;
          if (imem_translate(d_resp_addr, off))
            store_le_u32(imem, off, wdata, be);
          else if (dmem_translate(d_resp_addr, off))
            store_le_u32(dmem, off, wdata, be);
        }
      }

      if (trace_d)
        std::printf("[D ] accept addr=0x%08x (%s) wdata=0x%08x be=0x%02x\n",
                    d_resp_addr, d_resp_is_write ? "WR" : "RD", wdata, be);
    }

    if (print_every && (cyc % print_every == 0))
      std::printf("[TB] cyc=%llu uart_chars=%llu sleep=%d\n",
                  (unsigned long long)cyc, (unsigned long long)uart_chars,
                  (int)dut->core_sleep_o);

    //------------------------------------------------------------------
    // Stall watchdog. UART goes quiet when the program stops making
    // progress; dump where the core is and what it is waiting on.
    //------------------------------------------------------------------
    if (uart_chars != last_uart_chars) {
      last_uart_chars = uart_chars;
      last_uart_cyc   = cyc;
    } else if (stall_after && !done_seen && cyc > last_uart_cyc + stall_after) {
      std::printf("\n[TB] STALL: no UART for %llu cycles (cyc=%llu)\n",
                  (unsigned long long)(cyc - last_uart_cyc),
                  (unsigned long long)cyc);
      std::printf("[TB]   fetch : req=%d gnt=%d rvalid=%d pc=0x%08x\n",
                  (int)dut->instr_req_o, (int)dut->instr_gnt_i,
                  (int)dut->instr_rvalid_i, (uint32_t)dut->instr_addr_o);
      std::printf("[TB]   data  : req=%d gnt=%d rvalid=%d we=%d addr=0x%08x\n",
                  (int)dut->data_req_o, (int)dut->data_gnt_i,
                  (int)dut->data_rvalid_i, (int)dut->data_we_o,
                  (uint32_t)dut->data_addr_o);
      std::printf("[TB]   sleep=%d  fetches=%d\n",
                  (int)dut->core_sleep_o, pc_ring_n);

      // Last fetched PCs, oldest first. A single repeated address means the
      // core is stalled on one instruction; a short cycle means a spin loop.
      int n = pc_ring_n < PC_RING ? pc_ring_n : PC_RING;
      int start = pc_ring_n < PC_RING ? 0 : (pc_ring_n % PC_RING);
      std::printf("[TB]   last %d fetched PCs:\n", n);
      for (int k = 0; k < n; k++) {
        if (k % 8 == 0) std::printf("[TB]    ");
        std::printf(" 0x%08x", pc_ring[(start + k) % PC_RING]);
        if (k % 8 == 7 || k == n - 1) std::printf("\n");
      }
      // Last accepted data transactions. A run of aligned addresses stepping by
      // 4 is a VMAC64 weight walk; the point where that becomes an unmapped or
      // unaligned address is where the base register went bad.
      int dn = d_ring_n < D_RING ? d_ring_n : D_RING;
      int dstart = d_ring_n < D_RING ? 0 : (d_ring_n % D_RING);
      std::printf("[TB]   last %d data transactions (oldest first):\n", dn);
      for (int k = 0; k < dn; k++) {
        uint32_t a = d_ring_addr[(dstart + k) % D_RING];
        uint32_t off = 0;
        const char *where = imem_translate(a, off) ? "imem"
                          : dmem_translate(a, off) ? "dmem" : "UNMAPPED";
        std::printf("[TB]     %s 0x%08x  %-8s%s\n",
                    d_ring_wr[(dstart + k) % D_RING] ? "WR" : "RD",
                    a, where, (a & 3) ? "  UNALIGNED" : "");
      }
      std::fflush(stdout);
      break;
    }

    dut->clk_i = 1;
    dut->eval();
    main_time++;

    if (done_seen && cyc > done_cycle + 4) break;
  }

  auto wall_end = std::chrono::steady_clock::now();
  double wall_s =
      std::chrono::duration<double>(wall_end - wall_start).count();

  std::cout << "\n";
  std::cout << "[TB] ELAPSED=" << wall_s << " s (wall clock, reference only)\n";

  if (done_seen)
    std::cout << "[TB] DONE at cycle " << done_cycle << "\n";
  else
    std::cerr << "[TB] TIMEOUT after " << max_cycles << " cycles\n";

  dut->final();
  delete dut;
  return done_seen ? 0 : 1;
}
