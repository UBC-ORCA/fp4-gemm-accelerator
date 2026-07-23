# matmul8 Vector Test Build Instructions

This directory contains the build files for the `matmul8` vector test.

## Building the Test Program

To compile the firmware and generate the Verilog HEX image, run:

```bash
make -f matmult8.mk
```

This command will:

1. Compile and link the source files into:
   - `matmul8_vec.elf`

2. Generate a linker map file:
   - `matmul8.map`

3. Convert the ELF executable into a Verilog-compatible memory image:
   - `matmul8_vec.hex`

## Generated Files

| File | Description |
|--------|-------------|
| `matmul8_vec.elf` | RISC-V executable |
| `matmul8.map` | Linker memory map |
| `matmul8_vec.hex` | Verilog memory initialization file |

## Cleaning Build Artifacts

To remove all generated files, run:

```bash
make -f matmult8.mk clean
```

## Requirements

The following tools must be available in your PATH:

- `riscv32-unknown-elf-gcc`
- `riscv32-unknown-elf-objcopy`

These are typically provided by a RISC-V GCC toolchain installation.
  - https://github.com/riscv-collab/riscv-gnu-toolchain/releases

## Source File Locations

**matmul8_vec.S**
  - https://github.com/UBC-ORCA/cve2-tinyrvv/blob/main/sw/benchmarks/matrix/matmul8_vec.S

**matmul8_vec_test.c**
  - https://github.com/UBC-ORCA/cve2-tinyrvv/blob/main/sw/benchmarks/matrix/matmul8_vec_test.c

**matmul8_shared_link.ld**
  - https://github.com/JerryYun2004/RISC-V-RVV-Lite/blob/LUTRAM-VRF/sw/lint/matmul8_shared_link.ld

**start.S** and **uart.c**
  - https://github.com/JerryYun2004/RISC-V-RVV-Lite/tree/LUTRAM-VRF/sw/support

## Example Commands

The following commands are executed by `make -f matmult8.mk`.

### Build ELF Executable

```bash
riscv32-unknown-elf-gcc \
  -T matmul8_shared_link.ld \
  -march=rv32im \
  -mabi=ilp32 \
  -nostdlib -ffreestanding \
  -Wl,-Map=matmul8.map \
  start.S uart.c matmul8_vec_test.c matmul8_vec.S \
  -o matmul8_vec.elf
```

This command compiles and links the application into the RISC-V executable:

- `matmul8_vec.elf`

and generates the linker map:

- `matmul8.map`

### Generate Verilog HEX Image

```bash
riscv32-unknown-elf-objcopy \
  -O verilog \
  matmul8_vec.elf \
  matmul8_vec.hex
```

This command converts the ELF executable into a Verilog memory initialization file:

- `matmul8_vec.hex`

which can be loaded into instruction memory for simulation or FPGA execution.
