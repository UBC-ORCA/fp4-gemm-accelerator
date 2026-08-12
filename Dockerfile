# fp4-gemm-accelerator simulation environment
#
# Ubuntu 22.04 + Python 3.10 (fusesoc), Verilator pinned to 5.048.
# Weight headers and test data live in sw/headers (no torch/mptorch needed).
#
# Build (from fp4-gemm-accelerator-merge/):
#   docker build -t [name] .
# Run:
#   docker run -it [name]
#   cd /work/rtl
#   make -f sim.mk run                       # build the Verilator model once
#   ./run_inference.sh hardware 80 --build   # build firmware and run (also: baseline)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RISCV_PREFIX=riscv-none-elf


# 1. System packages: build tools, python, and Verilator build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git make build-essential \
      python3 python3-pip python3-dev \
      autoconf flex bison libfl-dev help2man perl ccache \
      zlib1g-dev libelf-dev vim nano locales libtinfo5 libtinfo6 && \
      locale-gen en_US.UTF-8 && \
      rm -rf /var/lib/apt/lists/*


# 2. RISC-V bare-metal toolchain (xPack riscv-none-elf, with rv32im multilib)
ARG RISCV_XPACK_VER=14.2.0-3
ARG XPACK_URL=https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download
RUN case "$(uname -m)" in \
      x86_64)  ARCH=x64   ;; \
      aarch64) ARCH=arm64 ;; \
      *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;; \
    esac && \
    curl -fsSL "${XPACK_URL}/v${RISCV_XPACK_VER}/xpack-riscv-none-elf-gcc-${RISCV_XPACK_VER}-linux-${ARCH}.tar.gz" \
      | tar -xz -C /opt && \
    ln -s "/opt/xpack-riscv-none-elf-gcc-${RISCV_XPACK_VER}" /opt/riscv
ENV PATH="/opt/riscv/bin:${PATH}"

# Confirm the rv32im link works before build
RUN printf 'int main(void){return 0;}\n' > /tmp/t.c && \
    riscv-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding /tmp/t.c -lgcc -o /tmp/t.elf && \
    echo "[toolchain] rv32im + libgcc OK" && \
    rm -f /tmp/t.c /tmp/t.elf


# 3. Verilator 5.048
RUN git clone https://github.com/verilator/verilator.git /tmp/verilator && \
    cd /tmp/verilator && git checkout v5.048 && \
    autoconf && ./configure && make -j"$(nproc)" && make install && \
    cd / && rm -rf /tmp/verilator && \
    verilator --version


# 4. Python deps
COPY rtl/python-requirements.txt /tmp/python-requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r /tmp/python-requirements.txt


# 5. Project sources: rtl/ and sw/
WORKDIR /work
# COPY rtl/ /work/rtl/
# COPY sw/  /work/sw/

CMD ["/bin/bash"]
