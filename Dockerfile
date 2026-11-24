###############################################################################
# CHIEF simulation environment
#
# Debian 13 Trixie (Slim), Python 3.12 (for fusesoc), Verilator 5.052
#
# Build (from repo root):
#   docker build -t [name] .
# Run:
#   docker run -it [name]
#   cd /work/rtl
#   make -f sim.mk run                      # build the Verilator model
#   ./run_inference.sh hardware mnist 10k   # build and run mnist 10k inference
###############################################################################

FROM python:3.12.16-slim-trixie

ARG DEBIAN_FRONTEND=noninteractive

ENV RISCV_PREFIX=riscv-none-elf

###############################################################################
# 1. Install system packages: build tools, python, dependencies
###############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        make \
        build-essential \
        autoconf \
        flex \
        bison \
        libfl-dev \
        help2man \
        perl \
        ccache \
        zlib1g-dev \
        libelf-dev \
        liblz4-dev \
        libtinfo6 \
    && rm -rf /var/lib/apt/lists/*

###############################################################################
# 2. Install Python requirements from python-requirements.txt
###############################################################################
COPY rtl/python-requirements.txt /tmp/python-requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /tmp/python-requirements.txt && \
    rm /tmp/python-requirements.txt

###############################################################################
# 3. xPack RISC-V GCC 15.2 toolchain
###############################################################################
ARG RISCV_XPACK_VERSION=15.2.0-1

RUN set -eux; \
    case "$(dpkg --print-architecture)" in \
        amd64) ARCH=x64 ;; \
        arm64) ARCH=arm64 ;; \
        *) echo "Unsupported architecture"; exit 1 ;; \
    esac; \
    ARCHIVE="xpack-riscv-none-elf-gcc-${RISCV_XPACK_VERSION}-linux-${ARCH}.tar.gz"; \
    curl -fsSL --retry 3 \
        "https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v${RISCV_XPACK_VERSION}/${ARCHIVE}" \
        -o "/tmp/${ARCHIVE}"; \
    tar -xzf "/tmp/${ARCHIVE}" -C /opt; \
    ln -s "/opt/xpack-riscv-none-elf-gcc-${RISCV_XPACK_VERSION}" /opt/riscv; \
    rm "/tmp/${ARCHIVE}"

ENV PATH="/opt/riscv/bin:${PATH}"

###############################################################################
# 4. Verify the RISC-V toolchain is set up correctly
###############################################################################
RUN set -eux; \
    riscv-none-elf-gcc --version; \
    riscv-none-elf-gcc -print-multi-lib | grep 'rv32im/ilp32'; \
    printf 'void _start(void) { for (;;) {} }\n' > /tmp/test.c; \
    riscv-none-elf-gcc \
        -march=rv32im \
        -mabi=ilp32 \
        -nostdlib \
        -ffreestanding \
        -Wl,-e,_start \
        /tmp/test.c \
        -lgcc \
        -o /tmp/test.elf; \
    echo "[toolchain] RV32IM / ILP32 linking OK"; \
    rm -f /tmp/test.c /tmp/test.elf

###############################################################################
# 5. Clone and build Verilator 5.052
###############################################################################
ARG VERILATOR_VERSION=5.052

RUN set -eux; \
    git clone --depth 1 \
        --branch "v${VERILATOR_VERSION}" \
        https://github.com/verilator/verilator.git \
        /tmp/verilator; \
    cd /tmp/verilator; \
    autoconf; \
    ./configure; \
    make -j"$(nproc)"; \
    make install; \
    cd /; \
    rm -rf /tmp/verilator; \
    verilator --version | grep -F "Verilator ${VERILATOR_VERSION}"

###############################################################################
# 6. Copy project sources as rtl/ and sw/
###############################################################################
WORKDIR /work
COPY rtl/ /work/rtl/
COPY sw/  /work/sw/

CMD ["/bin/bash"]
