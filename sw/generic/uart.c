#include <stdint.h>

// IMPORTANT:
// Set this to the same MMIO address your min_tb.cpp watches for UART output.
// Common choices are 0x10000000 or 0x80000000 depending on the testbench.
#ifndef UART_MMIO_ADDR
#define UART_MMIO_ADDR 0x10000000u
#endif

void putchar_uart(char c) {
  volatile uint32_t *uart = (volatile uint32_t *)UART_MMIO_ADDR;
  *uart = (uint32_t)(uint8_t)c;
}