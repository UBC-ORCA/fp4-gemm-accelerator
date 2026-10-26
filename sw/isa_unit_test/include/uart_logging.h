#pragma once

/*
    UART logging over MMIO.
    Each character is one 32-bit store to UART_MMIO_ADDR; the tb's UartMMIO
    region echoes it to stdout. 
*/

#include <stdint.h>

#define UART_MMIO_ADDR 0x10000000u

#ifdef __cplusplus
extern "C" {
#endif

/* Write one character */
void putchar_uart(char c);

/* Write a null-terminated string */
void print_str(const char *s);

/* Unsigned decimal */
void putdec(uint32_t n);

/* Unsigned decimal, for totals that can exceed 32 bits */
void putdec64(uint64_t n);

/* Fixed-width 8-digit hex, no 0x prefix */
void print_u32_hex(uint32_t x);

#ifdef __cplusplus
}
#endif
