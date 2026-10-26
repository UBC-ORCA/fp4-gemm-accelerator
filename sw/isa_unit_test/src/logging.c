#include "uart_logging.h"

// =======================================
// UART output helpers
// =======================================
// Lifted from sw/generic/uart.c (simulator version) and inference_nvfp4.c

void putchar_uart(char c) {
    volatile uint32_t *uart = (volatile uint32_t *)UART_MMIO_ADDR;
    *uart = (uint32_t)(uint8_t)c;
}

void print_str(const char *s) {
    while (*s) putchar_uart(*s++);
}

void putdec(uint32_t n) {
    char buf[11];
    int i = 0;
    if (n == 0) {
        putchar_uart('0');
        return;
    }
    while (n > 0) {
        buf[i++] = '0' + (n % 10);
        n /= 10;
    }
    while (i--) putchar_uart(buf[i]);
}

void putdec64(uint64_t n) {
    char buf[21];
    int i = 0;
    if (n == 0) {
        putchar_uart('0');
        return;
    }
    while (n > 0) {
        buf[i++] = '0' + (int)(n % 10);
        n /= 10;
    }
    while (i--) putchar_uart(buf[i]);
}

void print_u32_hex(uint32_t x) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 7; i >= 0; --i) putchar_uart(hex[(x >> (i * 4)) & 0xFu]);
}
