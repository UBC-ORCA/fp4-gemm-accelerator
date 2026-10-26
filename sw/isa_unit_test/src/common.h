/**
 *  Common utility functions
 */
#pragma once

#include "mmio.h"
#include "stdbool.h"
#include "uart_logging.h"
#include <stdint.h>

/*
    Report the exit code, raise DONE for the tb, and park.
    Same DONE protocol as start.S, so returning from main also ends the sim.
*/
#define exit(code)               \
    do {                         \
        *STATUS_MMIO = (code);   \
        *DONE_MMIO = 1;          \
        for (;;) {}              \
    } while (0)


static inline void assert(bool condition){
    if (condition){
        return;
    } 

    /* 
        TODO: Print the error message 
            to the serial interface.
    */

    exit(-1);
}


static inline void assert_msg(bool condition, 
                              const char *msg){
    if (condition){
        return;
    }

    print_str(msg);
    print_str("\n");
    exit(-1);
}

typedef uint8_t fp4_t;
typedef uint16_t bf16_t; 

