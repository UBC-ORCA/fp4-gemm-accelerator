`timescale 1ns / 1ps

package mx_pkg;

    // Packed structure representing standard BF16 (bfloat16)
    typedef struct packed {
        logic       sign;
        logic [7:0] exp;
        logic [6:0] mant;
    } bf16_t;

endpackage : mx_pkg
