`timescale 1ns / 1ps

package fp4_pkg;

typedef enum int  { 
    FP4_SCALE_MXE8M0, 
    FP4_SCALE_E4M3
} fp4_scale_format_t;

typedef struct packed {
    logic sign; 
    logic [3:0] exp;
    logic [2:0] mant;
} fp4_scaler_e4m3_t;

typedef struct packed {
    logic sign;
    logic [1:0] exp;
    logic mant;
} fp4_e2m1_t;

typedef logic signed [3:0] int4_t;

typedef struct packed {
    /* MX format of the scaler E8M0 has no sign bits */
    logic [7:0] exp;
} fp4_scaler_mxe8m0_t;

typedef union packed {
    fp4_scaler_e4m3_t e4m3;
    fp4_scaler_mxe8m0_t mxe8m0;
    logic [7:0] raw;
} fp4_scaler_t;

/* E8M7 */
typedef struct packed {
    logic sign; 
    logic [7:0] exp;
    logic [6:0] mant;
} bf16_t;
    
endpackage : fp4_pkg
 

