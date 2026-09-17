`timescale 1ns / 1ps

module bf16_accumulate 
import mx_pkg::*;
(
    input  bf16_t       bf16_scaled,
    input  logic [15:0] accumulator_in,
    output logic [15:0] accumulator_out
);

    // Convert the incoming packed struct cleanly into a logic array
    logic [15:0] scaled_vector;
    assign scaled_vector = {bf16_scaled.sign, bf16_scaled.exp, bf16_scaled.mant};

    // Instantiate your existing parameterized FP/BF16 FP adder here
    // Replace "parameterized_adder" with your design's actual module name if needed.
    parameterized_adder #(
        .EXP_WIDTH(8),
        .MANT_WIDTH(7)
    ) u_fp_adder (
        .a(scaled_vector),
        .b(accumulator_in),
        .sum(accumulator_out)
    );

endmodule
