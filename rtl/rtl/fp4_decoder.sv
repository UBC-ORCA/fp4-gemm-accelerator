`timescale 1ns/1ps

module fp4_decoder (

    input  logic [3:0] fp4_i,
    output logic signed [4:0] quanta_o

);

    logic sign;
    logic [2:0] mag;
    logic signed [4:0] q_mag;

    always_comb begin

        sign = fp4_i[3];
        mag  = fp4_i[2:0];

        unique case (mag)
            3'b000: q_mag = 5'sd0;
            3'b001: q_mag = 5'sd1;
            3'b010: q_mag = 5'sd2;
            3'b011: q_mag = 5'sd3;
            3'b100: q_mag = 5'sd4;
            3'b101: q_mag = 5'sd6;
            3'b110: q_mag = 5'sd8;
            3'b111: q_mag = 5'sd12;
            default:q_mag = 5'sd0;
        endcase

        quanta_o = sign ? -q_mag : q_mag;

    end

endmodule
