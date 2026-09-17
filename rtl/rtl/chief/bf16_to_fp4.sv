`timescale 1ns/1ps

// [rbs] whole file
// INT4 readout converter
module bf16_to_fp4 (
    input logic     [15:0] bf16_i,
    output logic     [3:0] fp4_o
);

    // round(z * 2^4) first reaches 1 at z = 2^-5, so BF16_EXP(2^-5) = 122
    localparam logic [7:0] LUT_FLOOR = 8'd122;

    logic   sign;
    logic   [7:0] exp;
    logic   [6:0] man;

    assign sign = bf16_i[15];
    assign  exp = bf16_i[14:7];
    assign  man = bf16_i[6:0];

    // Thresholds for the uniform binades, unchanged from the fp4 version:
    //   man >  0  -> above 1.00       man >= 64 -> at/above 1.50
    //   man > 32  -> above 1.25       man >= 96 -> at/above 1.75
    logic man_non_zero, man_33, man_64, man_96;
    assign man_non_zero = |man;
    assign man_33 = man[6] | (man[5] & |man[4:0]);
    assign man_64 = man[6];
    assign man_96 = man[6] & man[5];

    // e=3 covers [4,8), where the integer rolls over at 4.5, 5.5, 6.5.
    logic man_16, man_48, man_80;
    assign man_16 = man[6] | man[5] | (man[4] & |man[3:0]);          // man > 16
    assign man_48 = man[6] | (man[5] & man[4]);                      // man >= 48
    assign man_80 = man[6] & (man[5] | (man[4] & |man[3:0]));        // man > 80

    // e = exp - lut_floor
    logic [1:0] e;
    assign e = exp[1:0] - LUT_FLOOR[1:0];

    logic [2:0] grid;
    always_comb begin
        unique case (e)
            2'd0: grid = {2'b00, man_non_zero};
            2'd1: grid = 3'd1 + {2'b00, man_64};
            2'd2: grid = 3'd2 + {2'b00, man_33} + {2'b00, man_96};
            // fp4 stops at 6 here, int4 carries on to 7
            2'd3: grid = 3'd4 + {2'b00, man_16} + {2'b00, man_48}
                              + {2'b00, man_80};
        endcase
    end

    logic below, clamp;
    assign below = (exp < LUT_FLOOR);
    assign clamp = (exp > 8'd125);

    logic [2:0] mag;
    assign mag = below ? 3'd0 : clamp ? 3'd7 : grid;

    // Two's complement. -8 has no positive twin, so the negative clamp is its
    // own case rather than a negation of the positive one.
    assign fp4_o = (sign & clamp) ? 4'b1000
                  : sign           ? (4'd0 - {1'b0, mag})
                                   : {1'b0, mag};
endmodule
