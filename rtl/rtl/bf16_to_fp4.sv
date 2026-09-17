`timescale 1ns/1ps

// [rbs] whole file
module bf16_to_fp4 (
    input logic     [15:0] bf16_i,
    output logic     [3:0] fp4_o
);

    // BF16_EXP_025 (125) - RDOUT_SHIFT (3) = 122
    // since this is for readout, which always uses a shift of 3, this is okay
    localparam logic [7:0] LUT_FLOOR = 8'd122;

    logic   sign;
    logic   [7:0] exp;
    logic   [6:0] man;

    assign sign = bf16_i[15];
    assign  exp = bf16_i[14:7];
    assign  man = bf16_i[6:0];

    // The FP4 rounding boundaries as quarter points
    // 1.man, so these four thresholds are all the grid needs:
    //   man >  0  -> above 1.00       man >= 64 -> at/above 1.50
    //   man > 32  -> above 1.25       man >= 96 -> at/above 1.75
    // The tie at 1.25 rounds down while the ties at 1.50 and 1.75 round up
    logic man_non_zero, man_33, man_64, man_96;
    assign man_non_zero = |man;
    assign man_33 = man[6] | (man[5] & |man[4:0]);
    assign man_64 = man[6];
    assign man_96 = man[6] & man[5];

    // e = exp - lut_floor
    logic [1:0] e;
    assign e = exp[1:0] - LUT_FLOOR[1:0];

    logic [2:0] grid;
    always_comb begin
        unique case (e)
            2'd0: grid = {2'b00, man_non_zero};
            2'd1: grid = 3'd1 + {2'b00, man_64};
            2'd2: grid = 3'd2 + {2'b00, man_33} + {2'b00, man_96};
            2'd3: grid = 3'd4 + {2'b00, man_33} + {2'b00, man_96};
        endcase
    end

    logic below, clamp;
    assign below = (exp < LUT_FLOOR);
    assign clamp = (exp > 8'd125);

    logic [2:0] mag;
    assign mag = below ? 3'd0 : clamp ? 3'd6 : grid;

    assign fp4_o = {sign & (|mag), mag};
endmodule

    
