`timescale 1ns/1ps
/******************************************************************************
 * mac_array.sv
 *
 * TT × TT array of FP4 MAC cells.
 *
 * Responsibilities:
 *   - Decode FP4 inputs into INT5 quanta.
 *   - Broadcast activation quanta across rows.
 *   - Broadcast weight quanta across columns.
 *   - Instantiate TT×TT identical MAC cells.
 *
 * This module is intentionally ISA-independent.
 ******************************************************************************/

module mac_array
import fp4_pkg::*; #(

    parameter int TT = 8

)(

    input logic clk,
    input logic rst_n,

    //----------------------------------------------------------
    // Global control
    //----------------------------------------------------------

    input logic mac_en_i,
    input logic clear_i,

    //----------------------------------------------------------
    // One INT4 activation per row
    //----------------------------------------------------------

    input int4_t act_i [0:TT-1],

    //----------------------------------------------------------
    // One INT4 weight per column
    //----------------------------------------------------------

    input int4_t wt_i [0:TT-1],

    //----------------------------------------------------------
    // Full accumulator tile
    //----------------------------------------------------------

    output logic signed [15:0] accum_o [0:TT-1][0:TT-1]
);

    //----------------------------------------------------------
    // Instantiate TT × TT MAC cells
    //----------------------------------------------------------

    genvar r, c;

    generate

        for (r = 0; r < TT; r++) begin : GEN_ROW

            for (c = 0; c < TT; c++) begin : GEN_COL

                mac_cell u_mac (

                    .clk      (clk),

                    .mac_en_i (mac_en_i),
                    .clear_i  (clear_i),

                    // decoded INT4 quanta
                    .act_i    (act_i[r]),
                    .wt_i     (wt_i[c]),

                    .accum_o  (accum_o[r][c])

                );


            end

        end

    endgenerate


// --- [stev] ---
//============================================================
// Simulation monitor
//============================================================
`ifdef MAC_DEBUG

integer rr, cc;

always_ff @(posedge clk) begin

    $display("");
    $display("======================================================");
    $display("MAC Tile @ time %0t", $time);
    $display("mac_en=%0b  clear=%0b", mac_en_i, clear_i);

    for (rr = 0; rr < TT; rr++) begin

        $write("Row %0d : ", rr);

        for (cc = 0; cc < TT; cc++) begin
            $write("%6d ", accum_o[rr][cc]);
        end

        $write("\n");

    end

    $display("======================================================");
    $display("");

end

`endif
// --- [end] ---

endmodule
