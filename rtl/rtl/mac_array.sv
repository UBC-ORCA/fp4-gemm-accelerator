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
    // One FP4 activation per row
    //----------------------------------------------------------

    input fp4_e2m1_t act_i [0:TT-1],

    //----------------------------------------------------------
    // One FP4 weight per column
    //----------------------------------------------------------

    input fp4_e2m1_t wt_i [0:TT-1],

    //----------------------------------------------------------
    // Full accumulator tile
    //----------------------------------------------------------

    output logic signed [15:0] accum_o [0:TT-1][0:TT-1],

    // mv inst
	input logic        mv_en_i,
	input logic [1:0]  mv_mode_i,   // 0=even 1=odd 2=pair
	input logic [2:0] mv_even_col_idx_i,
	input logic [2:0] mv_odd_col_idx_i,
	input logic [2:0] mv_row_idx_i,
output logic [31:0] mv_data_o

);

    //----------------------------------------------------------
    // Instantiate TT × TT MAC cells
    //----------------------------------------------------------
localparam logic [1:0] MV_EVEN = 2'd0;
localparam logic [1:0] MV_ODD  = 2'd1;
localparam logic [1:0] MV_PAIR = 2'd2;

    genvar r, c;

    generate

        for (r = 0; r < TT; r++) begin : GEN_ROW

            for (c = 0; c < TT; c++) begin : GEN_COL
                logic cell_mv_clear; //for mv

                assign cell_mv_clear =
                    mv_en_i &&
                    (
                        ((mv_mode_i == MV_EVEN) &&
                        (r == mv_row_idx_i) &&
                        (c == mv_even_col_idx_i))

                    ||

                        ((mv_mode_i == MV_ODD) &&
                        (r == mv_row_idx_i) &&
                        (c == mv_odd_col_idx_i))

                    ||

                        ((mv_mode_i == MV_PAIR) &&
                        (r == mv_row_idx_i) &&
                        (
                            (c == mv_even_col_idx_i) ||
                            (c == mv_odd_col_idx_i)
                        ))
                    );


                mac_cell u_mac (

                    .clk      (clk),

                    .mac_en_i (mac_en_i),
                    .clear_i  (clear_i),
		             .mv_clear_i(cell_mv_clear), // for mv

                    // decoded FP4 quanta
                    .act_i    (act_i[r]),
                    .wt_i     (wt_i[c]),

                    .accum_o  (accum_o[r][c])

                );


            end

        end

    endgenerate

//MV_rd
always_comb begin

    mv_data_o = 32'h0;

    unique case (mv_mode_i)

        MV_EVEN:
            mv_data_o =
            {{16{accum_o[mv_row_idx_i][mv_even_col_idx_i][15]}},
              accum_o[mv_row_idx_i][mv_even_col_idx_i]};

        MV_ODD:
            mv_data_o =
            {{16{accum_o[mv_row_idx_i][mv_odd_col_idx_i][15]}},
              accum_o[mv_row_idx_i][mv_odd_col_idx_i]};

        MV_PAIR:
            mv_data_o =
            {
                accum_o[mv_row_idx_i][mv_odd_col_idx_i],
                accum_o[mv_row_idx_i][mv_even_col_idx_i]
            };

        default: ;

    endcase

end
//MV_rd_end




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
