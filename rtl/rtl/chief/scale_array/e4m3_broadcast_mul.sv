
`timescale 1ns / 1ps


(* use_dsp = "yes"*)
module e4m3_broadcast_mul
import chief_pkg::*;
#(
    parameter int N_SCALE_UNITS = 4, 
    localparam int LANE_PROD_WIDTH = 2*E4M3_M+1
)(
    input logic [E4M3_M:0] row_scale_mant_ext_i [0:N_SCALE_UNITS-1],
    input logic [E4M3_M:0] col_scale_mant_ext_i, 

    /* Handoff signals for the next pipeline stage */
    output logic row_scale_mant_hidden_d_o [0:N_SCALE_UNITS-1],
    output logic [LANE_PROD_WIDTH-1:0] prod_d_o [0:N_SCALE_UNITS-1]
);  

/*   
    DSP Mapping 
*/
localparam int DSP_A_W = 27;
localparam int DSP_B_W = 18;
localparam int DSP_M_W = 48;

// Lane product width = (m3 + hidden bit) + m3
localparam int LANE_ELEM_WIDTH = E4M3_M+1;

logic signed [DSP_A_W-1:0] dsp_a; 
logic signed [DSP_B_W-1:0] dsp_b;
logic signed [DSP_M_W-1:0] dsp_m;

always_comb begin 
    dsp_a = 'b0;
    dsp_b = col_scale_mant_ext_i;
    for (int i = 0; i < N_SCALE_UNITS; ++i) begin 
        dsp_a[LANE_PROD_WIDTH * i +: LANE_ELEM_WIDTH] = row_scale_mant_ext_i[i][E4M3_M-1:0];
    end

    dsp_m = dsp_a * dsp_b;
end

/* Unpack the lane fields. Only the 3-bit row mantissa goes into the DSP;
   the row hidden bit is handed off so the next stage can add
   hidden * col_scale_mant_ext << E4M3_M. */
always_comb begin
    for (int i = 0; i < N_SCALE_UNITS; ++i) begin
        prod_d_o[i]                  = dsp_m[LANE_PROD_WIDTH * i +: LANE_PROD_WIDTH];
        row_scale_mant_hidden_d_o[i] = row_scale_mant_ext_i[i][E4M3_M];
    end
end

endmodule
