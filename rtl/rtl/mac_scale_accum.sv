`timescale 1ns / 1ps

module mac_scale_accum 
import mx_pkg::*;
(
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic signed [15:0] tile_value,
    input  logic [7:0]         scaleA,
    input  logic [7:0]         scaleB,
    input  logic [15:0]        accumulator,
    output logic [15:0]        accumulator_out
);

    //------------------------------------------------------------
    // Pipeline Registers & Internal Signals
    //------------------------------------------------------------
    
    // Stage 1 Registers (Input Capture / Work-Item Boundary)
    logic signed [15:0] tile_value_s1;
    logic [7:0]         scaleA_s1;
    logic [7:0]         scaleB_s1;
    logic [15:0]        accumulator_s1;

    // Stage 2 Signals & Registers (Post INT16 -> BF16 Conversion)
    bf16_t              bf16_tile_comb;
    bf16_t              bf16_tile_s2;
    logic [7:0]         scaleA_s2;
    logic [7:0]         scaleB_s2;
    logic [15:0]        accumulator_s2;

    // Stage 3 Signals & Registers (Post E8M0 Scaling)
    bf16_t              bf16_scaled_comb;
    bf16_t              bf16_scaled_s3;
    logic [15:0]        accumulator_s3;

    // Stage 4 Signals (Post BF16 Accumulation)
    logic [15:0]        accumulator_out_comb;

    //------------------------------------------------------------
    // PIPELINE STAGE 1: Input Work-Item Capture
    //------------------------------------------------------------
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            tile_value_s1  <= '0;
            scaleA_s1      <= '0;
            scaleB_s1      <= '0;
            accumulator_s1 <= '0;
        end else begin
            tile_value_s1  <= tile_value;
            scaleA_s1      <= scaleA;
            scaleB_s1      <= scaleB;
            accumulator_s1 <= accumulator;
        end
    end

    //------------------------------------------------------------
    // PIPELINE STAGE 2: INT16 -> BF16 Conversion
    //------------------------------------------------------------
    int16_to_bf16 u_convert (
        .int_in   (tile_value_s1),
        .bf16_out (bf16_tile_comb)
    );

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            bf16_tile_s2   <= '0;
            scaleA_s2      <= '0;
            scaleB_s2      <= '0;
            accumulator_s2 <= '0;
        end else begin
            bf16_tile_s2   <= bf16_tile_comb;
            scaleA_s2      <= scaleA_s1;
            scaleB_s2      <= scaleB_s1;
            accumulator_s2 <= accumulator_s1;
        end
    end

    //------------------------------------------------------------
    // PIPELINE STAGE 3: E8M0 Scaling
    //------------------------------------------------------------
    e8m0_scale u_scale (
        .bf16_in  (bf16_tile_s2),
        .scale_a  (scaleA_s2),
        .scale_b  (scaleB_s2),
        .bf16_out (bf16_scaled_comb)
    );

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            bf16_scaled_s3 <= '0;
            accumulator_s3 <= '0;
        end else begin
            bf16_scaled_s3 <= bf16_scaled_comb;
            accumulator_s3 <= accumulator_s2;
        end
    end

    //------------------------------------------------------------
    // PIPELINE STAGE 4: BF16 Accumulation
    //------------------------------------------------------------
    bf16_accumulate u_add (
        .bf16_scaled     (bf16_scaled_s3),
        .accumulator_in  (accumulator_s3),
        .accumulator_out (accumulator_out_comb)
    );

    // Module output driven directly by the final computing stage 
    assign accumulator_out = accumulator_out_comb;

    //------------------------------------------------------------
    // MAC Scale Accum Internal Debug (Adapted for Stage Synchrony)
    //------------------------------------------------------------
`ifdef MAC_DEBUG
    always @(posedge clk_i) begin
        $display("");
        $display("==============================================================");
        $display("[MAC_SCALE_ACCUM_PIPELINE_DEBUG]");
        $display("TIME = %0t ns", $time);

        // Stage 1 Monitoring
        $display("\n[STAGE 1]: Captured Inputs");
        $display("--------------------------------------------------------------");
        $display("tile_value_s1    = %0d (0x%h)", $signed(tile_value_s1), tile_value_s1);
        $display("scaleA_s1        = 0x%02h", scaleA_s1);
        $display("scaleB_s1        = 0x%02h", scaleB_s1);
        $display("accumulator_s1   = 0x%04h", accumulator_s1);

        // Stage 2 Monitoring
        $display("\n[STAGE 2]: INT16 -> BF16 Output");
        $display("--------------------------------------------------------------");
        $display("bf16_tile_s2     = 0x%04h", bf16_tile_s2);
        $display("BF16 sign        = %b",    bf16_tile_s2[15]);
        $display("BF16 exponent    = 0x%02h", bf16_tile_s2[14:7]);
        $display("BF16 mantissa    = 0x%02h", bf16_tile_s2[6:0]);

        // Stage 3 Monitoring
        $display("\n[STAGE 3]: E8M0 Scale Output");
        $display("--------------------------------------------------------------");
        $display("bf16_scaled_s3   = 0x%04h", bf16_scaled_s3);
        $display("Scaled sign      = %b",    bf16_scaled_s3[15]);
        $display("Scaled exponent  = 0x%02h", bf16_scaled_s3[14:7]);
        $display("Scaled mantissa  = 0x%02h", bf16_scaled_s3[6:0]);

        // Stage 4 Monitoring
        $display("\n[STAGE 4]: Accumulator Output");
        $display("--------------------------------------------------------------");
        $display("accumulator_out  = 0x%04h", accumulator_out);

        // Operational Sanity Checks
        if(accumulator_s3 == bf16_scaled_s3) begin
            $display("\nWARNING: Accumulator context equals scaled value at Stage 4 evaluation.");
        end

        if(accumulator_out == accumulator_s3) begin
            $display("\nERROR: Accumulator output unchanged! BF16 adder may be bypassed.");
        end
        $display("==============================================================");
    end
`endif

endmodule
