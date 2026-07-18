`timescale 1ns / 1ps

module mac_scale_accum 
import mx_pkg::*;
(
    input  logic signed [15:0] tile_value,
    input  logic [7:0]         scaleA,
    input  logic [7:0]         scaleB,
    input  logic [15:0]        accumulator,
    output logic [15:0]        accumulator_out
);

    // Intermediate structured wiring signals
    bf16_t bf16_tile;
    bf16_t bf16_scaled;

    // 1. Convert Int16 Accumulator Matrix Output into standard BF16 format
    int16_to_bf16 u_convert (
        .int_in   (tile_value),
        .bf16_out (bf16_tile)
    );

    // 2. Adjust Exponents dynamically utilizing E8M0 elements
    e8m0_scale u_scale (
        .bf16_in  (bf16_tile),
        .scale_a  (scaleA),
        .scale_b  (scaleB),
        .bf16_out (bf16_scaled)
    );

    // 3. Accumulate with BRAM memory context lines
    bf16_accumulate u_add (
        .bf16_scaled     (bf16_scaled),
        .accumulator_in  (accumulator),
        .accumulator_out (accumulator_out)
    );

//------------------------------------------------------------
// MAC Scale Accum Internal Debug
//------------------------------------------------------------
always_comb begin

    $display("");
    $display("==============================================================");
    $display("[MAC_SCALE_ACCUM_DEBUG]");

    $display("TIME = %0t ns",$time);


    //------------------------------------------------------------
    // Input information
    //------------------------------------------------------------

    $display("");
    $display("INPUTS");
    $display("--------------------------------------------------------------");

    $display("tile_value       = %0d (0x%h)",
             $signed(tile_value),
             tile_value);

    $display("scaleA           = 0x%02h",
             scaleA);

    $display("scaleB           = 0x%02h",
             scaleB);

    $display("accumulator IN   = 0x%04h",
             accumulator);



    //------------------------------------------------------------
    // INT16 -> BF16 conversion
    //------------------------------------------------------------

    $display("");
    $display("INT16 TO BF16");
    $display("--------------------------------------------------------------");

    $display("BF16 tile        = 0x%04h",
             bf16_tile);

    $display("BF16 sign        = %b",
             bf16_tile[15]);

    $display("BF16 exponent    = 0x%02h",
             bf16_tile[14:7]);

    $display("BF16 mantissa    = 0x%02h",
             bf16_tile[6:0]);



    //------------------------------------------------------------
    // Scaling stage
    //------------------------------------------------------------

    $display("");
    $display("E8M0 SCALE");
    $display("--------------------------------------------------------------");


    $display("Scale A exponent = 0x%02h",
             scaleA);

    $display("Scale B exponent = 0x%02h",
             scaleB);


    $display("Input BF16       = 0x%04h",
             bf16_tile);


    $display("Scaled BF16      = 0x%04h",
             bf16_scaled);


    $display("Scaled sign      = %b",
             bf16_scaled[15]);

    $display("Scaled exponent  = 0x%02h",
             bf16_scaled[14:7]);

    $display("Scaled mantissa  = 0x%02h",
             bf16_scaled[6:0]);



    //------------------------------------------------------------
    // Accumulator input
    //------------------------------------------------------------

    $display("");
    $display("BF16 ACCUMULATOR INPUT");
    $display("--------------------------------------------------------------");

    $display("Old accumulator  = 0x%04h",
             accumulator);


    $display("New scaled value = 0x%04h",
             bf16_scaled);



    //------------------------------------------------------------
    // Expected mathematical operation
    //------------------------------------------------------------

    $display("");
    $display("EXPECTED OPERATION");
    $display("--------------------------------------------------------------");

    $display("BF16:");
    $display("    0x%04h + 0x%04h",
             accumulator,
             bf16_scaled);


    //------------------------------------------------------------
    // Output
    //------------------------------------------------------------

    $display("");
    $display("ACCUMULATOR OUTPUT");
    $display("--------------------------------------------------------------");


    $display("accumulator_out = 0x%04h",
             accumulator_out);



    //------------------------------------------------------------
    // Sanity checks
    //------------------------------------------------------------


    if(accumulator == bf16_scaled) begin
        $display("");
        $display("WARNING:");
        $display("Accumulator input equals scaled value");
        $display("Expected output should be different");
    end


    if(accumulator_out == accumulator) begin
        $display("");
        $display("ERROR:");
        $display("Accumulator output unchanged!");
        $display("BF16 adder may be bypassed");
    end


    if(accumulator_out == bf16_scaled) begin
        $display("");
        $display("ERROR:");
        $display("Accumulator output equals scaled value!");
        $display("Old accumulator may be ignored");
    end


    $display("");
    $display("==============================================================");
    $display("");

end

endmodule
