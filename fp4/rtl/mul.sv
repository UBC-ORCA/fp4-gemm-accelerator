
/**
    Multiplies 2 FP4 numbers, convert
    the result to a 16 bit number.
**/


`define INT_OUTPUT_SIZE_BITS 9
`define INT_OUTPUT_SIZE_MAG_BITS (`INT_OUTPUT_SIZE_BITS - 1) 

module fp4_mul (
    FP4inA, FP4inB,  
    int9Out
);

input logic [3:0] FP4inA;
input logic [3:0] FP4inB;
output logic [`INT_OUTPUT_SIZE_BITS-1:0] int9Out;

logic outSign;
logic [`INT_OUTPUT_SIZE_MAG_BITS - 1:0] outMag;
assign int9Out[`INT_OUTPUT_SIZE_BITS - 1] = outSign;
assign int9Out[`INT_OUTPUT_SIZE_MAG_BITS - 1 : 0] = outMag; 

logic signA, signB;
logic signOut;
logic [1:0] expA, expB;
logic mantA, mantB; 

assign signA = FP4inA[3];
assign signB = FP4inB[3];

/* Output sign calculation */
assign outSign = signA ^ signB;

assign expA = FP4inA[2:1];
assign expB = FP4inB[2:1];
assign mantA = FP4inA[0];
assign mantA = FP4inA[0];

logic isZeroA, isZeroB;
logic isSubnormalA, isSubnormalB; 

assign isSubnormalA = expA == 2'b0;
assign isSubnormalB = expB == 2'b0;
assign isZeroA = isSubnormalA & (mantA == 1'b0);
assign isZeroB = isSubnormalB & (mantB == 1'b0);

/* 
    Hard-coded LUT to encode the logic
    to an int4 space. 
 */
logic [7:0] outLUT[0:63];

localparam real fp4_vals [0:7] = '{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
localparam logic [2:0] e2m1_enc [0:7] = '{3'b000, 3'b001, 3'b010, 3'b011, 3'b100, 3'b101, 3'b110, 3'b111};

/* LUT Table generation, this essentially hardcodes the LUT */
initial begin
    int result;

    for (int i = 0; i < 8; ++i) begin
        for (int j = 0; j < 8; ++j) begin
            result = int(fp4_vals[i] / 0.5) * int(fp4_vals[j] / 0.5);
            outLUT[{e2m1_enc[i], e2m1_enc[j]}] = result[7:0];        
        end
    end
end

assign outMag = outLUT[{expA, mantA, expB, mantB}];

endmodule;
