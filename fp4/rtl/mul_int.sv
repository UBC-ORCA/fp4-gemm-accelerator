
/**
    Multiplies 2 FP4 numbers, convert
    the result to a 16 bit number.
**/


`define INT_OUTPUT_SIZE_BITS 9
`define INT_OUTPUT_SIZE_MAG_BITS (`INT_OUTPUT_SIZE_BITS - 1) 

module fp4_mul_int (
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
assign mantB = FP4inB[0];

/* 
    Hard-coded LUT to encode the logic
    to an int4 space.

    The int4 encoding is a Q3.1 fixed 
    point value. 
 */
localparam logic [3:0] e2m1_enc[0:7] = '{
     /* 3'b000 => 0.0 */  4'd0,
     /* 3'b001 => 0.5 */  4'd1,
     /* 3'b010 => 1.0 */  4'd2,
     /* 3'b011 => 1.5 */  4'd3,
     /* 3'b100 => 2.0 */  4'd4,
     /* 3'b101 => 3.0 */  4'd6,
     /* 3'b110 => 4.0 */  4'd8,
     /* 3'b111 => 6.0 */  4'd12
};

logic [3:0] int4A, int4B;
assign int4A = e2m1_enc[{expA, mantA}];
assign int4B = e2m1_enc[{expB, mantB}];
assign outMag = int4A * int4B;

endmodule
