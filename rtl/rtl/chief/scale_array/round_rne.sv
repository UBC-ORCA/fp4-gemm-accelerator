`timescale 1ns / 1ps

/*
    Round-to-nearest, ties-to-even (RNE).

    Keeps the top OUTPUT_WIDTH bits of round_input_in and rounds away the
    remaining DROP = INPUT_WIDTH - OUTPUT_WIDTH LSBs.

        kept | g | r | s...
             ^ first dropped bit (half-ULP)

        round_up = g & (r | s | kept[0])

    round_output_o : rounded value (wraps to 0 on carry-out, i.e. a mantissa
                     of all ones rounds up to 1.000... -> exponent += 1)
    round_carry_o  : carry-out of the rounding increment
*/
module round_rne
#(
    parameter int INPUT_WIDTH,
    parameter int OUTPUT_WIDTH
) (
    input  logic [INPUT_WIDTH-1:0]  round_input_in,
    output logic [OUTPUT_WIDTH-1:0] round_output_o,
    output logic                    round_carry_o
);

    localparam int DROP = INPUT_WIDTH - OUTPUT_WIDTH;

    initial begin
        if (DROP < 0)
            $error("round_rne: OUTPUT_WIDTH (%0d) > INPUT_WIDTH (%0d)",
                   OUTPUT_WIDTH, INPUT_WIDTH);
    end

    logic [INPUT_WIDTH+2:0]  ext;
    logic [OUTPUT_WIDTH-1:0] kept;
    logic                    g, r, s;
    logic                    round_up;

    assign ext  = {round_input_in, 3'b000};

    assign kept = ext[INPUT_WIDTH+2 -: OUTPUT_WIDTH];
    assign g    = ext[DROP+2];          // guard  (half ULP)
    assign r    = ext[DROP+1];          // round
    assign s    = |ext[DROP:0];         // sticky (OR of the rest)

    assign round_up = g & (r | s | kept[0]);
    assign {round_carry_o, round_output_o} = {1'b0, kept} + (OUTPUT_WIDTH+1)'(round_up);

endmodule
