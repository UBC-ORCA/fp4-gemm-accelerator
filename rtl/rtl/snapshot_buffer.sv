module snapshot_buffer #(
    parameter TT=8
)(
    input clk,
    input rst_n,

    input snapshot_en_i,

    input signed [15:0] accum_i [0:TT-1][0:TT-1],

    output logic signed [15:0] snapshot_o [0:TT-1][0:TT-1]
);


integer r,c;

always_ff @(posedge clk) begin

    if(!rst_n) begin

        for(r=0;r<TT;r++)
            for(c=0;c<TT;c++)
                snapshot_o[r][c] <= '0;

    end

    else if(snapshot_en_i) begin

        for(r=0;r<TT;r++)
            for(c=0;c<TT;c++)
                snapshot_o[r][c] <= accum_i[r][c];

    end

end

endmodule
