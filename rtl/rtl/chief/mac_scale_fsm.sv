`timescale 1ns/1ps

/**
    Control wrapper for scale unit, converts pipeline signals
    from scale unit to proper BRAM controls.
**/
module mac_scale_fsm #(
    // Number of scale units
    parameter int N_SCALE = 4,
    
    // Number of columns handled per scale unit
    parameter int N_COLS  = 8, 
    
    // Number of rows handled per scale unit
    parameter int N_ROWS  = 2, 

    localparam CLOG_NCOLS = $clog2(N_COLS),
    localparam CLOG_NROWS = $clog2(N_ROWS)
) (
    input  logic                 clk_i,
    input  logic                 rst_ni,

    input  logic                 context_ready_i,
    output logic                 context_accept_o,

    output logic                 scale_busy_o,
    output logic                 scale_done_o,
    
    ///////////////////////////////////////////////////////////////////////////
    // Scale unit <-> Scale FSM Signals
    ///////////////////////////////////////////////////////////////////////////

    // Signals from scale_unit -> controller, indicating 
    // processing completion for each token and 
    // the last token respectively
    input  logic [N_SCALE-1:0]              scale_wr_end_tok_i, 

    // Signal from controller -> scale unit, indicates
    // both valid data and whether the next token is 
    // the last one.
    output logic [N_SCALE-1:0]              scale_rd_end_tok_o,    
    output logic [N_SCALE-1:0]              scale_rd_valid_o,      // Data passed into scale unit is valid 

    // Context coordinates (T+1): aligned with the returning BRAM read data,
    // so the snapshot/scale muxes pair with the accumulator of the same cell
    output logic [CLOG_NCOLS-1:0]           scale_ctx_col_o,
    output logic [CLOG_NROWS-1:0]           scale_ctx_row_group_o,

    ///////////////////////////////////////////////////////////////////////////
    // Scale FSM <-> BRAM Signals
    ///////////////////////////////////////////////////////////////////////////
    
    // Read coordinates (pushed to BRAM on cycle T)
    output logic                            bram_rd_en_o, 
    output logic [CLOG_NCOLS-1:0]           bram_rd_col_o,
    output logic [CLOG_NROWS-1:0]           bram_rd_row_group_o
);

    typedef enum logic [1:0] {
        IDLE,
        INIT_RD,    // Prime the memory pipeline with the first read
        STREAM,     // Continuous pipeline: read N+4, compute/write N
        DRAIN       // Flush remaining pipeline writes
    } state_e;

    state_e state_q, state_d;

    // Read Counters
    logic [CLOG_NCOLS-1:0] rd_col_q, rd_col_d;
    logic [CLOG_NROWS-1:0] rd_row_grp_q, rd_row_grp_d;
    
    // Read counter increment helper flag
    logic       rd_last;
    logic       rd_last_q;
    assign rd_last = (rd_col_q == N_COLS-1) && (rd_row_grp_q == N_ROWS-1);

    //--------------------------------------------------------------------------
    // Sequential Pipeline Logic
    //--------------------------------------------------------------------------
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q          <= IDLE;
            rd_col_q         <= '0;
            rd_row_grp_q     <= '0;
        end else begin
            state_q      <= state_d;
            rd_col_q     <= rd_col_d;
            rd_row_grp_q <= rd_row_grp_d;

            /* 
                Valid read signal and read address to scale unit is 
                delayed by 1 cycle from the BRAM signals
            */
            rd_last_q <= rd_last;
            scale_ctx_col_o <= bram_rd_col_o;
            scale_ctx_row_group_o <= bram_rd_row_group_o;
            scale_rd_valid_o <= {N_SCALE{bram_rd_en_o}};
        end
    end

    //--------------------------------------------------------------------------
    // Decoupled Counter Next-State Logic
    //--------------------------------------------------------------------------
    always_comb begin
        state_d      = state_q;
        rd_col_d     = rd_col_q;
        rd_row_grp_d = rd_row_grp_q;

        case (state_q)
            IDLE: begin
                rd_col_d     = '0;
                rd_row_grp_d = '0;
                if (context_ready_i) begin
                    state_d = INIT_RD;
                end
            end

            INIT_RD: begin
                state_d  = STREAM;
                rd_col_d = rd_col_q + 1'b1;
            end

            STREAM: begin
                if (rd_last) begin
                    state_d = DRAIN;
                end else begin
                    if (rd_col_q == 3'd7) begin
                        rd_col_d     = '0;
                        rd_row_grp_d = rd_row_grp_q + 1'b1;
                    end else begin
                        rd_col_d = rd_col_q + 1'b1;
                    end
                end
            end

            DRAIN: begin
                if (scale_wr_end_tok_i[0]) begin
                    state_d = IDLE;
                end
            end

            default: state_d = IDLE;
        endcase
    end

    //--------------------------------------------------------------------------
    // Control Signal Outputs
    //--------------------------------------------------------------------------       
 
    assign context_accept_o = (state_q == IDLE) && context_ready_i;
    assign scale_busy_o     = (state_q != IDLE);

    // End token signal towards scale unit, lags 1 cycle from bram.
    assign scale_rd_end_tok_o =  {N_SCALE{rd_last_q}};

    // Drive write assignments directly from the terminal pipeline stage (T+4)
    assign scale_done_o         = (state_q == DRAIN) && scale_wr_end_tok_i[0];

    // BRAM read data lands 1 cycle after the address, so stage 0 is the
    // coordinate whose accumulator is on the bus right now
    assign bram_rd_col_o = rd_col_q;
    assign bram_rd_row_group_o = rd_row_grp_q;
    assign bram_rd_en_o = (state_q == INIT_RD) || (state_q == STREAM);


//--------------------------------------------------------------------------
//  Assertions 
//--------------------------------------------------------------------------

`ifdef BRAM_DEBUG
always_ff @(posedge clk_i) begin
    case(state_q)  
        DRAIN: begin 

            // Assumption: All N_SCALE scale units are scynchronized, 
            // so only need to sample one of the scale units
            if (scale_wr_end_tok_i[0]) begin 
                // Sanity checking for ther scale units
                assert (scale_wr_end_tok_i == {N_SCALE{1'b1}}) 
                else  $error("Scale units end tokens not synchronized; wr_end_tok_flags: %b\n"
                                , scale_wr_end_tok_i);
            end
        end
    endcase
end
`endif // BRAM_DEBUG


//--------------------------------------------------------------------------
// Simulation Debug Prints
//--------------------------------------------------------------------------
`ifdef BRAM_DEBUG
    always @(posedge clk_i) begin
        if (rst_ni && ((state_q != IDLE) || context_ready_i)) begin
            $display("[MAC_SCALE_FSM @ %0t ps] --------------------------------------------------", $time);
            $display("  STATE    : curr=%s (0x%0h)  -->  next=%s (0x%0h)",
                     state_q.name(), state_q, state_d.name(), state_d);
            $display("  INPUTS   : context_ready=%b | rd_last=%b", 
                     context_ready_i, rd_last);
            $display("  FLAGS    : busy=%b | rd_init_q=%b | write_valid_q=%b | accept=%b | done=%b", 
                     scale_busy_o, rd_init_q, wr_valid_pipe_q[3], context_accept_o, scale_done_o);
            $display("  READ ADDR: rd_en=%b | rd_row_grp_q=%0d (next=%0d) | rd_col_q=%0d (next=%0d)",
                     scale_rd_en_o, rd_row_grp_q, rd_row_grp_d, rd_col_q, rd_col_d);
            $display("  WRIT ADDR: wr_en=%b | wr_row_grp_q=%0d          | wr_col_q=%0d",
                     wr_valid_pipe_q[3], wr_row_pipe_q[3], wr_col_pipe_q[3]);
        end
    end
`endif

endmodule
