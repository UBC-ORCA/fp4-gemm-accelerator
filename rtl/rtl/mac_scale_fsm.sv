`timescale 1ns/1ps

module mac_scale_fsm #(
    parameter int NUM_GROUPS = 32
) (
    input  logic                 clk_i,
    input  logic                 rst_ni,

    input  logic                 context_ready_i,
    output logic                 context_accept_o,

    input  logic [31:0]          act_scale_lo_i,
    input  logic [31:0]          act_scale_hi_i,
    input  logic [31:0]          weight_scale_lo_i,
    input  logic [31:0]          weight_scale_hi_i,
    input  logic signed [15:0]   tile_snapshot_i [0:7][0:7],

    output logic                 scale_busy_o,
    output logic                 scale_rd_en_o,
    output logic                 scale_write_o,
    output logic                 scale_done_o,

    // Read coordinates (pushed to BRAM & Scale Muxes)
    output logic [2:0]           scale_rd_col_o,
    output logic [1:0]           scale_rd_row_group_o,

    // Write coordinates (pushed to BRAM on cycle T+1)
    output logic [2:0]           scale_wr_col_o,
    output logic [1:0]           scale_wr_row_group_o
);

    typedef enum logic [1:0] {
        IDLE,
        INIT_RD,    // Prime the memory pipeline with the first read
        STREAM,     // Continuous pipeline: read N+1, compute/write N
        DRAIN       // Flush remaining pipeline write N_last
    } state_e;

    state_e state_q, state_d;

    // Read Counters
    logic [2:0] rd_col_q, rd_col_d;
    logic [1:0] rd_row_grp_q, rd_row_grp_d;

    // Pipeline Write Tracking Registers (Delay 1 cycle behind Read)
    logic [2:0] wr_col_q;
    logic [1:0] wr_row_grp_q;
    logic       write_valid_q;

    // Flag indicating read pipeline priming status
    logic       rd_init_q;

    // Read counter increment helper flag
    logic       rd_last;
    assign rd_last = (rd_col_q == 3'd7) && (rd_row_grp_q == 2'd3);

    //--------------------------------------------------------------------------
    // Sequential Pipeline Logic
    //--------------------------------------------------------------------------
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q       <= IDLE;
            rd_col_q      <= '0;
            rd_row_grp_q  <= '0;
            wr_col_q      <= '0;
            wr_row_grp_q  <= '0;
            write_valid_q <= 1'b0;
            rd_init_q     <= 1'b0;
        end else begin
            state_q      <= state_d;
            rd_col_q     <= rd_col_d;
            rd_row_grp_q <= rd_row_grp_d;

            // Write pipeline delay line (captures read address on active read)
            if (state_q == INIT_RD || state_q == STREAM) begin
                wr_col_q     <= rd_col_q;
                wr_row_grp_q <= rd_row_grp_q;
            end

            // Write enable valid 1 cycle after read starts
            write_valid_q <= (state_q == STREAM) || (state_q == DRAIN);

            // Prime status tracking
            if (state_q == INIT_RD) begin
                rd_init_q <= 1'b1;
            end else if (state_q == IDLE || state_q == DRAIN) begin
                rd_init_q <= 1'b0;
            end
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
                // Kick off 1st read at (0,0), step read counter on next cycle
                //rd_col_d = rd_col_q + 1'b1;
                state_d  = STREAM;
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
                // Drain last element write, then return to IDLE
                state_d = IDLE;
            end

            default: state_d = IDLE;
        endcase
    end

    //--------------------------------------------------------------------------
    // Control Signal Outputs
    //--------------------------------------------------------------------------
    assign context_accept_o = (state_q == IDLE) && context_ready_i;
    assign scale_busy_o     = (state_q != IDLE);
    assign scale_rd_en_o    = (state_q == INIT_RD) || (state_q == STREAM);
    assign scale_write_o    = write_valid_q;
    assign scale_done_o     = (state_q == DRAIN);

    // Export decoupled read and write indices
    assign scale_rd_col_o       = rd_col_q;
    assign scale_rd_row_group_o = rd_row_grp_q;

    assign scale_wr_col_o       = wr_col_q;
    assign scale_wr_row_group_o = wr_row_grp_q;

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
                     scale_busy_o, rd_init_q, write_valid_q, context_accept_o, scale_done_o);
            $display("  READ ADDR: rd_en=%b | rd_row_grp_q=%0d (next=%0d) | rd_col_q=%0d (next=%0d)",
                     scale_rd_en_o, rd_row_grp_q, rd_row_grp_d, rd_col_q, rd_col_d);
            $display("  WRIT ADDR: wr_en=%b | wr_row_grp_q=%0d          | wr_col_q=%0d",
                     scale_write_o, wr_row_grp_q, wr_col_q);
        end
    end
`endif

endmodule
