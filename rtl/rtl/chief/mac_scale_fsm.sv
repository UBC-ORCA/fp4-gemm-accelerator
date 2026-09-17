`timescale 1ns/1ps

module mac_scale_fsm #(
    parameter int NUM_GROUPS = 32
) (
    input  logic                 clk_i,
    input  logic                 rst_ni,

    input  logic                 context_ready_i,
    output logic                 context_accept_o,

    output logic                 scale_busy_o,
    output logic                 scale_rd_en_o,
    output logic                 scale_write_o,
    output logic                 scale_done_o,

    // Read coordinates (pushed to BRAM on cycle T)
    output logic [2:0]           scale_rd_col_o,
    output logic [1:0]           scale_rd_row_group_o,

    // Context coordinates (T+1): aligned with the returning BRAM read data,
    // so the snapshot/scale muxes pair with the accumulator of the same cell
    output logic [2:0]           scale_ctx_col_o,
    output logic [1:0]           scale_ctx_row_group_o,

    // Write coordinates (pushed to BRAM on cycle T+4)
    output logic [2:0]           scale_wr_col_o,
    output logic [1:0]           scale_wr_row_group_o
);

    typedef enum logic [1:0] {
        IDLE,
        INIT_RD,    // Prime the memory pipeline with the first read
        STREAM,     // Continuous pipeline: read N+4, compute/write N
        DRAIN       // Flush remaining pipeline writes
    } state_e;

    state_e state_q, state_d;

    // Read Counters
    logic [2:0] rd_col_q, rd_col_d;
    logic [1:0] rd_row_grp_q, rd_row_grp_d;

    // 4-Stage Pipeline Shift Registers (1 cycle BRAM read + 3 cycles Accumulator)
    logic [2:0] wr_col_pipe_q  [0:3];
    logic [1:0] wr_row_pipe_q  [0:3];
    logic [3:0] wr_valid_pipe_q;

    // Drain cycle countdown tracking
    logic [2:0] drain_cnt_q;
    logic       rd_init_q;

    // Read counter increment helper flag
    logic       rd_last;
    assign rd_last = (rd_col_q == 3'd7) && (rd_row_grp_q == 2'd3);

    //--------------------------------------------------------------------------
    // Sequential Pipeline Logic
    //--------------------------------------------------------------------------
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q          <= IDLE;
            rd_col_q         <= '0;
            rd_row_grp_q     <= '0;
            wr_valid_pipe_q  <= '0;
            drain_cnt_q      <= '0;
            rd_init_q        <= 1'b0;
            for (int i = 0; i < 4; i++) begin
                wr_col_pipe_q[i] <= '0;
                wr_row_pipe_q[i] <= '0;
            end
        end else begin
            state_q      <= state_d;
            rd_col_q     <= rd_col_d;
            rd_row_grp_q <= rd_row_grp_d;

            // Pipeline Stage 0: Sample current read info on active read cycle
            wr_col_pipe_q[0]  <= rd_col_q;
            wr_row_pipe_q[0]  <= rd_row_grp_q;
            wr_valid_pipe_q[0]<= (state_q == INIT_RD || state_q == STREAM);

            // Pipeline Stages 1-3: Shift down every clock cycle
            for (int i = 1; i < 4; i++) begin
                wr_col_pipe_q[i]   <= wr_col_pipe_q[i-1];
                wr_row_pipe_q[i]   <= wr_row_pipe_q[i-1];
                wr_valid_pipe_q[i] <= wr_valid_pipe_q[i-1];
            end

            // Drain tracking logic: hold DRAIN for exactly 4 cycles to empty pipeline
            if (state_q == STREAM && rd_last) begin
                drain_cnt_q <= 3'd4;
            end else if (state_q == DRAIN && drain_cnt_q != 0) begin
                drain_cnt_q <= drain_cnt_q - 1'b1;
            end

            // Prime status tracking
            if (state_q == INIT_RD) begin
                rd_init_q <= 1'b1;
            // Fix: Reverted safely back to exact clean syntax
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
                if (drain_cnt_q == 3'd0) begin
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
    assign scale_rd_en_o    = (state_q == INIT_RD) || (state_q == STREAM);
    
    // Drive write assignments directly from the terminal pipeline stage (T+4)
    assign scale_write_o        = wr_valid_pipe_q[3];
    assign scale_done_o         = (state_q == DRAIN) && (drain_cnt_q == 3'd0);

    // Export decoupled read and write indices
    assign scale_rd_col_o       = rd_col_q;
    assign scale_rd_row_group_o = rd_row_grp_q;

    // BRAM read data lands 1 cycle after the address, so stage 0 is the
    // coordinate whose accumulator is on the bus right now
    assign scale_ctx_col_o       = wr_col_pipe_q[0];
    assign scale_ctx_row_group_o = wr_row_pipe_q[0];

    assign scale_wr_col_o       = wr_col_pipe_q[3];
    assign scale_wr_row_group_o = wr_row_pipe_q[3];

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
