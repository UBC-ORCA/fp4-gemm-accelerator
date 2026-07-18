`timescale 1ns/1ps

module mac_scale_fsm #(
    parameter int NUM_GROUPS = 32
) (
    input  logic                 clk_i,
    input  logic                 rst_ni,

    // Context interface from top-level wrapper
    input  logic                 context_ready_i,
    output logic                 context_accept_o,

    // Configuration scales and datapath snapshots
    input  logic [31:0]          act_scale_lo_i,
    input  logic [31:0]          act_scale_hi_i,
    input  logic [31:0]          weight_scale_lo_i,
    input  logic [31:0]          weight_scale_hi_i,
    input  logic signed [15:0]   tile_snapshot_i [0:7][0:7],

    // Handshake status signals back to controller
    output logic                 scale_busy_o,
    output logic                 scale_write_o, // High strictly during BRAM write phase
    output logic                 scale_done_o,

    // Scale datapath indexing outputs
    output logic [2:0]           scale_col_o,
    output logic [1:0]           scale_row_group_o
);

    //--------------------------------------------------------------------------
    // FSM States & Internals
    //--------------------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE,
        READ,
        WRITE,
        DONE
    } state_e;

    state_e state_q, state_d;
    
    localparam int CNT_W = $clog2(NUM_GROUPS);
    logic [CNT_W-1:0] count_q, count_d;

    // Self-contained internal context storage registers
    logic [31:0]          act_scale_lo_q;
    logic [31:0]          act_scale_hi_q;
    logic [31:0]          weight_scale_lo_q;
    logic [31:0]          weight_scale_hi_q;
    logic signed [15:0]   tile_snapshot_q [0:7][0:7];

    //--------------------------------------------------------------------------
    // State & Context Register Latching Logic
    //--------------------------------------------------------------------------
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q           <= IDLE;
            count_q           <= '0;
            act_scale_lo_q    <= '0;
            act_scale_hi_q    <= '0;
            weight_scale_lo_q <= '0;
            weight_scale_hi_q <= '0;

            for (int r = 0; r < 8; r++) begin
                for (int c = 0; c < 8; c++) begin
                    tile_snapshot_q[r][c] <= '0;
                end
            end
        end else begin
            state_q <= state_d;
            count_q <= count_d;

            // Latch execution context exactly on the cycle the handshake matches
            if ((state_q == IDLE) && context_ready_i) begin
                act_scale_lo_q    <= act_scale_lo_i;
                act_scale_hi_q    <= act_scale_hi_i;
                weight_scale_lo_q <= weight_scale_lo_i;
                weight_scale_hi_q <= weight_scale_hi_i;
                tile_snapshot_q   <= tile_snapshot_i;
            end
        end
    end

    //--------------------------------------------------------------------------
    // Next-State & Counter Combinational Evaluation
    //--------------------------------------------------------------------------
    always_comb begin
        state_d = state_q;
        count_d = count_q;

        case (state_q)
            IDLE: begin
                count_d = '0;
                if (context_ready_i) begin
                    state_d = READ;
                end
            end

            READ: begin
                // Synchronous BRAM Read step complete, advance to write calculation phase
                state_d = WRITE;
            end

            WRITE: begin
                // Commit write element and evaluate sequence bounds
                if (count_q == (NUM_GROUPS[CNT_W-1:0] - 1'b1)) begin
                    state_d = DONE;
                end else begin
                    count_d = count_q + 1'b1;
                    state_d = READ;
                end
            end

            DONE: begin
                state_d = IDLE;
            end

            default: begin
                state_d = IDLE;
            end
        endcase
    end

    //--------------------------------------------------------------------------
    // Glitch-Free Combinational Output Generation
    //--------------------------------------------------------------------------
    assign context_accept_o = (state_q == IDLE) && context_ready_i;
    assign scale_busy_o      = (state_q == READ) || (state_q == WRITE);
    assign scale_write_o     = (state_q == WRITE);
    assign scale_done_o      = (state_q == DONE);

    // Drive coordinates combinational-direct from count tracking to match across both cycles
    assign scale_col_o       = count_q[2:0];
    assign scale_row_group_o = count_q[4:3];

endmodule
