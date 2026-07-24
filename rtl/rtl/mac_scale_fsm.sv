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
    output logic                 scale_write_o,
    output logic                 scale_done_o,

    output logic [2:0]           scale_col_o,
    output logic [1:0]           scale_row_group_o
);

    typedef enum logic [1:0] {
        IDLE,
        STREAM,
        DRAIN
    } state_e;

    state_e state_q, state_d;

    // Independent structural read counters
    logic [2:0] rd_col_q, rd_col_d;
    logic [1:0] rd_row_grp_q, rd_row_grp_d;

    // Registered pipeline write flag
    logic write_valid_q;

    //--------------------------------------------------------------------------
    // State & Counter Register Logic
    //--------------------------------------------------------------------------
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q       <= IDLE;
            rd_col_q      <= '0;
            rd_row_grp_q  <= '0;
            write_valid_q <= 1'b0;
        end else begin
            state_q       <= state_d;
            rd_col_q      <= rd_col_d;
            rd_row_grp_q  <= rd_row_grp_d;
            write_valid_q <= (state_q == STREAM);
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
                    state_d = STREAM;
                end
            end

            STREAM: begin
                // Independent column step
                if (rd_col_q == 3'd7) begin
                    rd_col_d = '0;
                    // Independent row-group step
                    if (rd_row_grp_q == 2'd3) begin
                        state_d = DRAIN;
                    end else begin
                        rd_row_grp_d = rd_row_grp_q + 1'b1;
                    end
                end else begin
                    rd_col_d = rd_col_q + 1'b1;
                end
            end

            DRAIN: begin
                state_d = IDLE;
            end

            default: state_d = IDLE;
        endcase
    end

    //--------------------------------------------------------------------------
    // Output Assignments
    //--------------------------------------------------------------------------
    assign context_accept_o = (state_q == IDLE) && context_ready_i;
    assign scale_busy_o      = (state_q != IDLE) || write_valid_q;
    assign scale_write_o     = write_valid_q;
    assign scale_done_o      = (state_q == DRAIN);

    // Dynamic output coordinates strictly controlled by structural counters
    assign scale_col_o       = rd_col_q;
    assign scale_row_group_o = rd_row_grp_q;

endmodule
