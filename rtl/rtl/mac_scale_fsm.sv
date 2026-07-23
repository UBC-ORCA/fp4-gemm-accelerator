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
        RUN,
        DONE
    } state_e;

    state_e state_q, state_d;

    localparam int CNT_W = $clog2(NUM_GROUPS);
    logic [CNT_W-1:0] count_q, count_d;
    logic             write_valid_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q       <= IDLE;
            count_q       <= '0;
            write_valid_q <= 1'b0;
        end else begin
            state_q       <= state_d;
            count_q       <= count_d;
            write_valid_q <= (state_q == RUN);
        end
    end

    always_comb begin
        state_d = state_q;
        count_d = count_q;

        case (state_q)
            IDLE: begin
                count_d = '0;
                if (context_ready_i) begin
                    state_d = RUN;
                end
            end

            RUN: begin
                if (count_q == (NUM_GROUPS[CNT_W-1:0] - 1'b1)) begin
                    state_d = DONE;
                end else begin
                    count_d = count_q + 1'b1;
                end
            end

            DONE: begin
                state_d = IDLE;
            end

            default: state_d = IDLE;
        endcase
    end

    assign context_accept_o = (state_q == IDLE) && context_ready_i;
    assign scale_busy_o      = (state_q == RUN) || write_valid_q;
    assign scale_write_o     = write_valid_q;
    assign scale_done_o      = (state_q == DONE);

    // Dynamic indexing outputs directly tied to active step counter
    assign scale_col_o       = count_q[2:0];
    assign scale_row_group_o = count_q[4:3];

endmodule
