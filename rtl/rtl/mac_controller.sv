module mac_controller #(
    parameter int VL = 8 // Vector Length configuration matches wrapper TT parameter
) (
    input  logic                 clk_i,
    input  logic                 rst_ni,

    // Request interface
    input  logic                 req_valid_i,
    output logic                 req_ready_o,
    input  cve2_pkg::mac_op_e    cf_req_op_i,
    input  logic [31:0]          rs1_i,
    input  logic [31:0]          rs2_i,

    // Status outputs
    output logic                 busy_o,
    output logic                 done_o,

    // Control to MAC array
    output logic                 mac_en_o,
    output logic                 clear_o,

    // --- [stev] ---
    input  logic [4:0]           vs1_i,
    input  logic [4:0]           weight_blk_i,
    input  logic [31:0]          base_i,

    output logic [4:0]           mac_vrf_raddr_o,
    output logic [2:0]           mac_vrf_relem_o,
    input  logic [31:0]          mac_vrf_rdata_i, // Added VRF execution feedback payload

    // Weight memory interface
    output logic                 data_req_o,
    input  logic                 data_gnt_i,

    output logic [31:0]          data_addr_o,
    output logic                 data_we_o,
    output logic [3:0]           data_be_o,
    output logic [31:0]          data_wdata_o,

    input  logic                 data_rvalid_i,
    input  logic [31:0]          data_rdata_i,
    input  logic                 data_err_i,

    // mv inst
	output logic        mv_en_o,
	output logic [1:0]  mv_mode_o,   // 0=even 1=odd 2=pair
	output logic [2:0] mv_even_col_idx_o,
	output logic [2:0] mv_odd_col_idx_o,
	output logic [2:0] mv_row_idx_o,
  	input  logic [4:0]  mv_row_i,       // instruction rs1 field
  	input  logic [4:0]  mv_pair_i,      // instruction rs2 field

    output logic                     scalar_we_o,
    output logic [4:0]               scalar_waddr_o,
    input logic [4:0]              scalar_waddr_i,


    // --- [end] ---

    // Optimized Interface: Directly outputs clean, sliced vector structures
    output logic [3:0]           act_vector_o    [0:VL-1],
    output logic [3:0]           weight_vector_o [0:VL-1]
);

    //----------------------------------------------------------
    // Registers & Internal Signals
    //----------------------------------------------------------
    logic [4:0]  vs1_q;
    logic [4:0]  weight_blk_q;
    logic [31:0] base_q;

    // MEM Request Bookkeeping
    logic        mem_req_sent_q;
    logic        mem_req_sent_d;

    // Latched instruction tracking
    cve2_pkg::mac_op_e op_q;

    // MAC cycle counter
    localparam int CNT_W = $clog2(VL);
    logic [CNT_W-1:0] count_q;
    logic [CNT_W-1:0] count_d;

    // FSM States
    typedef enum logic [1:0] {
        IDLE,
        EXEC,
        DONE
    } state_e;

    state_e state_q, state_d;

// hwmac
logic [31:0] act_packed;
logic [31:0] weight_packed;

always_comb begin
    act_packed    = mac_vrf_rdata_i;
    weight_packed = data_rdata_i;

    if (op_q == cve2_pkg::OP_MAC) begin
        act_packed    = rs1_i;
        weight_packed = rs2_i;
    end
end
// end

// mv

/**************************************************************************
 * Move instruction decode
 **************************************************************************/

localparam logic [1:0] MV_EVEN = 2'd0;
localparam logic [1:0] MV_ODD  = 2'd1;
localparam logic [1:0] MV_PAIR = 2'd2;

logic [1:0] mv_pair_idx;

assign mv_row_idx_o      = mv_row_i[2:0];
assign mv_pair_idx     = mv_pair_i[1:0];
assign mv_even_col_idx_o = {mv_pair_idx,1'b0};   // 2*pair
assign mv_odd_col_idx_o  = {mv_pair_idx,1'b1};   // 2*pair+1

logic [4:0] scalar_waddr_q;
// end

    //----------------------------------------------------------
    // Sequential Logic (Latches & State)
    //----------------------------------------------------------
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            op_q           <= cve2_pkg::OP_ZZ;
            vs1_q          <= '0;
            weight_blk_q   <= '0; 
            base_q         <= '0; 
            state_q        <= IDLE;
            count_q        <= '0;
            mem_req_sent_q <= 1'b0;
	    scalar_waddr_q <= '0;
        end else begin
            state_q        <= state_d;
            count_q        <= count_d;
            mem_req_sent_q <= mem_req_sent_d;

            if (req_valid_i && req_ready_o) begin
                op_q         <= cf_req_op_i;
                vs1_q        <= vs1_i;
                weight_blk_q <= weight_blk_i;
                base_q       <= base_i;
		scalar_waddr_q <= scalar_waddr_i;
            end
        end
    end

    //----------------------------------------------------------
    // Next-State Logic
    //----------------------------------------------------------
    always_comb begin
        state_d        = state_q;
        count_d        = count_q;
        mem_req_sent_d = mem_req_sent_q;

        case (state_q)
            IDLE: begin
                count_d        = '0;
                mem_req_sent_d = 1'b0;
                if (req_valid_i) begin
                    state_d = EXEC;
                end
            end

           EXEC: begin
	     if ((op_q == cve2_pkg::OP_ZZ ) ||
    		(op_q == cve2_pkg::OP_MAC) ||
    		(op_q == cve2_pkg::OP_MVE) ||
    		(op_q == cve2_pkg::OP_MVO) ||
    		(op_q == cve2_pkg::OP_MV2))
	     begin
        		// one-cycle operation
        		state_d = DONE;
    	     end

	     else if (op_q == cve2_pkg::OP_VMAC) begin
                if (!mem_req_sent_q) begin
                    // Phase 1: Waiting for the LSU to accept the address request
                    if (data_gnt_i) begin
                        mem_req_sent_d = 1'b1;
                    end
                end else begin
                    // Phase 2: Request accepted, waiting for valid data from memory bus
                    if (data_rvalid_i) begin
                        mem_req_sent_d = 1'b0; // Reset tracking flag for next element loop

                        if (count_q == (VL-1)) begin
                            state_d = DONE;
                            count_d = '0;
                        end else begin
                            count_d = count_q + 1'b1;
                        end
                    end
                end
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

    //----------------------------------------------------------
    // Combinational Control & Address Mapping
    //----------------------------------------------------------
    always_comb begin
        // Default Control Path Signals
        req_ready_o  = 1'b0;
        busy_o       = 1'b1;
        done_o       = 1'b0;
        clear_o      = 1'b0;

	mv_en_o      = 1'b0;
	mv_mode_o    = MV_EVEN;

	scalar_we_o    = 1'b0;
	scalar_waddr_o = scalar_waddr_q;

        // Strict Hardware Gating: Prevent structural firing on stray or out-of-context bus pulses
        mac_en_o = ((state_q == EXEC) && (op_q == cve2_pkg::OP_VMAC) && data_rvalid_i) || // VMAC fires only when memory returns a word
 	((state_q == EXEC) && (op_q == cve2_pkg::OP_MAC));

	// ZZMAC64 clears the whole tile for one cycle
	clear_o = (state_q == EXEC) && (op_q == cve2_pkg::OP_ZZ);

        // VRF Control Defaults
        mac_vrf_raddr_o = '0;
        mac_vrf_relem_o = '0;

        // Memory Interface Defaults
        data_req_o   = 1'b0;
        data_addr_o  = '0;
        data_we_o    = 1'b0;
        data_be_o    = 4'b1111; 
        data_wdata_o = '0;

        case (state_q)
            IDLE: begin
                req_ready_o = 1'b1;
                busy_o      = 1'b0;
            end

            EXEC: begin

    		//--------------------------------------------------
    		// Move instructions (one-cycle)
    		//--------------------------------------------------
    		unique case (op_q)
        		cve2_pkg::OP_MVE: begin
            		mv_en_o   = 1'b1;
            		mv_mode_o = MV_EVEN;
			scalar_we_o  = 1'b1;
        	end

        	cve2_pkg::OP_MVO: begin
            		mv_en_o   = 1'b1;
            		mv_mode_o = MV_ODD;
			scalar_we_o  = 1'b1;
        	end

        	cve2_pkg::OP_MV2: begin
            		mv_en_o   = 1'b1;
            		mv_mode_o = MV_PAIR;
			scalar_we_o  = 1'b1;
        	end

        	default: ;
		endcase

		//--------------------------------------------------
    		// VMAC
    		//--------------------------------------------------
                if (op_q == cve2_pkg::OP_VMAC) begin
                    // Continuous registration addressing 
                    mac_vrf_raddr_o = vs1_q;
                    mac_vrf_relem_o = count_q[2:0];

                    // Hold memory request until high acknowledgement handshaking occurs
                    if (!mem_req_sent_q) begin
                        data_req_o  = 1'b1;
                        data_addr_o = base_q + (count_q << 2); // Wrapped calculation logic matches base pipeline
                    end
                end
            end

            DONE: begin
                done_o = 1'b1;
            end

            default: ;
        endcase
    end

    //----------------------------------------------------------
    // Combinational Combinatorial Unpack Loops
    //----------------------------------------------------------
    genvar k;
    generate
        for (k = 0; k < VL; k++) begin : GEN_UNPACK_NIBBLES
            assign act_vector_o[k]    = act_packed[4*k +:4];
            assign weight_vector_o[k] = weight_packed[4*k +:4];
        end
    endgenerate

    //----------------------------------------------------------
    // Debug Trace Tasks
    //----------------------------------------------------------
`ifdef VEC_DEBUG
    always_ff @(posedge clk_i) begin
        if (rst_ni && data_rvalid_i && (state_q == EXEC)) begin
            $display("[MAC_LOAD_RETURN] addr=%08x data=%08x count=%0d",
                     data_addr_o, data_rdata_i, count_q);
        end
    end
`endif

endmodule
