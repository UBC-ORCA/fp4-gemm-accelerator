// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Optimized Single-Cycle Fast Accumulator Tile Block RAM Module.
// Features flexible dynamic indexing functions and forward bypassing
// to support zero-stall read-modify-write accumulation loops.

`timescale 1ns/1ps

module mac_accum_bram #(
    parameter int unsigned NTILES = 32,
    parameter int unsigned NROWS  = 8,
    parameter int unsigned NCOLS  = 8,
    parameter int unsigned NBANKS  = 4,   
    parameter int unsigned DATA_W = 16, 
    
    // Derrived Params 
    localparam int unsigned DEPTH  = NTILES * (NROWS / NBANKS) * NCOLS, // 1024 words
    localparam int unsigned TILE_W    = $clog2(NTILES),
    localparam int unsigned ROW_WIDTH = $clog2(NROWS),
    localparam int unsigned COL_WIDTH = $clog2(NCOLS),
    localparam int unsigned ADDR_W = $clog2(DEPTH), // 10 bits
    localparam int unsigned BANK_ADDR_W = $clog2(NBANKS)
) (
    input  logic        clk_i,
    input  logic        rst_ni,

    //----------------------------------
    // Read Port (Tile Structural Coordinates)
    //----------------------------------
    input  logic                                rd_en_i,
    input  logic [TILE_W-1:0]                   rd_tile_i,
    input  logic [ROW_WIDTH-1:0]                rd_row_i,
    input  logic [COL_WIDTH-1:0]                rd_col_i,
    output logic [NBANKS-1:0][DATA_W-1:0]       rd_data_o, 

    //----------------------------------
    // Write Port (Tile Structural Coordinates)
    //----------------------------------
    input  logic                                wr_en_i,
    input  logic [TILE_W-1:0]                   wr_tile_i,
    input  logic [ROW_WIDTH-1:0]                wr_row_i,
    input  logic [COL_WIDTH-1:0]                wr_col_i,
    input  logic [NBANKS-1:0][DATA_W-1:0]       wr_data_i, 
   
    // One-Hot Write Select  
    input logic [NBANKS-1:0]               bank_sel 
);

    /* Elaboration Time Check for parameters */
    generate 
        if (NROWS < NBANKS) begin 
            $error("Number of rows cannot be smaller than the number of banks!\n");
        end
    endgenerate

    // Split Memory Banks (Maps directly to Xilinx Block RAM Primitives)
    // (* ram_style = "block" *) logic [15:0] bram_low  [0:DEPTH-1];
    // (* ram_style = "block" *) logic [15:0] bram_high [0:DEPTH-1];

    //--------------------------------------------------------------------------
    // Dynamic / Changeable Indexing Rules
    // Edit this function to modify physical coordinate mapping dynamically.
    //--------------------------------------------------------------------------
    function automatic logic [ADDR_W-1:0] calc_addr(
        input logic [TILE_W-1:0] tile,
        input logic [ROW_WIDTH-1:0] row,
        input logic [COL_WIDTH-1:0] col
    );
        // 
        //  Address Format 
        //  ----------------------------------------------------------
        //  |   TILE_WIDTH   | ROW_WIDTH - BANK_WIDTH |  COL_WIDTH   |  
        //  ----------------------------------------------------------
        return {tile, row[ROW_WIDTH-1:BANK_ADDR_W], col};
    endfunction

    // Calculated addresses
    logic [ADDR_W-1:0] rd_addr;
    logic [ADDR_W-1:0] wr_addr;

    assign rd_addr = calc_addr(rd_tile_i, rd_row_i, rd_col_i);
    assign wr_addr = calc_addr(wr_tile_i, wr_row_i, wr_col_i);

    // Data read from RAM directly, 
    logic [NBANKS-1:0][DATA_W-1:0] ram_read_q;

    // Debug Only BRAM Debug signals
`ifdef BRAM_DEBUG   
    logic [DATA_W-1:0] bram_dbg_d [0:NBANKS-1][0:DEPTH-1];    
`endif

    // Generation of RAM Banks for parallel access
    generate
        for (genvar b = 0; b < NBANKS; ++b) begin : gen_bram_banks
            (* ram_style = "block" *) logic [DATA_W-1:0] bram_bank_b [0:DEPTH-1];

            always_ff @(posedge clk_i) begin 
                if (wr_en_i & bank_sel[b]) begin 
                    bram_bank_b[wr_addr] <= wr_data_i[b];        
                end

                if (rd_en_i) begin 
                    ram_read_q[b] <= bram_bank_b[rd_addr];
                end
            end 

            `ifdef BRAM_DEBUG
                assign bram_dbg_d[b] = bram_bank_b;
            `endif
        end : gen_bram_banks
    endgenerate


    // Direct write-forwarding registers to eliminate write-to-read latency stalls
    logic [ADDR_W-1:0]              wr_addr_q;
    logic [NBANKS-1:0][DATA_W-1:0]  wr_data_q;
    logic [NBANKS-1:0]              bank_sel_q;
    logic              wr_en_q;
    // logic              wr_pair_q;
    logic [2:0]        wr_row_q;

    // Tracking read address for forwarding hazard checks
    logic rd_en_q;
    logic [ADDR_W-1:0] rd_addr_q;

    //--------------------------------------------------------------------------
    // Pipeline Forwarding Tracking Logic (For Single-Cycle Accumulation Loops)
    //--------------------------------------------------------------------------
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            wr_addr_q <= '0;
            wr_data_q <= '0;
            wr_en_q   <= 1'b0;
            wr_row_q  <= '0;
            rd_addr_q <= '0;
            rd_en_q <= '0;
        end else begin
            wr_addr_q <= wr_addr;
            wr_data_q <= wr_data_i;
            wr_en_q   <= wr_en_i;
            rd_en_q   <= rd_en_i;
            wr_row_q  <= wr_row_i;
            bank_sel_q <= bank_sel;
            if (rd_en_i) begin
                rd_addr_q <= rd_addr;
            end
        end
    end

    //--------------------------------------------------------------------------
    // Fast Output Muxing with Zero-Latency Forwarding Bypass
    //--------------------------------------------------------------------------
    logic [NBANKS-1:0][DATA_W-1:0] final_rdata;     
    always_comb begin
        /* 
            Transaction was both a write and a read, 
            the read would get the latest write instead of getting 
            the value stored during the write request.
         */
        if (wr_en_q && rd_en_q && (rd_addr_q == wr_addr_q)) begin  
            
            /* Only bypass selected banks */
            for (int unsigned b = 0; b < NBANKS; ++b) begin  
                final_rdata[b] = bank_sel_q[b] ? wr_data_q[b] : ram_read_q[b];
            end
        end else begin
            final_rdata = ram_read_q;                
        end
    end

    assign rd_data_o = final_rdata;
    
    // FIXME: Fix the debugging blocks

    //--------------------------------------------------------------------------
    // Assertions
    //--------------------------------------------------------------------------
// `ifndef SYNTHESIS
//     always_ff @(posedge clk_i) begin
//         if (rd_en_i) begin
//             assert(rd_row_i[0] == 1'b0) else
//                 $error("[BRAM_ACCUM_ERROR] Read row must be even for paired layout, got %0d", rd_row_i);
//         end
//         if (wr_en_i && wr_pair_i) begin
//             assert(wr_row_i[0] == 1'b0) else
//                 $error("[BRAM_ACCUM_ERROR] Paired write row must be even, got %0d", wr_row_i);
//         end
//     end
// `endif

    //--------------------------------------------------------------------------
    // Simulation Visual Debug Display
    //--------------------------------------------------------------------------
`ifdef BRAM_DEBUG
    begin 
        integer r,c, dbg_addr;
        always_ff @(posedge clk_i) begin
            if (rst_ni && wr_en_i) begin
                $display("\n======================================================");
                $display("ACCUMULATOR BRAM TILE %0d SNAPSHOT @ time %0t", wr_tile_i, $time);
                for (r = 0; r < NROWS; r++) begin
                    $write("Row %0d :", r);
                    for (c = 0; c < NCOLS; c++) begin
                        dbg_addr = calc_addr(wr_tile_i[TILE_W-1:0],
                                             r[2:0],
                                             c[2:0]);
                        $write(" %6h ", bram_dbg_d[r[BANK_ADDR_W-1:0]][dbg_addr]);
                    end
                    $write("\n");
                end
                $display("======================================================\n");
            end
        end
    end
`endif

//--------------------------------------------------------------------------
// Simulation Visual Debug Display
//--------------------------------------------------------------------------
`ifdef BRAM_DEBUG
    // Register read address for logging debug output 1 cycle later
    logic [2:0] rd_row_q, rd_col_q;
    logic [4:0] rd_tile_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            rd_tile_q <= '0;
            rd_row_q  <= '0;
            rd_col_q  <= '0;
        end else begin
            rd_tile_q <= rd_tile_i;
            rd_row_q  <= rd_row_i;
            rd_col_q  <= rd_col_i;
        end
    end

    // Transaction Logger
     // Transaction Logger
    always_ff @(posedge clk_i) begin
        if (rst_ni) begin
            if (rd_en_q) begin
                automatic logic [ROW_WIDTH-1:0] rd_grp_base =
                    {rd_row_q[ROW_WIDTH-1:BANK_ADDR_W], {BANK_ADDR_W{1'b0}}};

                $display("[BRAM_ACCUM_DEBUG] [%0t ns] MEMORY READ COMPLETE:", $time);
                $display("[BRAM_ACCUM_DEBUG]    Coordinates -> Tile=%2d | Rows=[%0d,%0d] | Col=%1d",
                          rd_tile_q, rd_grp_base, rd_grp_base + (NBANKS-1), rd_col_q);
                $display("[BRAM_ACCUM_DEBUG]    Payload     -> Out Data=64'h%8h", rd_data_o);
            end

            if (wr_en_i) begin
                automatic logic [ROW_WIDTH-1:0] wr_grp_base =
                    {wr_row_i[ROW_WIDTH-1:BANK_ADDR_W], {BANK_ADDR_W{1'b0}}};

                $display("[BRAM_ACCUM_DEBUG] [%0t ns] MEMORY WRITE TRANSACTION COMMITTED:", $time);
                $display("[BRAM_ACCUM_DEBUG]    Coordinates -> Tile=%2d | Rows=[%0d,%0d] | Col=%1d",
                          wr_tile_i, wr_grp_base, wr_grp_base + (NBANKS-1), wr_col_i);
                $display("[BRAM_ACCUM_DEBUG]    Payload     -> In Data=0x%h | Bank Sel=%0b", wr_data_i, bank_sel);
            end
        end
    end

    // Visual Matrix Snapshot Generator for Tiles 0, 1, and 31
    localparam int unsigned NUM_DBG_TILES = 3;
    localparam int unsigned DBG_TILE_IDX [NUM_DBG_TILES] = '{0, 1, 31};

    always_ff @(posedge clk_i) begin
        if (rst_ni && wr_en_i) begin
            for (int unsigned t = 0; t < NUM_DBG_TILES; ++t) begin
                automatic int unsigned dbg_tile_idx = DBG_TILE_IDX[t];

                if (dbg_tile_idx < NTILES) begin
                    automatic logic [TILE_W-1:0] dbg_tile = TILE_W'(dbg_tile_idx);

                    $display("\n======================================================");
                    $display("ACCUMULATOR BRAM TILE %0d SNAPSHOT @ time %0t", dbg_tile, $time);
                    for (int unsigned r = 0; r < NROWS; r++) begin
                        automatic logic [ROW_WIDTH-1:0]   dbg_row  = ROW_WIDTH'(r);
                        automatic logic [BANK_ADDR_W-1:0] dbg_bank = dbg_row[BANK_ADDR_W-1:0];
                        $write("Row %0d (bank%0d):", r, dbg_bank);
                        for (int unsigned c = 0; c < NCOLS; c++) begin
                            automatic logic [COL_WIDTH-1:0] dbg_col  = COL_WIDTH'(c);
                            automatic logic [ADDR_W-1:0]    dbg_addr = calc_addr(dbg_tile, dbg_row, dbg_col);
                            $write(" %6h", bram_dbg_d[dbg_bank][dbg_addr]);
                        end
                        $write("\n");
                    end
                    $display("======================================================\n");
                end
            end
        end
    end
`endif
endmodule
