// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Structured coordinate-addressed Accumulator Tile BRAM module.
// Decoupled from physical addressing layers to preserve ISA abstractions.

module mac_accum_bram (
    input  logic        clk_i,
    input  logic        rst_ni,

    //----------------------------------
    // Read Port (Tile Structural Coordinates)
    //----------------------------------
    input  logic        rd_en_i,
    input  logic [4:0]  rd_tile_i,
    input  logic [2:0]  rd_row_i,
    input  logic [2:0]  rd_col_i,
    output logic [31:0] rd_data_o, 

    //----------------------------------
    // Write Port (Tile Structural Coordinates)
    //----------------------------------
    input  logic        wr_en_i,
    input  logic [4:0]  wr_tile_i,
    input  logic [2:0]  wr_row_i,
    input  logic [2:0]  wr_col_i,
    input  logic [31:0] wr_data_i, // Patched to 32-bit width for balanced tracking
    input  logic        wr_pair_i  // 1 = paired write (scale), 0 = single-cell (bias)
);

    // Architectural Dimension Parameters
    localparam int unsigned NTILES = 32;
    localparam int unsigned TT     = 8;
    localparam int unsigned DEPTH  = NTILES * TT * TT; // 32 * 8 * 8 = 2048 entries
    localparam int unsigned ADDR_W = 11;

    // Accumulator Tile Memory Array
    (* ram_style = "block" *)
    logic [15:0] accum_mem [0:DEPTH-1];

    // Internal flattened physical address structures
    logic [ADDR_W-1:0] wr_addr_flat;
    assign wr_addr_flat = (ADDR_W'(wr_tile_i) << 6) + (ADDR_W'(wr_row_i) << 3) + ADDR_W'(wr_col_i);

    //----------------------------------
    // Tracking Variables for Latency Aligned Debugging
    //----------------------------------
    logic        rd_en_q;
    logic [4:0]  rd_tile_q;
    logic [2:0]  rd_row_q;
    logic [2:0]  rd_col_q;

    //----------------------------------
    // Synchronous Read Process
    //----------------------------------
    always_ff @(posedge clk_i) begin
        if (rd_en_i) begin
            // Safety enforcement: ensure read alignments hit even group row boundaries
            assert(rd_row_i[0] == 1'b0) else
                $error("[BRAM_ACCUM_ERROR] Read row must be even for paired layout, got %0d", rd_row_i);

            // Fetch row pair in a single memory lookup cycle
            rd_data_o <= {
                accum_mem[(ADDR_W'(rd_tile_i) << 6) + (ADDR_W'(rd_row_i + 1'b1) << 3) + ADDR_W'(rd_col_i)], // row_n + 1 (bits 31:16)
                accum_mem[(ADDR_W'(rd_tile_i) << 6) + (ADDR_W'(rd_row_i)        << 3) + ADDR_W'(rd_col_i)]  // row_n     (bits 15:0)
            };
        end
        
        if (!rst_ni) begin
            rd_en_q   <= 1'b0;
            rd_tile_q <= '0;
            rd_row_q  <= '0;
            rd_col_q  <= '0;
        end else begin
            rd_en_q   <= rd_en_i;
            rd_tile_q <= rd_tile_i;
            rd_row_q  <= rd_row_i;
            rd_col_q  <= rd_col_i;
        end
    end

    //----------------------------------
    // Synchronous Write Process
    //----------------------------------
    always_ff @(posedge clk_i) begin
        if (wr_en_i) begin
            if (wr_pair_i) begin
                // paired write: row n from [15:0], row n+1 from [31:16] (scale fold)
                assert(wr_row_i[0] == 1'b0) else
                    $error("[BRAM_ACCUM_ERROR] Paired write row must be even, got %0d", wr_row_i);

                // row n
                accum_mem[wr_addr_flat] <= wr_data_i[15:0];

                // row n + 1
                accum_mem[(ADDR_W'(wr_tile_i) << 6) + (ADDR_W'(wr_row_i + 1'b1) << 3) + ADDR_W'(wr_col_i)] <= wr_data_i[31:16];
            end else begin
                // single-cell write: only the addressed (tile,row,col), any row (bias)
                accum_mem[wr_addr_flat] <= wr_data_i[15:0];
            end
        end
    end

    //----------------------------------
    // Simulation Debug Dumps
    //----------------------------------
    always_ff @(posedge clk_i) begin
        if (rst_ni) begin
            if (rd_en_q) begin
                $display("[BRAM_ACCUM_DEBUG] [%0t ns] MEMORY READ COMPLETE:", $time);
                $display("[BRAM_ACCUM_DEBUG]   Coordinates -> Tile=%2d | Rows=%1d,%1d | Col=%1d", rd_tile_q, rd_row_q, rd_row_q+1, rd_col_q);
                $display("[BRAM_ACCUM_DEBUG]   Payload     -> Out Data=32'h%h", rd_data_o);
                $display("[BRAM_ACCUM_DEBUG]                row0=%4h row1=%4h", rd_data_o[15:0], rd_data_o[31:16]);
            end

            if (wr_en_i) begin
                $display("[BRAM_ACCUM_DEBUG] [%0t ns] MEMORY WRITE TRANSACTION COMMITTED:", $time);
                $display("[BRAM_ACCUM_DEBUG]   Coordinates -> Tile=%2d | Rows=%1d,%1d | Col=%1d", wr_tile_i, wr_row_i, wr_row_i+1, wr_col_i);
                $display("[BRAM_ACCUM_DEBUG]   Payload     -> In Data=32'h%h", wr_data_i);
                $display("[BRAM_ACCUM_DEBUG]                row0=%4h row1=%4h", wr_data_i[15:0], wr_data_i[31:16]);
            end
        end
    end

    // Display Array Monitor helper logic
integer r, c, addr;

always_ff @(posedge clk_i) begin
    //if (rst_ni && wr_en_i) begin

        //--------------------------------------------------
        // Tile 0
        //--------------------------------------------------
        $display("\n======================================================");
        $display("ACCUMULATOR BRAM TILE 0 @ time %0t", $time);

        for (r = 0; r < TT; r++) begin
            $write("Row %0d :", r);
            for (c = 0; c < TT; c++) begin
                addr = (0 << 6) + (r << 3) + c;
                $write(" %6h", accum_mem[addr]);
            end
            $write("\n");
        end

        $display("======================================================");


        //--------------------------------------------------
        // Tile 1
        //--------------------------------------------------
       // $display("ACCUMULATOR BRAM TILE 1 @ time %0t", $time);

        //for (r = 0; r < TT; r++) begin
         //   $write("Row %0d :", r);
         //   for (c = 0; c < TT; c++) begin
          //      addr = (1 << 6) + (r << 3) + c;
           //     $write(" %6h", accum_mem[addr]);
            //end
           // $write("\n");
       // end

      //  $display("======================================================");


        //--------------------------------------------------
        // Tile 31
        //--------------------------------------------------
      //  $display("ACCUMULATOR BRAM TILE 31 @ time %0t", $time);

      //  for (r = 0; r < TT; r++) begin
       //     $write("Row %0d :", r);
        //    for (c = 0; c < TT; c++) begin
       ///         addr = (31 << 6) + (r << 3) + c;
        //        $write(" %6h", accum_mem[addr]);
        //    end
        //    $write("\n");
       // end

       // $display("======================================================\n");

    //end
end
endmodule
