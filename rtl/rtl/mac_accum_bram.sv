// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Structured coordinate-addressed Accumulator Tile BRAM module.
// Maintains clean ISA abstractions externally while organizing internal physical 
// memory structures to reliably force Vivado Block RAM (BRAM) inference.

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
    input  logic [31:0] wr_data_i,   
    input  logic        wr_pair_i    // 1 = paired write (scale), 0 = single-cell (bias)
);

    // Architectural Dimension Parameters
    localparam int unsigned NTILES = 32;
    localparam int unsigned TT     = 8;
    localparam int unsigned DEPTH  = NTILES * (TT / 2) * TT; // 32 * 4 * 8 = 1024 physical entries
    localparam int unsigned ADDR_W = 10;

    // Accumulator Tile Memory Array (Maps natively onto 1 RAMB36 configured as 1Kx32)
    (* ram_style = "block" *)
    logic [31:0] accum_mem [0:DEPTH-1];

    //----------------------------------
    // Architectural-to-Physical Translation Layer
    //----------------------------------
    logic [ADDR_W-1:0] rd_addr_flat;
    logic [ADDR_W-1:0] wr_addr_flat;

    // Row bits [2:1] select the row pair, preserving the linear col mapping
    assign rd_addr_flat = (ADDR_W'(rd_tile_i) << 5) + (ADDR_W'(rd_row_i[2:1]) << 3) + ADDR_W'(rd_col_i);
    assign wr_addr_flat = (ADDR_W'(wr_tile_i) << 5) + (ADDR_W'(wr_row_i[2:1]) << 3) + ADDR_W'(wr_col_i);

    //----------------------------------
    // Read Path (Synchronous Memory + Output Demux)
    //----------------------------------
    logic [ADDR_W-1:0] rd_addr_q;
    logic              rd_en_q;
    logic [2:0]        rd_row_q;
    logic [31:0]       rd_word_q;

    // Pipeline Stage 1: Register Address & Control
    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            rd_en_q   <= 1'b0;
            rd_addr_q <= '0;
            rd_row_q  <= '0;
        end else begin
            rd_en_q   <= rd_en_i;
            if (rd_en_i) begin
                rd_addr_q <= rd_addr_flat;
                rd_row_q  <= rd_row_i;
            end
        end
    end

    // Pipeline Stage 2: Array Core Access
    always_ff @(posedge clk_i) begin
        if (rd_en_q) begin
            rd_word_q <= accum_mem[rd_addr_q];
        end
    end

    // Pipeline Stage 3: Architectural Alignment Output Formatter
    always_comb begin
        if (rd_row_q[0] == 1'b0) begin
            // Even Row Access: Returns the full 32-bit word matching {row+1, row}
            rd_data_o = rd_word_q;
        end else begin
            // Odd Row Access: Shift upper 16-bits to the lower position (if required by system context)
            rd_data_o = {16'h0000, rd_word_q[31:16]};
        end
    end

    //----------------------------------
    // Write Path (Masked Half-Word Matrix)
    //----------------------------------
    logic [1:0]  wr_mask;
    logic [15:0] data_lower;
    logic [15:0] data_upper;

    always_comb begin
        if (wr_pair_i) begin
            // Paired Mode (Scale update): Commit both half-words simultaneously
            wr_mask    = 2'b11;
            data_lower = wr_data_i[15:0];
            data_upper = wr_data_i[31:16];
        end else begin
            // Single-Cell Mode (Bias update): Drive sub-word write enable pins via bit[0] row parity
            if (wr_row_i[0] == 1'b0) begin
                wr_mask    = 2'b01;           // Write even row (lower 16 bits)
                data_lower = wr_data_i[15:0];
                data_upper = 16'h0000;
            end else begin
                wr_mask    = 2'b10;           // Write odd row (upper 16 bits)
                data_lower = 16'h0000;
                data_upper = wr_data_i[15:0]; // Single-cell elements arrive on lower lines
            end
        end
    end

    // Synchronous Matrix Write Block (Cleanly mapped to standard byte-steer logic)
    always_ff @(posedge clk_i) begin
        if (wr_en_i) begin
            if (wr_pair_i) begin
                assert(wr_row_i[0] == 1'b0) else
                    $error("[BRAM_ACCUM_ERROR] Paired write row must be even, got %0d", wr_row_i);
            end
            
            if (wr_mask[0]) accum_mem[wr_addr_flat][15:0]  <= data_lower;
            if (wr_mask[1]) accum_mem[wr_addr_flat][31:16] <= data_upper;
        end
    end

    //----------------------------------
    // Legacy Diagnostic Pipeline Tracking
    //----------------------------------
    logic [4:0] rd_tile_q_delay;
    logic [2:0] rd_col_q_delay;
    logic       rd_en_qq;

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            rd_en_qq         <= 1'b0;
            rd_tile_q_delay  <= '0;
            rd_col_q_delay   <= '0;
        end else begin
            rd_en_qq         <= rd_en_q;
            rd_tile_q_delay  <= rd_tile_i; // aligned for display cycle timing
            rd_col_q_delay   <= rd_col_i;
        end
    end

//DEBUG
    //----------------------------------
    // Simulation Debug Dumps
    //----------------------------------
`ifdef BRAM_DEBUG

    integer r, c, addr;

    always_ff @(posedge clk_i) begin
        if (rst_ni) begin

            //----------------------------------
            // READ PIPELINE COMPLETE
            //----------------------------------
            if (rd_en_qq) begin
                $display("\n======================================================");
                $display("[BRAM_ACCUM_DEBUG] [%0t ns] READ TRANSACTION COMPLETE", $time);

                $display("[ARCHITECTURAL VIEW]");
                $display("    Tile          = %0d", rd_tile_q_delay);
                $display("    Requested Row = %0d", rd_row_q);
                $display("    Requested Col = %0d", rd_col_q_delay);

                $display("[PHYSICAL BRAM VIEW]");
                $display("    Row Pair      = %0d", rd_row_q[2:1]);
                $display("    BRAM Address  = %0d (0x%03h)", rd_addr_q, rd_addr_q);

                $display("[RAW BRAM PAYLOAD]");
                $display("    BRAM Word     = 0x%08h", rd_word_q);
                $display("        Upper16   = 0x%04h", rd_word_q[31:16]);
                $display("        Lower16   = 0x%04h", rd_word_q[15:0]);

                $display("[ARCHITECTURAL OUTPUT]");
                $display("    rd_data_o     = 0x%08h", rd_data_o);

                if (rd_row_q[0] == 1'b0) begin
                    $display("    Access Type   = EVEN ROW PAIR READ");
                    $display("    Maps To       = {row=%0d,col=%0d , row=%0d,col=%0d}",
                             rd_row_q + 1'b1,
                             rd_col_q_delay,
                             rd_row_q,
                             rd_col_q_delay);

                    $display("    Returned     -> row%0d = 0x%04h",
                             rd_row_q,
                             rd_data_o[15:0]);

                    $display("                    row%0d = 0x%04h",
                             rd_row_q+1'b1,
                             rd_data_o[31:16]);

                end
                else begin
                    $display("    Access Type   = ODD ROW SINGLE READ");
                    $display("    Maps To       = row=%0d,col=%0d",
                             rd_row_q,
                             rd_col_q_delay);

                    $display("    Returned     -> row%0d = 0x%04h",
                             rd_row_q,
                             rd_data_o[15:0]);
                end

                $display("======================================================\n");
            end


            //----------------------------------
            // WRITE TRANSACTION
            //----------------------------------
            if (wr_en_i) begin

                $display("\n======================================================");
                $display("[BRAM_ACCUM_DEBUG] [%0t ns] WRITE TRANSACTION", $time);

                $display("[ARCHITECTURAL VIEW]");
                $display("    Tile          = %0d", wr_tile_i);
                $display("    Target Row    = %0d", wr_row_i);
                $display("    Target Col    = %0d", wr_col_i);

                $display("[PHYSICAL BRAM VIEW]");
                $display("    Row Pair      = %0d", wr_row_i[2:1]);
                $display("    BRAM Address  = %0d (0x%03h)",
                         wr_addr_flat,
                         wr_addr_flat);

                $display("[WRITE CONTROL]");
                $display("    Pair Write    = %0b", wr_pair_i);
                $display("    Write Mask    = 2'b%b", wr_mask);

                $display("[INPUT PAYLOAD]");
                $display("    wr_data_i     = 0x%08h", wr_data_i);
                $display("        Upper16   = 0x%04h", wr_data_i[31:16]);
                $display("        Lower16   = 0x%04h", wr_data_i[15:0]);

                if (wr_pair_i) begin
                    $display("[PAIRED SCALE WRITE]");
                    $display("    row%0d <- 0x%04h",
                             wr_row_i,
                             wr_data_i[15:0]);

                    $display("    row%0d <- 0x%04h",
                             wr_row_i+1'b1,
                             wr_data_i[31:16]);

                end
                else if (wr_row_i[0] == 1'b0) begin

                    $display("[SINGLE CELL BIAS WRITE]");
                    $display("    EVEN ROW WRITE");
                    $display("    row%0d <- 0x%04h",
                             wr_row_i,
                             wr_data_i[15:0]);

                end
                else begin

                    $display("[SINGLE CELL BIAS WRITE]");
                    $display("    ODD ROW WRITE");
                    $display("    row%0d <- 0x%04h",
                             wr_row_i,
                             wr_data_i[15:0]);

                end

                $display("======================================================\n");
            end

        end
    end


    //----------------------------------
    // Optional Full Tile Dump
    //----------------------------------
    always_ff @(posedge clk_i) begin

        if (rst_ni && wr_en_i) begin

            $display("\n======================================================");
            $display("ACCUMULATOR BRAM TILE 0 @ time %0t", $time);

            for (r = 0; r < TT/2; r++) begin

                $display("--------------------------------------");
                $display("Physical Row Pair %0d", r);

                for (c = 0; c < TT; c++) begin

                    addr = (0 << 5) + (r << 3) + c;

                    $display(
                        "Addr=%03d Col=%0d | Row%0d=0x%04h Row%0d=0x%04h",
                        addr,
                        c,
                        (r<<1),
                        accum_mem[addr][15:0],
                        (r<<1)+1,
                        accum_mem[addr][31:16]
                    );
                end
            end

            $display("======================================================\n");

        end

//Tile0
//--------------------------------------------------
// Tile 0
//--------------------------------------------------
$display("\n======================================================");
$display("ACCUMULATOR BRAM TILE 0 @ time %0t", $time);

$display("Row 0 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(0<<5)+(0<<3)+0][15:0],
    accum_mem[(0<<5)+(0<<3)+1][15:0],
    accum_mem[(0<<5)+(0<<3)+2][15:0],
    accum_mem[(0<<5)+(0<<3)+3][15:0],
    accum_mem[(0<<5)+(0<<3)+4][15:0],
    accum_mem[(0<<5)+(0<<3)+5][15:0],
    accum_mem[(0<<5)+(0<<3)+6][15:0],
    accum_mem[(0<<5)+(0<<3)+7][15:0]);

$display("Row 1 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(0<<5)+(0<<3)+0][31:16],
    accum_mem[(0<<5)+(0<<3)+1][31:16],
    accum_mem[(0<<5)+(0<<3)+2][31:16],
    accum_mem[(0<<5)+(0<<3)+3][31:16],
    accum_mem[(0<<5)+(0<<3)+4][31:16],
    accum_mem[(0<<5)+(0<<3)+5][31:16],
    accum_mem[(0<<5)+(0<<3)+6][31:16],
    accum_mem[(0<<5)+(0<<3)+7][31:16]);

$display("Row 2 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(0<<5)+(1<<3)+0][15:0],
    accum_mem[(0<<5)+(1<<3)+1][15:0],
    accum_mem[(0<<5)+(1<<3)+2][15:0],
    accum_mem[(0<<5)+(1<<3)+3][15:0],
    accum_mem[(0<<5)+(1<<3)+4][15:0],
    accum_mem[(0<<5)+(1<<3)+5][15:0],
    accum_mem[(0<<5)+(1<<3)+6][15:0],
    accum_mem[(0<<5)+(1<<3)+7][15:0]);

$display("Row 3 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(0<<5)+(1<<3)+0][31:16],
    accum_mem[(0<<5)+(1<<3)+1][31:16],
    accum_mem[(0<<5)+(1<<3)+2][31:16],
    accum_mem[(0<<5)+(1<<3)+3][31:16],
    accum_mem[(0<<5)+(1<<3)+4][31:16],
    accum_mem[(0<<5)+(1<<3)+5][31:16],
    accum_mem[(0<<5)+(1<<3)+6][31:16],
    accum_mem[(0<<5)+(1<<3)+7][31:16]);

$display("Row 4 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(0<<5)+(2<<3)+0][15:0],
    accum_mem[(0<<5)+(2<<3)+1][15:0],
    accum_mem[(0<<5)+(2<<3)+2][15:0],
    accum_mem[(0<<5)+(2<<3)+3][15:0],
    accum_mem[(0<<5)+(2<<3)+4][15:0],
    accum_mem[(0<<5)+(2<<3)+5][15:0],
    accum_mem[(0<<5)+(2<<3)+6][15:0],
    accum_mem[(0<<5)+(2<<3)+7][15:0]);

$display("Row 5 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(0<<5)+(2<<3)+0][31:16],
    accum_mem[(0<<5)+(2<<3)+1][31:16],
    accum_mem[(0<<5)+(2<<3)+2][31:16],
    accum_mem[(0<<5)+(2<<3)+3][31:16],
    accum_mem[(0<<5)+(2<<3)+4][31:16],
    accum_mem[(0<<5)+(2<<3)+5][31:16],
    accum_mem[(0<<5)+(2<<3)+6][31:16],
    accum_mem[(0<<5)+(2<<3)+7][31:16]);

$display("Row 6 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(0<<5)+(3<<3)+0][15:0],
    accum_mem[(0<<5)+(3<<3)+1][15:0],
    accum_mem[(0<<5)+(3<<3)+2][15:0],
    accum_mem[(0<<5)+(3<<3)+3][15:0],
    accum_mem[(0<<5)+(3<<3)+4][15:0],
    accum_mem[(0<<5)+(3<<3)+5][15:0],
    accum_mem[(0<<5)+(3<<3)+6][15:0],
    accum_mem[(0<<5)+(3<<3)+7][15:0]);

$display("Row 7 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(0<<5)+(3<<3)+0][31:16],
    accum_mem[(0<<5)+(3<<3)+1][31:16],
    accum_mem[(0<<5)+(3<<3)+2][31:16],
    accum_mem[(0<<5)+(3<<3)+3][31:16],
    accum_mem[(0<<5)+(3<<3)+4][31:16],
    accum_mem[(0<<5)+(3<<3)+5][31:16],
    accum_mem[(0<<5)+(3<<3)+6][31:16],
    accum_mem[(0<<5)+(3<<3)+7][31:16]);

$display("======================================================");
//Tile0_end

//Tile31
//--------------------------------------------------
// Tile 31
//--------------------------------------------------
$display("\n======================================================");
$display("ACCUMULATOR BRAM TILE 31 @ time %0t", $time);

$display("Row 0 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(31<<5)+(0<<3)+0][15:0],
    accum_mem[(31<<5)+(0<<3)+1][15:0],
    accum_mem[(31<<5)+(0<<3)+2][15:0],
    accum_mem[(31<<5)+(0<<3)+3][15:0],
    accum_mem[(31<<5)+(0<<3)+4][15:0],
    accum_mem[(31<<5)+(0<<3)+5][15:0],
    accum_mem[(31<<5)+(0<<3)+6][15:0],
    accum_mem[(31<<5)+(0<<3)+7][15:0]);

$display("Row 1 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(31<<5)+(0<<3)+0][31:16],
    accum_mem[(31<<5)+(0<<3)+1][31:16],
    accum_mem[(31<<5)+(0<<3)+2][31:16],
    accum_mem[(31<<5)+(0<<3)+3][31:16],
    accum_mem[(31<<5)+(0<<3)+4][31:16],
    accum_mem[(31<<5)+(0<<3)+5][31:16],
    accum_mem[(31<<5)+(0<<3)+6][31:16],
    accum_mem[(31<<5)+(0<<3)+7][31:16]);

$display("Row 2 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(31<<5)+(1<<3)+0][15:0],
    accum_mem[(31<<5)+(1<<3)+1][15:0],
    accum_mem[(31<<5)+(1<<3)+2][15:0],
    accum_mem[(31<<5)+(1<<3)+3][15:0],
    accum_mem[(31<<5)+(1<<3)+4][15:0],
    accum_mem[(31<<5)+(1<<3)+5][15:0],
    accum_mem[(31<<5)+(1<<3)+6][15:0],
    accum_mem[(31<<5)+(1<<3)+7][15:0]);

$display("Row 3 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(31<<5)+(1<<3)+0][31:16],
    accum_mem[(31<<5)+(1<<3)+1][31:16],
    accum_mem[(31<<5)+(1<<3)+2][31:16],
    accum_mem[(31<<5)+(1<<3)+3][31:16],
    accum_mem[(31<<5)+(1<<3)+4][31:16],
    accum_mem[(31<<5)+(1<<3)+5][31:16],
    accum_mem[(31<<5)+(1<<3)+6][31:16],
    accum_mem[(31<<5)+(1<<3)+7][31:16]);

$display("Row 4 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(31<<5)+(2<<3)+0][15:0],
    accum_mem[(31<<5)+(2<<3)+1][15:0],
    accum_mem[(31<<5)+(2<<3)+2][15:0],
    accum_mem[(31<<5)+(2<<3)+3][15:0],
    accum_mem[(31<<5)+(2<<3)+4][15:0],
    accum_mem[(31<<5)+(2<<3)+5][15:0],
    accum_mem[(31<<5)+(2<<3)+6][15:0],
    accum_mem[(31<<5)+(2<<3)+7][15:0]);

$display("Row 5 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(31<<5)+(2<<3)+0][31:16],
    accum_mem[(31<<5)+(2<<3)+1][31:16],
    accum_mem[(31<<5)+(2<<3)+2][31:16],
    accum_mem[(31<<5)+(2<<3)+3][31:16],
    accum_mem[(31<<5)+(2<<3)+4][31:16],
    accum_mem[(31<<5)+(2<<3)+5][31:16],
    accum_mem[(31<<5)+(2<<3)+6][31:16],
    accum_mem[(31<<5)+(2<<3)+7][31:16]);

$display("Row 6 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(31<<5)+(3<<3)+0][15:0],
    accum_mem[(31<<5)+(3<<3)+1][15:0],
    accum_mem[(31<<5)+(3<<3)+2][15:0],
    accum_mem[(31<<5)+(3<<3)+3][15:0],
    accum_mem[(31<<5)+(3<<3)+4][15:0],
    accum_mem[(31<<5)+(3<<3)+5][15:0],
    accum_mem[(31<<5)+(3<<3)+6][15:0],
    accum_mem[(31<<5)+(3<<3)+7][15:0]);

$display("Row 7 : %6h %6h %6h %6h %6h %6h %6h %6h",
    accum_mem[(31<<5)+(3<<3)+0][31:16],
    accum_mem[(31<<5)+(3<<3)+1][31:16],
    accum_mem[(31<<5)+(3<<3)+2][31:16],
    accum_mem[(31<<5)+(3<<3)+3][31:16],
    accum_mem[(31<<5)+(3<<3)+4][31:16],
    accum_mem[(31<<5)+(3<<3)+5][31:16],
    accum_mem[(31<<5)+(3<<3)+6][31:16],
    accum_mem[(31<<5)+(3<<3)+7][31:16]);

$display("======================================================\n");
//Tile31_end

    end

`endif
//DEBUG_end

endmodule
