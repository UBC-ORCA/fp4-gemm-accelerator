module cve2_fp4 import cve2_pkg::*; #(
  parameter bit ConvSupport = 1'b1,
  parameter logic TILE_SIZE = 8
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  // Decode / control
  input  logic              fp4_en_i,
  input  fp4_op_e           operator_i,
  input  logic [11:0]       imm12_i, 

  // Operand Specifiers
  input  logic [4:0]        op_a_spec,
  input  logic [4:0]        op_b_spec,
  input  logic [4:0]        op_dst_spec, 

  // Operand Contents
  input  logic [31:0]       op_a_i,
  input  logic [31:0]       op_b_i,

  // Result
  output logic              valid_o,
  output logic [31:0]       fp4_result_o,

  // Memory interface 
  output logic              mem_w_en,
  output logic [31:0]       mem_w_data,
  output logic [31:0]       mem_w_addr 
);

  localparam int MAX_TILE_SIZE = 32;
  localparam int MAX_ROWS = MAX_TILE_SIZE;
  localparam int MAX_COLS = MAX_TILE_SIZE * 2;
  localparam int MIN_TILE_SIZE = 8;

  initial begin
    assert(MIN_TILE_SIZE < TILE_SIZE && TILE_SIZE <= MAX_TILE_SIZE) else 
    $fatal(1,
      "TILE_SIZE (%0d) must satisfy %0d < TILE_SIZE <= %0d\n",
      TILE_SIZE,
      MIN_TILE_SIZE,
      MAX_TILE_SIZE
    );
  end

 
  localparam int TILE_ROWS = TILE_SIZE;
  localparam int TILE_COLS = TILE_SIZE * 2;

  /* INT16 accumulator tile */
  logic signed [15:0] t_q [TILE_ROWS][TILE_COLS];
  logic signed [15:0] t_d [TILE_ROWS][TILE_COLS];

  /* FP4 Weight Registers */
  logic signed [3:0] w_q [TILE_ROWS];
  logic signed [3:0] w_d [TILE_ROWS];

  /* FP4 Activation Registers */
  logic signed [3:0] a_q [TILE_COLS];
  logic signed [3:0] a_d [TILE_COLS];

  // Write enables

  /* 
      Fine grained per row access 
      t_we[i] enables row i to be written
   */
  logic [TILE_ROWS-1 : 0] t_we;
  logic a_we;
  logic w_we;
  
  /* Result array for the FP4 x FP4 multiplication */
  logic signed [8:0]  hw_product_i9  [TILE_ROWS][TILE_COLS];
  
  /* Outer product array */
  for (genvar r = 0; r < TILE_ROWS; r++) begin : gen_hw_mul_row
    for (genvar c = 0; c < TILE_COLS; c++) begin : gen_hw_mul_col
    
      fp4_mul u_fp4_mul (
        .FP4inA  (w_q[r]),
        .FP4inB  (a_q[c]),
        .int9Out (hw_product_i9[r][c])
      );
    
    end
  end

  /* 
      Clamped 16-bit addition.  
      Result saturates if a + b falls out of 
      the int16 range. 
   */
  function automatic logic signed [15:0] saturated_add16(
    input logic signed [15:0] a,
    input logic signed [15:0] b
  );

    logic signed [16:0] add16_result; 
    add16_result = a + b;   

    if (add16_result > 16'sh7fff)
      return  16'sh7fff;
    
    if (add16_result < 16'sh8000)
      return 16'sh8000;
    
    return add16_result[15:0];

  endfunction

  // ------------------------------------------------------------
  // Combinational next-state logic
  // ------------------------------------------------------------

  always_comb begin

    // Defaults
    t_d           = t_q;
    a_d           = a_q;

    t_we          = 'b1;
    a_we          = 1'b0;

    valid_o       = fp4_en_i;

    mem_w_en      = 1'b0;
    mem_w_data    = 32'b0;
    mem_w_addr    = 32'b0;

    fp4_result_o  = 32'b0;

    // ----------------------------------------------------------
    // Instruction decode
    // ----------------------------------------------------------

    unique case (operator_i)

      // ========================================================
      // zzMAC64
      //
      // Clear entire accumulator tile T
      // ========================================================

      FP4_ZZMAC: begin

        t_we = 'b1;

        for (int r = 0; r < TILE_ROWS; r++) begin
          for (int c = 0; c < TILE_COLS; c++) begin
            t_d[r][c] = 16'sd0;
          end
        end

      end

      // ========================================================
      // maxMAC64
      //
      // Example template:
      // T = max(T, scalar)
      // ========================================================

      FP4_MAXMAC: begin

        t_we = 'b1;
        
        for (int r = 0; r < TILE_ROWS; r++) begin
          for (int c = 0; c < TILE_COLS; c++) begin
            t_d[r][c] = (t_q[r][c] > $signed(op_a_i[15:0])) ? t_q[r][c] : $signed(op_a_i[15:0]);
          end
        end

      end

      // ========================================================
      // hwMAC64
      //
      // Outer-product accumulation
      //
      // T[r][c] += rs1[r] * rs2[c]
      // ========================================================

      FP4_HWMAC: begin

        t_we = 'b1;
        
        for (int r = 0; r < TILE_ROWS; r++) begin
          for (int c = 0; c < TILE_COLS; c++) begin
            t_d[r][c] = saturated_add16(t_q[r][c], hw_product_i9[r][c]);
          end
        end
        
      end 

      // ========================================================
      // ad2MAC64
      //
      // Add two packed int16 bias values T[i][2*j] += rs1[15:0] 
      // and T[i][2*j+1] += rs1[31:16]
      // ========================================================

      FP4_ADDMAC: begin

        automatic logic [4:0] r = op_a_spec;
        /* Only perform action if the row is in range */
        if ( r < TILE_ROWS ) begin
          /* Only enable write on selected row */
          t_we = (1'b1 << r); 
          
          for (int c = 0; c < TILE_COLS; ++c) begin
            t_d[r][c] = saturated_add16(t_q[r][c], op_b_i[15:0]);
          end
        end

      end 

      // ========================================================
      // mveMAC64
      //
      // Move even tile entry to rd and clear tile entry
      // ========================================================

      FP4_MVEMAC64: begin
        t_we = 'b1;
        fp4_result_o = t_q[op_a_spec][2 * op_b_spec];
        valid_o = 1'b1;

        t_d[op_a_spec] = t_q[op_a_spec];
        t_d[op_a_spec][2 * op_b_spec] = 0;
      end

      // ========================================================
      // mvoMAC64
      // 
      // Move odd tile entry to rd and clear tile entry
      // ========================================================

      FP4_MVOMAC64: begin
        t_we = 1'b1;
        fp4_result_o = t_q[op_a_spec][2 * op_b_spec + 1];
        valid_o = 1'b1;

        t_d[op_a_spec] = t_q;
        t_d[op_a_spec][2 * op_b_spec + 1] = 0;

      end

      // ========================================================
      // mv2MAC64
      // ========================================================

      FP4_MV2MAC64: begin

        t_we = 1'b1;
        fp4_result_o = {t_q[op_a_spec][2 * op_b_spec + 1],
                         t_q[op_a_spec][2 * op_b_spec]};
        valid_o = 1'b1;
        t_d[op_a_spec] = t_q;
        t_d[op_a_spec][2 * op_b_spec + 1] = 0;
        t_d[op_a_spec][2 * op_b_spec] = 0;
  
      end

      // ========================================================
      // ld2MAC64
      // ========================================================

      FP4_LD2MAC64: begin

        t_we = 1'b1;
        // to be done
        
      end

      // ========================================================
      // st2MAC64
      // Stores {T[rs1][IMM12[6:1]+1], T[rs1][IMM12[6:1]]}
      // to address rs2 + IMM12[6:1]. 
      // 
      // Zeroes both tile entries to 0.
      // ========================================================

      FP4_ST2MAC64: begin

        automatic int r = op_a_spec;
        automatic int c = imm12_i[6:1];

        t_we = 'b1; 
        mem_w_en = 1'b1;
        mem_w_addr = imm12_i + op_b_i;

        /* Do nothing if out of range */
        if (r < TILE_ROWS && (c < TILE_COLS) 
            && ((c+1) < TILE_COLS)) begin
          mem_w_data = {t_q[r][c+1], t_q[r][c]};
          
          /* For now, row granularity assigment */          
          t_d[r] = t_q[r];
          t_d[r][c] = 16'b0;
          t_d[r][c+1] = 16'b0;
        end            
      end

      // ========================================================
      // mvA
      //
      // Load two rows into activation tile A
      // ========================================================

      FP4_MVA: begin

        if (ConvSupport) begin

          a_we = 1'b1;
          // to be done

        end

      end

      // ========================================================
      // Convolution instructions
      // ========================================================

      FP4_CONV,
      FP4_CONVLC,
      FP4_CONVRC,
      FP4_CONVUR,
      FP4_CONVDR: begin

        if (ConvSupport) begin

          t_we = 1'b1;
          // to be done

        end

      end

      // ========================================================
      // Default
      // ========================================================

      default: begin

        valid_o = 1'b0;

      end

    endcase

  end

  // ------------------------------------------------------------
  // Sequential state update
  // ------------------------------------------------------------

  always_ff @(posedge clk_i) begin

    if (!rst_ni) begin

      for (int r = 0; r < 8; r++) begin
        for (int c = 0; c < 8; c++) begin

          t_q[r][c] <= 16'sd0;
          a_q[r][c] <= 4'd0;

        end
      end

    end else begin
      
      /* Row-grained t_q select. */
      for (int i = 0; i < MAX_ROWS; ++i) begin
        if (t_we[i]) begin 
          t_q[i] <= t_d[i];
        end
      end

      if (a_we)
        a_q <= a_d;

    end

  end

endmodule

