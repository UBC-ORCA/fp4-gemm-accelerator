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
  // int_to_fp4
  //
  // Converts a shifted INT16 value into a 4-bit FP4 value 
  //
  // INPUTS:
  //   val — a signed 16-bit integer whose magnitude represents a quantity in
  //         units of 0.5.  That is, the integer 1 means 0.5, integer 2 means
  //         1.0, integer 8 means 4.0, integer 12 means 6.0, and so on.
  //         This is the natural result after setAMAC's right-shift step.
  //
  // OUTPUT:
  //   4-bit FP4:  bit[3]   = sign (1 = negative)
  //               bits[2:0] = magnitude encoding
  //                 000 = 0.0,  001 = 0.5,  010 = 1.0,  011 = 1.5
  //                 100 = 2.0,  101 = 3.0,  110 = 4.0,  111 = 6.0
  //
  // ROUNDING:
  //   Truncation (round toward zero): the largest FP4 magnitude that is
  //   <= the actual magnitude is chosen.  So 1.7 → 1.5 (not 2.0).
  //   Values exceeding 6.0 clamp to 6.0 (the FP4 maximum).
  // ------------------------------------------------------------
  function automatic logic [3:0] int_to_fp4(
      input logic signed [15:0] val
  );
      logic        sign;
      logic [15:0] abs_val;
      logic [2:0]  fp4_mag;

      // extract the sign bit.
      sign = val[15];

      // compute unsigned absolute value
      // For positive val, abs_val is the value itself (bit 15 is already 0)
      // For negative val, negate using two's complement: ~val + 1
      abs_val = sign ? 16'(-val) : 16'(val);

      // map the absolute value to the nearest FP4 magnitude <= abs_val.
      // Comparison thresholds are the FP4 magnitudes expressed in "half-units"
      if      (abs_val >= 16'd12) fp4_mag = 3'b111; // 6.0
      else if (abs_val >= 16'd8)  fp4_mag = 3'b110; // 4.0
      else if (abs_val >= 16'd6)  fp4_mag = 3'b101; // 3.0
      else if (abs_val >= 16'd4)  fp4_mag = 3'b100; // 2.0
      else if (abs_val >= 16'd3)  fp4_mag = 3'b011; // 1.5
      else if (abs_val >= 16'd2)  fp4_mag = 3'b010; // 1.0
      else if (abs_val >= 16'd1)  fp4_mag = 3'b001; // 0.5
      else                        fp4_mag = 3'b000; // 0.0

      // Step 4: pack sign and magnitude into the 4-bit FP4 encoding.
      return {sign, fp4_mag};

  endfunction

  // ------------------------------------------------------------
  // Combinational next-state logic
  // ------------------------------------------------------------

  always_comb begin

    // ----------------------------------------------------------
    //
    //   shift_amt : the 5-bit right-shift amount read directly
    //
    //   val_lo    : lower INT16 of op_a_i after the shift is applied.
    //               This becomes A[2*rd] after FP4 conversion
    //
    //   val_hi    : upper INT16 of op_a_i after the shift is applied.
    //               This becomes A[2*rd+1] after FP4 conversion
    //
    //   col_lo    : computed A array index for val_lo  (= 2 * rd).
    //   col_hi    : computed A array index for val_hi  (= 2 * rd + 1).
    // ----------------------------------------------------------
    logic [4:0]         shift_amt;
    logic signed [15:0] val_lo;
    logic signed [15:0] val_hi;
    int                 col_lo;
    int                 col_hi;


    t_d           = t_q;
    a_d           = a_q;
    w_d           = w_q; 

    t_we          = 'b1;
    a_we          = 1'b0;
    w_we          = 1'b0; 

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
      // setWMAC  rd, rs1
      //
      // PURPOSE:
      //   Loads the W (weight) register array with 8 FP4 values that
      //   were packed side-by-side into the 32-bit source register rs1
      //
      // OPERANDS:
      //   op_a_i      (rs1 contents)  — 32-bit value holding 8 FP4 values
      //
      //   op_dst_spec (rd field)      — 5-bit group index selecting which
      //                 block of 8 weight slots to fill.
      //                 rd=0 → W[0..7], rd=1 → W[8..15], etc.
      //                 For the default tile (TILE_ROWS=8) only rd=0 is valid
      //
      // RESULT:
      //   w_d[rd*8 + k] = op_a_i[k*4 +: 4]  for k = 0..7
      //   w_we = 1  so the always_ff block commits w_d to w_q next cycle.
      //
      // GUARD:
      //   If rd*8+7 >= TILE_ROWS the write is silently skipped (safe no-op).
      // ========================================================

      FP4_SETWMAC: begin

        // Only proceed if the full group of 8 fits inside the weight array.
        if ((int'(op_dst_spec) * 8 + 7) < TILE_ROWS) begin

          w_we = 1'b1;

          // Unpack all 8 FP4 values from the 32-bit source register.
          // Each FP4 is 4 bits wide; the k-th value occupies bits [k*4+3 : k*4].
          // The +: operator is a fixed-width slice: op_a_i[k*4 +: 4] reads
          // 4 bits starting at bit k*4, which is exactly one FP4 field.
          for (int k = 0; k < 8; k++) begin
            w_d[op_dst_spec * 8 + k] = op_a_i[k*4 +: 4];
          end

        end

      end

      // ========================================================
      // setAMAC  rd, rs1, rs2
      //
      // PURPOSE:
      //   Loads TWO adjacent A (activation) register slots from the
      //   lower and upper INT16 halves of rs1.  Before storing, each
      //   INT16 is scaled down by an arithmetic right-shift, then
      //   quantised to the nearest FP4 value <= the result (truncation)
      //
      // OPERANDS:
      //   op_a_i      (rs1 contents)  — 32-bit value = two packed INT16s:
      //                 bits [15: 0] → lower INT16 → will become A[2*rd]
      //                 bits [31:16] → upper INT16 → will become A[2*rd+1]
      //
      //   op_b_spec   (rs2 FIELD, not rs2 register contents!) — 5-bit
      //                 immediate shift amount (0–10).  The actual register
      //                 value op_b_i is intentionally ignored here; the
      //                 shift amount is encoded directly in the instruction
      //
      //   op_dst_spec (rd field)      — 5-bit pair index selecting which
      //                 two adjacent A slots to fill:
      //                 rd=0 → A[0] and A[1]
      //                 rd=1 → A[2] and A[3]
      //                 rd=n → A[2n] and A[2n+1]
      //
      // RESULT:
      //   a_d[2*rd]   = int_to_fp4( op_a_i[15: 0] >>> op_b_spec )
      //   a_d[2*rd+1] = int_to_fp4( op_a_i[31:16] >>> op_b_spec )
      //   a_we = 1  so the always_ff block commits a_d to a_q next cycle.
      //
      // GUARD:
      //   If 2*rd+1 >= TILE_COLS the write is silently skipped (safe no-op).
      // ========================================================

      FP4_SETAMAC: begin

        // Read the shift amount directly from the rs2 instruction field.
        shift_amt = op_b_spec[4:0];

        // Compute the two A-array indices for this rd value.
        // Multiplying by 2 steps over pairs: rd=0→cols 0,1  rd=1→cols 2,3 etc.
        col_lo = int'(op_dst_spec) * 2;
        col_hi = int'(op_dst_spec) * 2 + 1;

        // Guard: only proceed if both column indices fall within the A array.
        if (col_hi < TILE_COLS) begin

          a_we = 1'b1;

          // Step 1: arithmetic right-shift each INT16 half by shift_amt bits.
          val_lo = $signed(op_a_i[15: 0]) >>> shift_amt;
          val_hi = $signed(op_a_i[31:16]) >>> shift_amt;

          // Step 2: convert each shifted value to FP4 and store it.
          a_d[col_lo] = int_to_fp4(val_lo);
          a_d[col_hi] = int_to_fp4(val_hi);

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

      // BUG FIX 1: loop bounds were hardcoded to 8, which left upper
      //   rows/columns of T uninitialised for tile sizes > 8.
      //   Changed to use the TILE_ROWS and TILE_COLS parameters.
      //
      // BUG FIX 2: a_q is a 1-D array declared as [TILE_COLS], not 2-D,
      //   so it cannot be indexed as a_q[r][c].  The original code treated
      //   it as 2-D inside the same nested loop as t_q, which is a type
      //   error.  Split into a separate loop over columns only.
      //
      // ADDED: w_q must also be zeroed on reset.  It was not touched at all
      //   in the original reset block, so W would have contained X values
      //   after reset, causing X-propagation into every hwMAC result.

      // Zero the entire accumulator tile T.
      for (int r = 0; r < TILE_ROWS; r++) begin
        for (int c = 0; c < TILE_COLS; c++) begin
          t_q[r][c] <= 16'sd0;
        end
      end

      // Zero all activation (A) register slots.
      for (int c = 0; c < TILE_COLS; c++) begin
        a_q[c] <= 4'sd0;
      end

      // Zero all weight (W) register slots.
      for (int r = 0; r < TILE_ROWS; r++) begin
        w_q[r] <= 4'sd0;
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
      if (w_we)
        w_q <= w_d;

    end

  end

endmodule