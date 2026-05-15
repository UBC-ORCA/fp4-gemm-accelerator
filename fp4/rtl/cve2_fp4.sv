module cve2_fp4 import cve2_pkg::*; #(
  parameter bit ConvSupport = 1'b1
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  // Decode / control
  input  logic              fp4_en_i,
  input  fp4_op_e           operator_i,

  // Source operands
  input  logic [31:0]       op_a_i,
  input  logic [31:0]       op_b_i,

  // Result
  output logic              valid_o,
  output logic [31:0]       fp4_result_o
);

  // 8x8 int16 accumulator tile
  logic signed [15:0] t_q [8][8];
  logic signed [15:0] t_d [8][8];

  // 8x8 FP4 activation tile for convolutions
  logic [3:0] a_q [8][8];
  logic [3:0] a_d [8][8];

  // Write enables
  logic t_we;
  logic a_we;
  
  logic signed [16:0] hw_sum [8][8];
  logic signed [16:0] ad2_sum_even;
  logic signed [16:0] ad2_sum_odd;
  
  logic [4:0] tile_index;
  logic [2:0] row_idx;
  logic [1:0] col_pair_idx;

  logic [2:0] col_even_idx;
  logic [2:0] col_odd_idx;

  assign tile_index  = op_b_i[4:0];

  assign row_idx     = tile_index[4:2];
  assign col_pair_idx = tile_index[1:0];

  assign col_even_idx = {col_pair_idx, 1'b0};
  assign col_odd_idx  = {col_pair_idx, 1'b1};
  
  logic [3:0] hw_weight     [8];
  logic [3:0] hw_activation [8];
    
  logic signed [8:0]  hw_product_i9  [8][8];
    
  for (genvar i = 0; i < 8; i++) begin : gen_hw_unpack
    assign hw_weight[i]     = op_a_i[i*4 +: 4];
    assign hw_activation[i] = op_b_i[i*4 +: 4];
  end
    
  for (genvar r = 0; r < 8; r++) begin : gen_hw_mul_row
    for (genvar c = 0; c < 8; c++) begin : gen_hw_mul_col
    
      fp4_mul u_fp4_mul (
        .FP4inA  (hw_weight[r]),
        .FP4inB  (hw_activation[c]),
        .int9Out (hw_product_i9[r][c])
      );
    
    end
  end

  // ------------------------------------------------------------
  // Combinational next-state logic
  // ------------------------------------------------------------

  always_comb begin

    // Defaults
    t_d           = t_q;
    a_d           = a_q;

    t_we          = 1'b0;
    a_we          = 1'b0;

    valid_o       = fp4_en_i;

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

      FP4_ZZMAC64: begin

        t_we = 1'b1;

        for (int r = 0; r < 8; r++) begin
          for (int c = 0; c < 8; c++) begin
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

      FP4_MAXMAC64: begin

        t_we = 1'b1;
        
        for (int r = 0; r < 8; r++) begin
          for (int c = 0; c < 8; c++) begin
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

      FP4_HWMAC64: begin

        t_we = 1'b1;
        
        for (int r = 0; r < 8; r++) begin
          for (int c = 0; c < 8; c++) begin
        
            hw_sum[r][c] = t_q[r][c] + hw_product_i9[r][c];

            if (hw_sum[r][c] > 17'sd32767)
              t_d[r][c] = 16'sh7fff;
            else if (hw_sum[r][c] < -17'sd32768)
              t_d[r][c] = 16'sh8000;
            else
              t_d[r][c] = hw_sum[r][c][15:0];
        
          end
        end
        
      end 

      // ========================================================
      // ad2MAC64
      //
      // Add two packed int16 bias values T[i][2*j] += rs1[15:0] 
      // and T[i][2*j+1] += rs1[31:16]
      // ========================================================

      FP4_AD2MAC64: begin
        t_we = 1'b1;
        
        ad2_sum_even = t_q[row_idx][col_even_idx] + $signed(op_a_i[15:0]);
        ad2_sum_odd  = t_q[row_idx][col_odd_idx]  + $signed(op_a_i[31:16]);
        
        if (ad2_sum_even > 17'sd32767)
          t_d[row_idx][col_even_idx] = 16'sh7fff;
        else if (ad2_sum_even < -17'sd32768)
          t_d[row_idx][col_even_idx] = 16'sh8000;
        else
          t_d[row_idx][col_even_idx] = ad2_sum_even[15:0];
       
        if (ad2_sum_odd > 17'sd32767)
          t_d[row_idx][col_odd_idx] = 16'sh7fff;
        else if (ad2_sum_odd < -17'sd32768)
          t_d[row_idx][col_odd_idx] = 16'sh8000;
        else
          t_d[row_idx][col_odd_idx] = ad2_sum_odd[15:0];
      end 

      // ========================================================
      // mveMAC64
      //
      // Move even tile entry to rd and clear tile entry
      // ========================================================

      FP4_MVEMAC64: begin

        t_we = 1'b1;
        fp4_result_o = t_q[row_idx][col_even_idx];
        valid_o = 1'b1;
        t_d[row_idx][col_even_idx] = 0;

      end

      // ========================================================
      // mvoMAC64
      // 
      // Move odd tile entry to rd and clear tile entry
      // ========================================================

      FP4_MVOMAC64: begin

        t_we = 1'b1;
        fp4_result_o = t_q[row_idx][col_odd_idx];
        valid_o = 1'b1;
        t_d[row_idx][col_odd_idx] = 0;

      end

      // ========================================================
      // mv2MAC64
      // ========================================================

      FP4_MV2MAC64: begin

        t_we = 1'b1;
        fp4_result_o = {t_q[row_idx][col_odd_idx], t_q[row_idx][col_even_idx]};
        valid_o = 1'b1;
        t_d[row_idx][col_odd_idx] = 0;
        t_d[row_idx][col_even_idx] = 0;
  
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
      // ========================================================

      FP4_ST2MAC64: begin

        t_we = 1'b1;
        // to be done

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

      if (t_we)
        t_q <= t_d;

      if (a_we)
        a_q <= a_d;

    end

  end

endmodule

