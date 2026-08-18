# Vivado synthesis constraints for cve2_top Fmax exploration.
#
# This is a max-frequency estimation constraint set, not a tapeout/board-ready
# timing spec: only the clock and reset are constrained so `report_timing_summary`
# reports a meaningful WNS on register-to-register paths. I/O timing is left
# unconstrained (Vivado will report it as user-ignored), since the datapath's
# internal Fmax, not board I/O, is what's being measured.
#
# Period is intentionally loose (100 MHz / 10 ns) -- it is not a target to hit,
# it just gives synthesis a clock to build a timing graph against. Read the
# WNS in the post-synth timing summary and compute:
#   Fmax = 1 / (period_ns - WNS_ns) * 1000   (MHz)

create_clock -name clk_i -period 8.000 [get_ports clk_i]

# rst_ni is asynchronous and expected to be synchronized internally;
# excluded from setup/hold analysis at the top-level port.
set_false_path -from [get_ports rst_ni]

create_clock -period 10.000 -name clk -waveform {0.000 5.000}
