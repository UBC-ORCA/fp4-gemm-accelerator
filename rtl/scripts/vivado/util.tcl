proc report_fmax {rpt_file clk_name} {

    # Get WNS
    set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]

    # Get clock freq
    set clk_period_ns [get_property PERIOD [get_clocks $clk_name]]

    set fmax [expr 1.0 / ($clk_period_ns - $wns) * 1000.0]  

    set fmax_rpt_file "$rpt_file"
    set fmax_fd [open $fmax_rpt_file "w"]
    puts $fmax_fd "WNS: $wns ns"
    puts $fmax_fd "Maximum Frequency (FMAX): $fmax MHz"
    close $fmax_fd
}