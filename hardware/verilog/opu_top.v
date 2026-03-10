module opu_top #(
    parameter DATA_WIDTH = 512,
    parameter ADDR_WIDTH = 128
)(
    input  wire                    sys_clk,
    input  wire                    rst_n,

    // Network Interface
    input  wire [DATA_WIDTH-1:0]   rx_data,
    input  wire                    rx_valid,
    output wire                    rx_ready,

    // Time Reference
    input  wire [63:0]             atomic_time,

    // Status
    output wire                    paradox_alarm
);

    wire [31:0] target_phase;
    wire [DATA_WIDTH-1:0] payload;
    wire retro;

    uqi_parser parser_inst (
        .clk(sys_clk),
        .rst_n(rst_n),
        .s_axis_tdata(rx_data),
        .s_axis_tvalid(rx_valid),
        .s_axis_tready(rx_ready),
        .payload_out(payload),
        .phase_target(target_phase)
    );

    temporal_engine sched_inst (
        .clk(sys_clk),
        .rst_n(rst_n),
        .target_time(payload[63:0]),
        .valid(rx_valid),
        .current_time(atomic_time),
        .retrocausal(retro),
        .ready()
    );

    assign paradox_alarm = retro && (target_phase == 0); // Simplified alarm

endmodule
