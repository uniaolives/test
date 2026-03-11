// oca_controller.v
// Orb Communication Array Controller
// Generates timing signals for entangled photon generation and OAM modulation

module oca_controller (
    input  wire        clk_50,
    input  wire        rst_n,
    input  wire [63:0] core_phase,   // from CSU
    input  wire        tx_start,            // initiate transmission
    output reg         laser_pulse,         // trigger femtosecond laser
    output reg [3:0]   oam_value,           // OAM topological charge (l = 0..15)
    output reg         coincidence_gate,    // enable coincidence counter
    output reg         rx_ready
);

    // States
    localparam IDLE       = 3'b000;
    localparam PUMP       = 3'b001;
    localparam OAM_ENCODE = 3'b010;
    localparam TRANSMIT   = 3'b011;
    localparam RX_GATE    = 3'b100;
    localparam RX_WAIT    = 3'b101;

    reg [2:0] state;
    reg [31:0] timer;

    // Timing constants (in cycles at 50 MHz = 20 ns)
    localparam T_PUMP    = 32'd250;       // 5 µs pump pulse
    localparam T_OAM_SET = 32'd10;         // 200 ns OAM settling
    localparam T_GATE    = 32'd50000;      // 1 ms coincidence window

    always @(posedge clk_50 or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            timer <= 32'd0;
            laser_pulse <= 1'b0;
            oam_value <= 4'd0;
            coincidence_gate <= 1'b0;
            rx_ready <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (tx_start) begin
                        // Use core_phase to determine OAM value (entropy source)
                        oam_value <= core_phase[3:0];  // low bits as OAM
                        state <= PUMP;
                        timer <= T_PUMP;
                    end else begin
                        // In receive mode, wait for coincidence gate
                        state <= RX_GATE;
                        timer <= T_GATE;
                    end
                end

                PUMP: begin
                    laser_pulse <= 1'b1;
                    if (timer == 32'd0) begin
                        laser_pulse <= 1'b0;
                        state <= OAM_ENCODE;
                        timer <= T_OAM_SET;
                    end else begin
                        timer <= timer - 1'b1;
                    end
                end

                OAM_ENCODE: begin
                    // OAM value already set, wait settling
                    if (timer == 32'd0) begin
                        state <= TRANSMIT;
                    end else begin
                        timer <= timer - 1'b1;
                    end
                end

                TRANSMIT: begin
                    // Photons are on their way – return to idle
                    state <= IDLE;
                end

                RX_GATE: begin
                    // Open coincidence window
                    coincidence_gate <= 1'b1;
                    timer <= T_GATE;
                    state <= RX_WAIT;
                end

                RX_WAIT: begin
                    if (timer == 32'd0) begin
                        coincidence_gate <= 1'b0;
                        rx_ready <= 1'b1;   // signal that data may be available
                        state <= IDLE;
                    end else begin
                        timer <= timer - 1'b1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
