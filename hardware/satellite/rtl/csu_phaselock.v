// csu_phaselock.v
// Core Sync Unit: 50 MHz PLL with 70-year epoch counter
// Input: magnetometer analog (simplified as 1-bit reference)

module csu_phaselock (
    input  wire        clk_50,          // 50 MHz system clock
    input  wire        rst_n,
    input  wire        mag_pulse,       // 1 pulse per magnetic reversal (simplified)
    output reg  [63:0] core_phase,      // 64-bit phase counter (0 to 2^64-1)
    output reg         epoch_tick        // pulse every 70 years (calibrated)
);

    // 70 years in seconds = 70 * 365.25 * 24 * 3600 ≈ 2.207e9 s
    // At 50 MHz, one tick = 20 ns. 70 years = 2.207e9 / 20e-9 = 1.1035e17 cycles.
    localparam [63:0] EPOCH_CYCLES = 64'd110350000000000000;

    reg [63:0] counter;

    always @(posedge clk_50 or negedge rst_n) begin
        if (!rst_n) begin
            counter <= 64'd0;
            core_phase <= 64'd0;
            epoch_tick <= 1'b0;
        end else begin
            // Increment phase counter every clock cycle
            counter <= counter + 1'b1;
            core_phase <= counter;

            // Generate epoch tick when counter reaches EPOCH_CYCLES
            if (counter == EPOCH_CYCLES) begin
                epoch_tick <= 1'b1;
                counter <= 64'd0;      // reset for next epoch
            end else begin
                epoch_tick <= 1'b0;
            end

            // Optional: synchronize with magnetic pulse (calibration)
            if (mag_pulse) begin
                // adjust counter to align with core phase (simplified)
                counter <= counter - 64'd1000; // dummy adjustment
            end
        end
    end
endmodule
