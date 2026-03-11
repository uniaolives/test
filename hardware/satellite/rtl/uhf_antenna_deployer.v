`timescale 1ns / 1ps

module uhf_antenna_deployer #(
    // Parameters for timing (Assuming a 50MHz clock for this example)
    parameter BURN_TIME_CYCLES = 32'd250_000_000, // 5 seconds of heating
    parameter DEPLOY_TIMEOUT   = 32'd500_000_000  // 10 seconds wait for deployment
)(
    input  wire clk,              // System clock
    input  wire rst_n,            // Active-low asynchronous reset
    input  wire cmd_deploy,       // Command from main bus to initiate deployment
    input  wire limit_switch,     // Physical switch: 1 = Antenna fully deployed

    output reg  burn_wire_en,     // Enables current to the burn resistor
    output reg  [2:0] deploy_status // Telemetry status
);

    // FSM State Encoding
    localparam ST_STOWED    = 3'b000;
    localparam ST_HEATING   = 3'b001;
    localparam ST_WAIT_OPEN = 3'b010;
    localparam ST_DEPLOYED  = 3'b011;
    localparam ST_ERROR     = 3'b100;

    reg [2:0] current_state, next_state;
    reg [31:0] timer;

    // State Register & Timer
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= ST_STOWED;
            timer <= 32'd0;
        end else begin
            current_state <= next_state;

            // Timer management
            if (current_state != next_state) begin
                timer <= 32'd0; // Reset timer on state change
            end else if (current_state == ST_HEATING || current_state == ST_WAIT_OPEN) begin
                timer <= timer + 1'b1;
            end
        end
    end

    // Next State Logic
    always @(*) begin
        // Default assignments
        next_state = current_state;

        case (current_state)
            ST_STOWED: begin
                if (cmd_deploy && !limit_switch)
                    next_state = ST_HEATING;
                else if (limit_switch)
                    next_state = ST_DEPLOYED; // Already deployed
            end

            ST_HEATING: begin
                if (timer >= BURN_TIME_CYCLES)
                    next_state = ST_WAIT_OPEN;
            end

            ST_WAIT_OPEN: begin
                if (limit_switch)
                    next_state = ST_DEPLOYED;
                else if (timer >= DEPLOY_TIMEOUT)
                    next_state = ST_ERROR; // Mechanical failure
            end

            ST_DEPLOYED: begin
                // Terminal state, wait for reset
                next_state = ST_DEPLOYED;
            end

            ST_ERROR: begin
                // Terminal error state. Could add logic to retry.
                if (cmd_deploy) // Allow retry on new command
                    next_state = ST_HEATING;
            end

            default: next_state = ST_STOWED;
        endcase
    end

    // Output Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            burn_wire_en  <= 1'b0;
            deploy_status <= ST_STOWED;
        end else begin
            deploy_status <= current_state; // Map state directly to telemetry

            if (current_state == ST_HEATING)
                burn_wire_en <= 1'b1;
            else
                burn_wire_en <= 1'b0; // Ensure heater is off in all other states
        end
    end

endmodule
